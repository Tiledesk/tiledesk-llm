from tilellm.models.item_model import (MetadataItem,
                                       PineconeIndexingResult
                                       )
from tilellm.tools.document_tool_simple import (get_content_by_url,
                                                get_content_by_url_with_bs,
                                                load_document,
                                                load_from_wikipedia
                                                )

from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase
from tilellm.shared.utility import inject_embedding

from tilellm.shared import const
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import uuid

import os

import logging

logger = logging.getLogger(__name__)


class PineconeRepositoryServerless(PineconeRepositoryBase):
    @inject_embedding()
    async def add_pc_item(self, item, embedding_obj=None, embedding_dimension=None):
        """
            Add items to name
            space into Pinecone index
            :param item:
            :param embedding_obj:
            :param embedding_dimension:
            :return:
            """
        logger.info(item)

        metadata_id = item.id
        source = item.source
        type_source = item.type
        content = item.content
        #gpt_key = item.gptkey
        embedding = item.embedding
        namespace = item.namespace
        semantic_chunk = item.semantic_chunk
        breakpoint_threshold_type = item.breakpoint_threshold_type
        scrape_type = item.scrape_type
        chunk_size = item.chunk_size
        chunk_overlap = item.chunk_overlap
        parameters_scrape_type_4 = item.parameters_scrape_type_4
        try:
            await self.delete_pc_ids_namespace(metadata_id=metadata_id, namespace=namespace)
        except Exception as ex:
            logger.warning(ex)
            pass

        emb_dimension = embedding_dimension # self.get_embeddings_dimension(embedding)

        # default text-embedding-ada-002 1536, text-embedding-3-large 3072, text-embedding-3-small 1536
        oai_embeddings = embedding_obj # OpenAIEmbeddings(api_key=gpt_key, model=embedding)
        vector_store = await self.create_pc_index(embeddings=oai_embeddings, emb_dimension=emb_dimension)
        # textprova ="test degli embeddings di voyage"
        # query_result = oai_embeddings.embed_query(textprova)
        # print(f"len: {len(query_result)}")
        # print(query_result)
        # raise Exception
        chunks = []
        total_tokens = 0
        cost = 0

        try:
            if (type_source == 'url' or
                    type_source == 'pdf' or
                    type_source == 'docx' or
                    type_source == 'txt'):

                documents = []
                if type_source == 'url' or type_source == 'txt':
                    documents = await get_content_by_url(source,
                                                         scrape_type,
                                                         parameters_scrape_type_4=parameters_scrape_type_4)
                else:  # elif type_source == 'pdf' or 'docx' or 'txt':
                    documents = load_document(source, type_source)

                for document in documents:

                    document.metadata["id"] = metadata_id
                    document.metadata["source"] = source
                    document.metadata["type"] = type_source
                    document.metadata["embedding"] = embedding

                    for key, value in document.metadata.items():
                        if isinstance(value, list) and all(item is None for item in value):
                            document.metadata[key] = [""]
                        elif value is None:
                            document.metadata[key] = ""

                    chunks.extend(self.chunk_data_extended(data=[document],
                                                           chunk_size=chunk_size,
                                                           chunk_overlap=chunk_overlap,
                                                           semantic=semantic_chunk,
                                                           embeddings=oai_embeddings,
                                                           breakpoint_threshold_type=breakpoint_threshold_type
                                                           )
                                 )
                # from pprint import pprint
                # pprint(documents)
                logger.debug(documents)

                # from pprint import pprint
                # pprint(chunks)

                a = vector_store.from_documents(chunks,
                                                embedding=oai_embeddings,
                                                index_name=const.PINECONE_INDEX,
                                                namespace=namespace,
                                                text_key=const.PINECONE_TEXT_KEY,
                                                ids=[f"{metadata_id}#{uuid.uuid4().hex}" for i in range(len(chunks))])

                total_tokens, cost = self.calc_embedding_cost(chunks, embedding)
                logger.info(f"chunks: {len(chunks)}, total_tokens: {total_tokens}, cost: {cost: .6f}")

                # from pprint import pprint
                # pprint(documents)
            elif type_source == 'urlbs':
                doc_array = get_content_by_url_with_bs(source)
                chunks = list()
                for doc in doc_array:
                    metadata = MetadataItem(id=metadata_id, source=source, type=type_source, embedding=embedding)

                    document = Document(page_content=doc, metadata=metadata.model_dump()) #dict())

                    chunks.append(document)
                # chunks.extend(chunk_data(data=documents))
                total_tokens, cost = self.calc_embedding_cost(chunks, embedding)
                a = vector_store.from_documents(chunks,
                                                embedding=oai_embeddings,
                                                index_name=const.PINECONE_INDEX,
                                                namespace=namespace,
                                                text_key=const.PINECONE_TEXT_KEY,
                                                ids=[f"{metadata_id}#{uuid.uuid4().hex}" for i in range(len(chunks))]
                                                )

            else:
                metadata = MetadataItem(id=metadata_id, source=source, type=type_source, embedding=embedding)
                document = Document(page_content=content, metadata=metadata.model_dump()) #tolto dict()

                chunks.extend(self.chunk_data_extended(data=[document],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       semantic=semantic_chunk,
                                                       embeddings=oai_embeddings,
                                                       breakpoint_threshold_type=breakpoint_threshold_type
                                                       )
                              )
                total_tokens, cost = self.calc_embedding_cost(chunks, embedding)
                a = vector_store.from_documents(chunks,
                                                embedding=oai_embeddings,
                                                index_name=const.PINECONE_INDEX,
                                                namespace=namespace,
                                                text_key=const.PINECONE_TEXT_KEY,
                                                ids=[f"{metadata_id}#{uuid.uuid4().hex}" for i in range(len(chunks))]
                                                )

            pinecone_result = PineconeIndexingResult(id=metadata_id, chunks=len(chunks), total_tokens=total_tokens,
                                                     cost=f"{cost:.6f}")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            logger.error(repr(ex))
            pinecone_result = PineconeIndexingResult(id=metadata_id, chunks=len(chunks), total_tokens=total_tokens,
                                                     status=400,
                                                     cost=f"{cost:.6f}")
        # {"id": f"{id}", "chunks": f"{len(chunks)}", "total_tokens": f"{total_tokens}", "cost": f"{cost:.6f}"}
        return pinecone_result

    async def delete_pc_ids_namespace(self, metadata_id: str, namespace: str):

        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=const.PINECONE_API_KEY
            )

            host = pc.describe_index(const.PINECONE_INDEX).host
            index = pc.Index(name=const.PINECONE_INDEX, host=host)

            describe = index.describe_index_stats()
            logger.debug(describe)
            # namespaces = describe.get("namespaces", {})
            # logger.debug(namespaces)

            for ids in index.list(prefix=f"{metadata_id}#", namespace=namespace):
                logger.info(f"deleted ids: {ids}")  # ['doc1#chunk1', 'doc1#chunk2', 'doc1#chunk3']
                index.delete(ids=ids, namespace=namespace)

        except Exception as ex:
            # logger.error(ex)
            raise ex

