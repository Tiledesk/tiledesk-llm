from tilellm.models.item_model import (MetadataItem,
                                       IndexingResult, Engine
                                       )
from tilellm.shared.utility import inject_embedding
from tilellm.tools.document_tools import (get_content_by_url,
                                          get_content_by_url_with_bs,
                                          load_document,
                                          load_from_wikipedia
                                          )

from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase

from tilellm.shared import const
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import os

import logging

logger = logging.getLogger(__name__)


class PineconeRepositoryPod(PineconeRepositoryBase):
    @inject_embedding()
    async def add_pc_item(self, item, embedding_obj=None, embedding_dimension=None) -> IndexingResult:
        """
            Add items to name
            space into Pinecone index
            :param item:
            :param embedding_obj:
            :param embedding_dimension:
            :return: PineconeIndexingResult
            """
        logger.info(item)
        metadata_id = item.id
        source = item.source
        type_source = item.type
        content = item.content
        gpt_key = item.gptkey
        embedding = item.embedding
        namespace = item.namespace
        semantic_chunk = item.semantic_chunk
        breakpoint_threshold_type = item.breakpoint_threshold_type
        scrape_type = item.scrape_type
        chunk_size = item.chunk_size
        chunk_overlap = item.chunk_overlap
        parameters_scrape_type_4 = item.parameters_scrape_type_4
        engine = item.engine
        try:
            await self.delete_pc_ids_namespace(engine=engine, metadata_id=metadata_id, namespace=namespace)
        except Exception as ex:
            logger.warning(ex)
            pass

        emb_dimension = embedding_dimension # self.get_embeddings_dimension(embedding)

        # default text-embedding-ada-002 1536, text-embedding-3-large 3072, text-embedding-3-small 1536
        oai_embeddings = embedding_obj # OpenAIEmbeddings(api_key=gpt_key, model=embedding)
        vector_store = await self.create_pc_index(engine=engine, embeddings=oai_embeddings, emb_dimension=emb_dimension)
        # print(f"=========== POD {type(vector_store)}")
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
                else:  # type_source == 'pdf' or 'docx' or 'txt':
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

                a = vector_store.add_documents(chunks,
                                               #embedding=oai_embeddings,
                                               #index_name=const.PINECONE_INDEX,
                                               namespace=namespace,
                                               #text_key=const.PINECONE_TEXT_KEY
                                               )

                total_tokens, cost = self.calc_embedding_cost(chunks, embedding)
                logger.info(f"chunks: {len(chunks)}, total_tokens: {total_tokens}, cost: {cost: .6f}")

                # from pprint import pprint
                # pprint(documents)
            elif type_source == 'urlbs':
                doc_array = get_content_by_url_with_bs(source)
                chunks = list()
                for doc in doc_array:
                    metadata = MetadataItem(id=metadata_id, source=source, type=type_source, embedding=embedding)
                    document = Document(page_content=doc, metadata=metadata.model_dump())
                    chunks.append(document)
                # chunks.extend(chunk_data(data=documents))
                total_tokens, cost = self.calc_embedding_cost(chunks, embedding)
                a = vector_store.add_documents(chunks,
                                               #embedding=oai_embeddings,
                                               # index_name=const.PINECONE_INDEX,
                                               namespace=namespace,
                                               # text_key=const.PINECONE_TEXT_KEY
                                               )

            else:
                metadata = MetadataItem(id=metadata_id, source=source, type=type_source, embedding=embedding)
                document = Document(page_content=content, metadata=metadata.dict())

                chunks.extend(self.chunk_data_extended(data=[document],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       semantic=semantic_chunk,
                                                       embeddings=oai_embeddings,
                                                       breakpoint_threshold_type=breakpoint_threshold_type
                                                       )
                              )
                total_tokens, cost = self.calc_embedding_cost(chunks, embedding)
                a = vector_store.add_documents(chunks,
                                               #embedding=oai_embeddings,
                                               # index_name=const.PINECONE_INDEX,
                                                namespace=namespace,
                                               # text_key=const.PINECONE_TEXT_KEY
                                               )

            pinecone_result = IndexingResult(id=metadata_id, chunks=len(chunks), total_tokens=total_tokens,
                                             cost=f"{cost:.6f}")
        except Exception as ex:
            logger.error(repr(ex))
            pinecone_result = IndexingResult(id=metadata_id, chunks=len(chunks), total_tokens=total_tokens,
                                             status=400,
                                             cost=f"{cost:.6f}")
        # {"id": f"{id}", "chunks": f"{len(chunks)}", "total_tokens": f"{total_tokens}", "cost": f"{cost:.6f}"}
        return pinecone_result

    @inject_embedding()
    async def add_pc_item_hybrid(self, item, embedding_obj=None, embedding_dimension=None):
        pass


    async def delete_pc_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str):
        """
        Delete from pinecone items
        :param engine:
        :param metadata_id:
        :param namespace:
        :return:
        """
        # print(f"DELETE ID FROM NAMESPACE api {engine.apikey} index_name {engine.index_name}")
        import pinecone
        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey # const.PINECONE_API_KEY
            )

            host = pc.describe_index(engine.index_name).host  # const.PINECONE_INDEX).host
            index = pc.Index(name=engine.index_name, host=host)  # const.PINECONE_INDEX, host=host)
            describe = index.describe_index_stats()
            logger.debug(describe)
            namespaces = describe.get("namespaces", {})
            total_vectors = 1

            if namespaces:
                if namespace in namespaces.keys():
                    total_vectors = namespaces.get(namespace).get('vector_count')

            logger.debug(total_vectors)

            index.delete(
                filter={"id": {"$eq": metadata_id}},
                namespace=namespace)

        except Exception as ex:
            # logger.error(ex)
            raise ex


