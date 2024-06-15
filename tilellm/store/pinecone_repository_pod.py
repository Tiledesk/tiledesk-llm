from tilellm.models.item_model import (MetadataItem,
                                       PineconeIndexingResult
                                       )
from tilellm.tools.document_tool_simple import (get_content_by_url,
                                                get_content_by_url_with_bs,
                                                load_document,
                                                load_from_wikipedia
                                                )

from tilellm.store.pinecone_repository_base import PineconeRepositoryBase

from tilellm.shared import const
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import os

import logging

logger = logging.getLogger(__name__)


class PineconeRepositoryPod(PineconeRepositoryBase):
    async def add_pc_item(self, item):
        """
            Add items to name
            space into Pinecone index
            :param item:
            :return:
            """
        logger.info(item)
        metadata_id = item.id
        source = item.source
        type_source = item.type
        content = item.content
        gpt_key = item.gptkey
        embedding = item.embedding
        namespace = item.namespace
        scrape_type = item.scrape_type
        try:
            await self.delete_pc_ids_namespace(metadata_id=metadata_id, namespace=namespace)
        except Exception as ex:
            logger.warning(ex)
            pass

        emb_dimension = self.get_embeddings_dimension(embedding)

        # default text-embedding-ada-002 1536, text-embedding-3-large 3072, text-embedding-3-small 1536
        oai_embeddings = OpenAIEmbeddings(api_key=gpt_key, model=embedding)
        vector_store = await self.create_pc_index(embeddings=oai_embeddings, emb_dimension=emb_dimension)

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
                    documents = get_content_by_url(source, scrape_type)
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

                    chunks.extend(self.chunk_data(data=[document]))

                # from pprint import pprint
                # pprint(documents)
                logger.debug(documents)

                a = vector_store.from_documents(chunks,
                                                embedding=oai_embeddings,
                                                index_name=const.PINECONE_INDEX,
                                                namespace=namespace,
                                                text_key=const.PINECONE_TEXT_KEY)

                total_tokens, cost = self.calc_embedding_cost(chunks, embedding)
                logger.info(f"chunks: {len(chunks)}, total_tokens: {total_tokens}, cost: {cost: .6f}")

                # from pprint import pprint
                # pprint(documents)
            elif type_source == 'urlbs':
                doc_array = get_content_by_url_with_bs(source)
                chunks = list()
                for doc in doc_array:
                    metadata = MetadataItem(id=metadata_id, source=source, type=type_source, embedding=embedding)
                    document = Document(page_content=doc, metadata=metadata.dict())
                    chunks.append(document)
                # chunks.extend(chunk_data(data=documents))
                total_tokens, cost = self.calc_embedding_cost(chunks, embedding)
                a = vector_store.from_documents(chunks,
                                                embedding=oai_embeddings,
                                                index_name=const.PINECONE_INDEX,
                                                namespace=namespace,
                                                text_key=const.PINECONE_TEXT_KEY)

            else:
                metadata = MetadataItem(id=metadata_id, source=source, type=type_source, embedding=embedding)
                document = Document(page_content=content, metadata=metadata.dict())

                chunks.extend(self.chunk_data(data=[document]))
                total_tokens, cost = self.calc_embedding_cost(chunks, embedding)
                a = vector_store.from_documents(chunks,
                                                embedding=oai_embeddings,
                                                index_name=const.PINECONE_INDEX,
                                                namespace=namespace,
                                                text_key=const.PINECONE_TEXT_KEY)

            pinecone_result = PineconeIndexingResult(id=metadata_id, chunks=len(chunks), total_tokens=total_tokens,
                                                     cost=f"{cost:.6f}")
        except Exception as ex:
            logger.error(repr(ex))
            pinecone_result = PineconeIndexingResult(id=metadata_id, chunks=len(chunks), total_tokens=total_tokens,
                                                     status=400,
                                                     cost=f"{cost:.6f}")
        # {"id": f"{id}", "chunks": f"{len(chunks)}", "total_tokens": f"{total_tokens}", "cost": f"{cost:.6f}"}
        return pinecone_result

    async def delete_pc_ids_namespace(self, metadata_id: str, namespace: str):
        """
        Delete from pinecone items
        :param metadata_id:
        :param namespace:
        :return:
        """
        # FIXME problema con namespace di cardinalità superiore a 10000

        import pinecone
        try:
            pc = pinecone.Pinecone(
                api_key=const.PINECONE_API_KEY
            )

            host = pc.describe_index(const.PINECONE_INDEX).host
            index = pc.Index(name=const.PINECONE_INDEX, host=host)
            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            describe = index.describe_index_stats()
            logger.debug(describe)
            namespaces = describe.get("namespaces", {})
            total_vectors = 1
            # batch_size = 100

            offset = 0

            if namespaces:
                if namespace in namespaces.keys():
                    total_vectors = namespaces.get(namespace).get('vector_count')

            logger.debug(total_vectors)

            batch_size = min([total_vectors, 10000])
            while offset < total_vectors:

                pc_res = index.query(
                    vector=[0] * 1536,  # [0,0,0,0......0]
                    top_k=batch_size,
                    filter={"id": {"$eq": metadata_id}},
                    namespace=namespace,
                    include_values=False,
                    include_metadata=False
                )
                matches = pc_res.get('matches')

                ids = [obj.get('id') for obj in matches]

                if offset == 0 and not ids:
                    raise IndexError(f"Empty list for {metadata_id} and namespace {namespace}")
                elif offset > 0 and not ids:
                    break

                index.delete(
                    ids=ids,
                    namespace=namespace)

                if batch_size < 10000:
                    break
                else:
                    offset += len(ids)

        except Exception as ex:
            # logger.error(ex)
            raise ex


