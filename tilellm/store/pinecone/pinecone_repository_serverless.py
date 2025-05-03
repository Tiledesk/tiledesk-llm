import datetime
import torch

from langchain_core.utils import batch_iterate

from tilellm.models.item_model import (MetadataItem,
                                       IndexingResult, Engine, ItemSingle
                                       )

from tilellm.tools.document_tools import (get_content_by_url,
                                          get_content_by_url_with_bs,
                                          load_document,
                                          load_from_wikipedia
                                          )

from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase
#from tilellm.shared.utility import inject_embedding
from tilellm.shared.embedding_factory import inject_embedding
from pinecone_text.sparse import SpladeEncoder

from langchain_core.documents import Document

import uuid



import logging

from tilellm.tools.sparse_encoders import TiledeskSparseEncoders

logger = logging.getLogger(__name__)


class PineconeRepositoryServerless(PineconeRepositoryBase):


    @inject_embedding()
    async def add_pc_item(self, item:ItemSingle, embedding_obj=None, embedding_dimension=None):
        """
            Add items to name
            space into Pinecone index
            :param item:
            :param embedding_obj:
            :param embedding_dimension:
            :return:
            """
        logger.info(item)

        await self.delete_pc_ids_namespace(engine=item.engine,
                                           metadata_id=item.id,
                                           namespace=item.namespace)

        vector_store = await self.create_vector_store(engine=item.engine,
                                                      embedding_obj=embedding_obj,
                                                      embedding_dimension=embedding_dimension,
                                                      metric="cosine")

        chunks = []
        total_tokens = 0
        cost = 0

        try:

            if item.type in ['url', 'pdf', 'docx', 'txt']:
                documents = await self.fetch_documents(type_source=item.type,
                                                  source=item.source,
                                                  scrape_type=item.scrape_type,
                                                  parameters_scrape_type_4=item.parameters_scrape_type_4)
                chunks = self.chunk_documents(item=item,
                                              documents=documents,
                                              embeddings=embedding_obj
                                              )
            else:
                metadata = MetadataItem(id=item.id,
                                        source=item.source,
                                        type=item.type,
                                        embedding=item.embedding).model_dump()
                documents = await self.process_contents(type_source=item.type,
                                                        source=item.source,
                                                        metadata=metadata,
                                                        content=item.content)

                chunks.extend(self.chunk_data_extended(
                    data=[documents[0]],
                    chunk_size=item.chunk_size,
                    chunk_overlap=item.chunk_overlap,
                    semantic=item.semantic_chunk,
                    embeddings=embedding_obj,
                    breakpoint_threshold_type=item.breakpoint_threshold_type)
                )

            logger.debug(documents)

            total_tokens, cost = self.calc_embedding_cost(chunks, item.embedding)

            returned_ids = await self.upsert_vector_store(vector_store=vector_store,
                                                          chunks=chunks,
                                                          metadata_id=item.id,
                                                          namespace=item.namespace)

            logger.debug(returned_ids)

            return IndexingResult(id=item.id,
                                  chunks=len(chunks),
                                  total_tokens=total_tokens,
                                  cost=f"{cost:.6f}")

        except Exception as ex:
            import traceback
            traceback.print_exc()
            logger.error(repr(ex))
            return IndexingResult(id=item.id,
                                  chunks=len(chunks),
                                  total_tokens=total_tokens,
                                  status=400,
                                  cost=f"{cost:.6f}")


    @inject_embedding()
    async def add_pc_item_hybrid(self, item, embedding_obj=None, embedding_dimension=None):
        """
        Add item for hybrid search
        :param item:
        :param embedding_obj:
        :param embedding_dimension:
        :return:
        """
        logger.info(item)
        print(type(embedding_obj))
        await self.delete_pc_ids_namespace(engine=item.engine,
                                           metadata_id=item.id,
                                           namespace=item.namespace)

        vector_store = await self.create_vector_store(engine=item.engine,
                                                      embedding_obj=embedding_obj,
                                                      embedding_dimension=embedding_dimension,
                                                      metric="dotproduct")

        # default text-embedding-ada-002 1536, text-embedding-3-large 3072, text-embedding-3-small 1536

        chunks = []
        total_tokens = 0
        cost = 0

        try:
            if item.type in ['url', 'pdf', 'docx', 'txt']:
                documents = await self.fetch_documents(type_source = item.type,
                                                  source=item.source,
                                                  scrape_type=item.scrape_type,
                                                  parameters_scrape_type_4=item.parameters_scrape_type_4)

                chunks = self.chunk_documents(item=item,
                                              documents=documents,
                                              embeddings=embedding_obj
                                              )
            else:
                metadata = MetadataItem(id=item.id,
                                        source=item.source,
                                        type=item.type,
                                        embedding=item.embedding).model_dump()
                documents = await self.process_contents(type_source=item.type,
                                                        source=item.source,
                                                        metadata=metadata,
                                                        content=item.content)
                chunks.extend(self.chunk_data_extended(
                    data=[documents[0]],
                    chunk_size=item.chunk_size,
                    chunk_overlap=item.chunk_overlap,
                    semantic=item.semantic_chunk,
                    embeddings=embedding_obj,
                    breakpoint_threshold_type=item.breakpoint_threshold_type)
                )

            contents = [chunk.page_content for chunk in chunks]
            total_tokens, cost = self.calc_embedding_cost(chunks, item.embedding)

            sparse_encoder = TiledeskSparseEncoders(item.sparse_encoder)
            doc_sparse_vectors = sparse_encoder.encode_documents(contents)

            indice = vector_store.get_pinecone_index(item.engine.index_name, pinecone_api_key=item.engine.apikey)

            await self.upsert_vector_store_hybrid(indice,
                                                  contents,
                                                  chunks,
                                                  item.id,
                                                  namespace = item.namespace,
                                                  engine=item.engine,
                                                  embeddings=embedding_obj,
                                                  sparse_vectors=doc_sparse_vectors)

            return IndexingResult(id=item.id, chunks=len(chunks), total_tokens=total_tokens,
                                  cost=f"{cost:.6f}")

        except Exception as ex:
            import traceback
            traceback.print_exc()
            logger.error(repr(ex))
            return IndexingResult(id=item.id, chunks=len(chunks), total_tokens=total_tokens,
                                             status=400,
                                             cost=f"{cost:.6f}")

        #return pinecone_result

    async def delete_pc_ids_namespace(self, engine, metadata_id: str, namespace: str):

        import pinecone

        try:

            pc = pinecone.Pinecone(
                api_key= engine.apikey #const.PINECONE_API_KEY
            )

            host = pc.describe_index(engine.index_name).host#const.PINECONE_INDEX).host
            index = pc.Index(name=engine.index_name , host=host)#const.PINECONE_INDEX, host=host)

            describe = index.describe_index_stats()
            logger.debug(describe)
            # namespaces = describe.get("namespaces", {})
            # logger.debug(namespaces)

            for ids in index.list(prefix=f"{metadata_id}#", namespace=namespace):
                logger.info(f"deleted ids: {ids}")  # ['doc1#chunk1', 'doc1#chunk2', 'doc1#chunk3']
                index.delete(ids=ids, namespace=namespace)

        except Exception as ex:
            # logger.error(ex)
            logger.warning(ex)
            #raise ex


    #async def delete_existing_items(self, engine: Engine, metadata_id: str, namespace: str):
    #    try:
    #        await self.delete_pc_ids_namespace(engine=engine, metadata_id=metadata_id, namespace=namespace)
    #    except Exception as ex:
    #        logger.warning(ex)

    async def create_vector_store(self, engine: Engine, embedding_obj, embedding_dimension: int, metric: str):
        engine.metric = metric
        return await self.create_pc_index(engine=engine, embeddings=embedding_obj, emb_dimension=embedding_dimension)

    @staticmethod
    async def fetch_documents(type_source, source, scrape_type, parameters_scrape_type_4):
        if type_source in ['url', 'txt']:
            return await get_content_by_url(source,
                                            scrape_type,
                                            parameters_scrape_type_4=parameters_scrape_type_4)
        return load_document(source, type_source)

    @staticmethod
    def process_document_metadata(document, metadata):
        document.metadata.update(metadata)
        document.metadata['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
        for key, value in document.metadata.items():
            if isinstance(value, list) and all(item is None for item in value):
                document.metadata[key] = [""]
            elif value is None:
                document.metadata[key] = ""
        return document

    def chunk_documents(self, item, documents, embeddings):
        chunks = []
        for document in documents:
            document.metadata["id"] = item.id
            document.metadata["source"] = item.source
            document.metadata["type"] = item.type
            document.metadata["embedding"] = item.embedding
            processed_document = self.process_document_metadata(document, document.metadata)
            chunks.extend(self.chunk_data_extended(
                data=[processed_document],
                chunk_size=item.chunk_size,
                chunk_overlap=item.chunk_overlap,
                semantic=item.semantic_chunk,
                embeddings=embeddings,
                breakpoint_threshold_type=item.breakpoint_threshold_type)
            )
        return chunks

    @staticmethod
    async def process_contents(type_source, source, metadata, content):
        if type_source == 'urlbs':
            doc_array = get_content_by_url_with_bs(source)
            return [Document(page_content=doc, metadata=MetadataItem(**metadata).model_dump()) for doc in doc_array]

        document = Document(page_content=content, metadata=MetadataItem(**metadata).model_dump())
        return [document]

    @staticmethod
    async def upsert_vector_store_hybrid(indice, contents, chunks, metadata_id, engine, namespace, embeddings, sparse_vectors):
        embedding_chunk_size = 1000
        batch_size: int = 32

        ids = [f"{metadata_id}#{uuid.uuid4().hex}" for _ in range(len(chunks))]
        metadatas = [{**chunk.metadata, engine.text_key: chunk.page_content} for chunk in chunks]
        async_req = True

        for i in range(0, len(contents), embedding_chunk_size):
            chunk_texts = contents[i: i + embedding_chunk_size]
            chunk_ids = ids[i: i + embedding_chunk_size]
            chunk_metadatas = metadatas[i: i + embedding_chunk_size]
            embedding_values = embeddings.embed_documents(chunk_texts)#embeddings[i: i + embedding_chunk_size]
            #print(len(embedding_values[1]))
            sparse_values = sparse_vectors[i: i + embedding_chunk_size]

            vector_tuples = [
                {'id': idr,
                 'values': embedding,
                 'metadata': chunk,
                 'sparse_values': sparse_value}
                for idr, embedding, chunk, sparse_value in
                zip(chunk_ids, embedding_values, chunk_metadatas, sparse_values)
            ]

            if async_req:
                async_res = [
                    indice.upsert(vectors=batch_vector_tuples,
                                  namespace=namespace,
                                  async_req=async_req)
                    for batch_vector_tuples in batch_iterate(batch_size, vector_tuples)
                ]
                [res.get() for res in async_res]
            else:
                indice.upsert(vectors=vector_tuples,
                              namespace=namespace,
                              async_req=async_req)

    @staticmethod
    async def upsert_vector_store(vector_store, chunks, metadata_id, namespace):
        ids = [f"{metadata_id}#{uuid.uuid4().hex}" for _ in range(len(chunks))]
        returned_ids = await vector_store.aadd_documents(chunks, namespace=namespace, ids=ids)