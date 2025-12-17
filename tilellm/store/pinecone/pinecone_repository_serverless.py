import datetime
import uuid

from tilellm.models.schemas import (IndexingResult,
                                    RetrievalChunksResult
                                    )
from tilellm.models import (MetadataItem,
                            Engine,
                            ItemSingle,
                            QuestionAnswer
                            )
from tilellm.shared.embeddings.embedding_client_manager import inject_embedding_qa_async_optimized
from tilellm.shared.sparse_util import hybrid_score_norm

from tilellm.tools.document_tools import (get_content_by_url,
                                          get_content_by_url_with_bs,
                                          load_document,
                                          )

from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase

from tilellm.shared.embedding_factory import inject_embedding

from langchain_core.documents import Document


import logging

from tilellm.tools.sparse_encoders import TiledeskSparseEncoders

logger = logging.getLogger(__name__)


class PineconeRepositoryServerless(PineconeRepositoryBase):


    async def perform_hybrid_search(self, question_answer, index, dense_vector, sparse_vector):
        dense, sparse = hybrid_score_norm(dense_vector, sparse_vector, alpha=question_answer.alpha)

        results = await index.query(
            top_k=question_answer.top_k,
            vector=dense,
            sparse_vector=sparse,
            namespace=question_answer.namespace,
            include_metadata=True
        )
        await index.close()

        return results


    async def initialize_embeddings_and_index(self, question_answer, llm_embeddings, emb_dimension=None, embedding_config_key=None):
        emb_dimension = await self.get_embeddings_dimension(question_answer.embedding)
        sparse_encoder = TiledeskSparseEncoders(question_answer.sparse_encoder)
        vector_store = await self.create_index(question_answer.engine, llm_embeddings, emb_dimension, embedding_config_key)
        index = await vector_store.async_index

        return emb_dimension, sparse_encoder, index

    @inject_embedding_qa_async_optimized()
    async def get_chunks_from_repo(self, question_answer: QuestionAnswer, embedding_obj=None, embedding_dimension=None):
        """

        :param question_answer:
        :param embedding_obj:
        :param embedding_dimension:
        :return:
        """
        try:
            vector_store = await self.create_index(engine=question_answer.engine,
                                                   embeddings=embedding_obj,
                                                   emb_dimension=embedding_dimension)

            start_time = datetime.datetime.now() if question_answer.debug else 0

            if question_answer.search_type == 'hybrid':
                emb_dimension = await self.get_embeddings_dimension(question_answer.embedding)
                logger.debug(f"emb dimension {emb_dimension}")
                sparse_encoder = TiledeskSparseEncoders(question_answer.sparse_encoder)
                index = await vector_store.async_index
                sparse_vector = sparse_encoder.encode_queries(question_answer.question)
                dense_vector = await embedding_obj.aembed_query(question_answer.question)
                results = []
                #async with index as index:
                    # Perform hybrid search

                query_response = await self.perform_hybrid_search(question_answer, index, dense_vector, sparse_vector)

                for doc in query_response.matches:
                    doc_id = doc['id']
                    # Creiamo una copia dei metadati per evitare modifiche indesiderate
                    # e rimuoviamo 'text' che diventerà page_content
                    doc_metadata = doc['metadata'].copy()
                    page_content = doc_metadata.pop('text', '')  # Rimuove 'text' e lo usa per page_content

                    # Crea un'istanza di Document e aggiungila all'array
                    document = Document(
                        id=doc_id,
                        metadata=doc_metadata,
                        page_content=page_content
                    )
                    results.append(document)
            else:
                results = await vector_store.asearch(query=question_answer.question,
                                                     search_type=question_answer.search_type,
                                                     k=question_answer.top_k,
                                                     namespace=question_answer.namespace
                                                     )

            end_time = datetime.datetime.now() if question_answer.debug else 0
            duration = (end_time - start_time).total_seconds() if question_answer.debug else 0.0


            retrieval = RetrievalChunksResult(success=True,
                                              namespace=question_answer.namespace,
                                              chunks=[chunk.page_content for chunk in results],
                                              metadata=[chunk.metadata for chunk in results],
                                              error_message=None,
                                              duration=duration
                                              )

            return retrieval
        except Exception as ex:

            logger.error(ex)

            raise ex


    @inject_embedding()
    async def add_item(self, item:ItemSingle, embedding_obj=None, embedding_dimension=None):
        """
            Add items to name
            space into Pinecone index
            :param item:
            :param embedding_obj:
            :param embedding_dimension:
            :return:
            """
        logger.info(item)
        try:
            await self.delete_ids_namespace(engine=item.engine,
                                            metadata_id=item.id,
                                            namespace=item.namespace)
        except Exception as ex:
            logger.warning(ex)
            pass

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
                                                       parameters_scrape_type_4=item.parameters_scrape_type_4,
                                                       browser_headers=item.browser_headers
                                                       )

                chunks = await self.chunk_documents(item=item,
                                              documents=documents,
                                              embeddings=embedding_obj
                                              )
                #print(f"chunks {chunks}")
            else:
                metadata = MetadataItem(id=item.id,
                                        source=item.source,
                                        type=item.type,
                                        embedding=str(item.embedding),
                                        namespace=None).model_dump(exclude_none=False)
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

            if len(chunks) == 0:
                return IndexingResult(id=item.id,
                                      chunks=0,
                                      total_tokens=0,
                                      cost="0.000000",
                                      error="No chunks generated from source")

            total_tokens, cost = self.calc_embedding_cost(chunks, item.embedding)

            returned_ids = await self.upsert_vector_store(vector_store=vector_store,
                                                          chunks=chunks,
                                                          metadata_id=item.id,
                                                          namespace=item.namespace)

            logger.debug(returned_ids)

            #async with vector_store.async_index as index:
            #    await index.close()

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
                                  cost=f"{cost:.6f}",
                                  error=repr(ex))


    @inject_embedding()
    async def add_item_hybrid(self, item:ItemSingle, embedding_obj=None, embedding_dimension=None):
        """
        Add item for hybrid search
        :param item:
        :param embedding_obj:
        :param embedding_dimension:
        :return:
        """
        logger.info(item)

        try:
            await self.delete_ids_namespace(engine=item.engine,
                                           metadata_id=item.id,
                                           namespace=item.namespace)
        except Exception as ex:
            logger.warning(ex)
            pass

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
                                                       parameters_scrape_type_4=item.parameters_scrape_type_4,
                                                       browser_headers=item.browser_headers)

                chunks = await self.chunk_documents(item=item,
                                              documents=documents,
                                              embeddings=embedding_obj
                                              )
            else:
                metadata = MetadataItem(id=item.id,
                                        source=item.source,
                                        type=item.type,
                                        embedding=str(item.embedding)).model_dump()
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

            if len(chunks) == 0:
                return IndexingResult(id=item.id,
                                      chunks=0,
                                      total_tokens=0,
                                      cost="0.000000",
                                      error="No chunks generated from source")

            contents = [chunk.page_content for chunk in chunks]
            total_tokens, cost = self.calc_embedding_cost(chunks, item.embedding)

            sparse_encoder = TiledeskSparseEncoders(item.sparse_encoder)

            doc_sparse_vectors = sparse_encoder.encode_documents(contents, batch_size=item.hybrid_batch_size)

            #indice = vector_store.async_index #index #get_pinecone_index(item.engine.index_name, pinecone_api_key=item.engine.apikey)
            idx = await vector_store.async_index
            await self.upsert_vector_store_hybrid(idx,
                                                  contents,
                                                  chunks,
                                                  item.id,
                                                  namespace = item.namespace,
                                                  engine=item.engine,
                                                  embeddings=embedding_obj,
                                                  sparse_vectors=doc_sparse_vectors,
                                                  doc_batch_size=item.doc_batch_size)

            await idx.close()
            return IndexingResult(id=item.id, chunks=len(chunks), total_tokens=total_tokens,
                                  cost=f"{cost:.6f}")

        except Exception as ex:
            import traceback
            traceback.print_exc()
            logger.error(repr(ex))
            idx = await vector_store.async_index
            await idx.close()
            return IndexingResult(id=item.id, chunks=len(chunks), total_tokens=total_tokens,
                                             status=400,
                                             cost=f"{cost:.6f}",
                                             error=repr(ex))

        #return pinecone_result

    async def delete_ids_namespace(self, engine, metadata_id: str, namespace: str):

        import pinecone

        try:

            pc = pinecone.Pinecone(
                api_key= engine.apikey.get_secret_value() #const.PINECONE_API_KEY
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
        return await self.create_index(engine=engine, embeddings=embedding_obj, emb_dimension=embedding_dimension)

    @staticmethod
    async def fetch_documents(type_source, source, scrape_type, parameters_scrape_type_4, browser_headers):
        if type_source in ['url', 'txt']:
            return await get_content_by_url(source,
                                            scrape_type,
                                            parameters_scrape_type_4=parameters_scrape_type_4,
                                            browser_headers=browser_headers)
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

    async def chunk_documents(self, item, documents, embeddings):
        logger.debug(f"in chunk_documents item: {item} \n documents {documents} emb: {embeddings}")
        chunks = []
        for document in documents:

            document.metadata["id"] = item.id
            document.metadata["source"] = item.source
            document.metadata["type"] = item.type
            document.metadata["embedding"] = str(item.embedding)
            processed_document = self.process_document_metadata(document, document.metadata)
            chunks.extend(self.chunk_data_extended(
                data=[processed_document],
                chunk_size=item.chunk_size,
                chunk_overlap=item.chunk_overlap,
                semantic=item.semantic_chunk,
                embeddings=embeddings,
                breakpoint_threshold_type=item.breakpoint_threshold_type)
            )
        #from pprint import pprint
        #pprint(chunks)
        return chunks

    @staticmethod
    async def process_contents(type_source, source, metadata, content):
        if type_source == 'urlbs':
            doc_array = get_content_by_url_with_bs(source)
            return [Document(page_content=doc, metadata=MetadataItem(**metadata).model_dump()) for doc in doc_array]

        document = Document(page_content=content, metadata=MetadataItem(**metadata).model_dump(exclude_none=True))
        return [document]

    @staticmethod
    async def upsert_vector_store_hybrid(indice, contents, chunks, metadata_id, engine, namespace, embeddings, sparse_vectors, doc_batch_size=100):
        # Ridotto da 1000 a 100 per evitare il limite di 300000 token per richiesta di OpenAI
        embedding_chunk_size = doc_batch_size
        #batch_size: int = 32

        ids = [f"{metadata_id}#{uuid.uuid4().hex}" for _ in range(len(chunks))]
        metadatas = [{**chunk.metadata, engine.text_key: chunk.page_content} for chunk in chunks]
        async_req = True

        for i in range(0, len(contents), embedding_chunk_size):
            chunk_texts = contents[i: i + embedding_chunk_size]
            chunk_ids = ids[i: i + embedding_chunk_size]
            chunk_metadatas = metadatas[i: i + embedding_chunk_size]
            embedding_values = await embeddings.aembed_documents(chunk_texts)#embeddings[i: i + embedding_chunk_size]
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
                # print(type(indice))
                resp = await indice.upsert(vectors=vector_tuples,
                                    namespace=namespace,
                                    batch_size=50)

                #async_res = [
                #    await indice.upsert(vectors=batch_vector_tuples,
                #                  namespace=namespace)
                #    for batch_vector_tuples in batch_iterate(batch_size, vector_tuples)
                #]
                #[res.get() for res in async_res]
                logger.info(f"response upsert: {resp}")
            else:
                await indice.upsert(vectors=vector_tuples,
                              namespace=namespace,
                              async_req=async_req)

    @staticmethod
    async def upsert_vector_store(vector_store, chunks, metadata_id, namespace):
        ids = [f"{metadata_id}#{uuid.uuid4().hex}" for _ in range(len(chunks))]
        returned_ids = await vector_store.aadd_documents(chunks, namespace=namespace, ids=ids)
        logger.debug(f"upsert_vector_store: {returned_ids}")