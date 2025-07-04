import datetime
import uuid

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models, AsyncQdrantClient, QdrantClient


from tilellm.models.item_model import (MetadataItem,
                                       RepositoryQueryResult,
                                       RepositoryItems,
                                       IndexingResult,
                                       RepositoryNamespaceResult,
                                       RepositoryItemNamespaceResult,
                                       RepositoryIdSummaryResult,
                                       RepositoryDescNamespaceResult, Engine, RepositoryNamespace, ItemSingle,
                                       QuestionAnswer, RetrievalChunksResult
                                       )

from typing import Dict

import logging

from tilellm.shared.embedding_factory import inject_embedding, inject_embedding_qa
from tilellm.shared.sparse_util import hybrid_score_norm
from tilellm.store.vector_store_repository import VectorStoreRepository
from tilellm.tools.document_tools import fetch_documents, get_content_by_url_with_bs, calc_embedding_cost
from tilellm.tools.sparse_encoders import TiledeskSparseEncoders

logger = logging.getLogger(__name__)


class QdrantRepository(VectorStoreRepository):



    async def perform_hybrid_search(self, question_answer, index, dense_vector, sparse_vector):
        if question_answer.alpha ==0.5:
            dense = dense_vector
            sparse = sparse_vector
        else:
            dense, sparse = hybrid_score_norm(dense_vector, sparse_vector, alpha=question_answer.alpha)

        filter_qdrant = models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.namespace",
                    match=models.MatchValue(
                        value=question_answer.namespace
                    ),
                ),
            ]
        )

        search_result = index.query_points(
            collection_name=question_answer.engine.index_name,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF  # we are using reciprocal rank fusion here
            ),
            prefetch=[
                models.Prefetch(
                    query=dense,
                    using="text-dense"
                ),

                models.Prefetch(
                    query=sparse,
                    using = "text-sparse"
                ),
            ],
            query_filter=filter_qdrant,  # If you don't want any filters for now
            limit=question_answer.top_k,  # 5 the closest results
        ).points

        #search_result = index.query_points(
        #    query_vector=dense,
        #    query_sparse_vector=models.SparseVector(
        #        indices=sparse.get('indices'),
        #        values=sparse.get('values')
        #    ),
        #    filter=filter_qdrant,
        #    limit=question_answer.top_k,
        #    with_payload=True
        #)

        #metadata = [point.payload for point in search_result]


        documents = []
        for point in search_result:
            if point.payload and "page_content" in point.payload:
                payload = point.payload
                payload["metadata"]["text"] = payload.pop("page_content")
                documents.append(payload)

        results = {"matches": documents}
        return results

    async def initialize_embeddings_and_index(self, question_answer, llm_embeddings):
        emb_dimension = self.get_embeddings_dimension(question_answer.embedding)
        sparse_encoder = TiledeskSparseEncoders(question_answer.sparse_encoder)
        vector_store = await self.create_index(question_answer.engine, llm_embeddings, emb_dimension)
        index = vector_store.client

        return emb_dimension, sparse_encoder, index

    @inject_embedding()
    async def add_item(self, item: ItemSingle, embedding_obj=None, embedding_dimension=None):
        """
        Add items to name
        space into Pinecone index
        :param item:
        :param embedding_obj:
        :param embedding_dimension:
        :return:
        """
        logger.info(item)


        vector_store = await self.create_index(engine=item.engine,
                                               embeddings=embedding_obj,
                                               emb_dimension=embedding_dimension)

        qdrant_client = vector_store.client

        try:
            await self.delete_ids_from_namespace(client=qdrant_client, collection_name=item.engine.index_name, metadata_id=item.id, namespace=item.namespace)

        except Exception as ex:
            logger.warning(ex)
            pass
        chunks = []
        total_tokens = 0
        cost = 0

        try:

            try:
                qdrant_client.create_payload_index(
                    collection_name=item.engine.index_name,
                    field_name="metadata.namespace",
                    field_schema="keyword"  # 'keyword' è buono per ID di tenant
                )
                print(f"Payload index created on 'tenant_id' for collection '{item.engine.index_name}'.")
            except Exception as e:
                if "already exists" in str(e).lower() or "already present" in str(e).lower():  # Adatta il controllo dell'errore
                    print(
                        f"Payload index on 'tenant_id' likely already exists for collection '{item.engine.index_name}'.")
                else:
                    print(f"Could not create payload index (it might already exist or another error): {e}")

            if item.type in ['url', 'pdf', 'docx', 'txt']:
                documents = await fetch_documents(type_source=item.type,
                                                  source=item.source,
                                                  scrape_type=item.scrape_type,
                                                  parameters_scrape_type_4=item.parameters_scrape_type_4)

                chunks = await self.chunk_documents(item=item,
                                                    documents=documents,
                                                    embeddings=embedding_obj
                                                    )
                # print(f"chunks {chunks}")
            else:
                metadata = MetadataItem(id=item.id,
                                        source=item.source,
                                        type=item.type,
                                        embedding=item.embedding,
                                        namespace=item.namespace).model_dump()
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

                #from pprint import pprint
                #pprint(chunks)

            logger.debug(documents)

            total_tokens, cost = calc_embedding_cost(chunks, item.embedding)

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
    async def add_item_hybrid(self, item, embedding_obj=None, embedding_dimension=None):
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

        vector_store = await self.create_index(engine=item.engine,
                                               embeddings=embedding_obj,
                                               emb_dimension=embedding_dimension)

        # default text-embedding-ada-002 1536, text-embedding-3-large 3072, text-embedding-3-small 1536

        chunks = []
        total_tokens = 0
        cost = 0

        qdrant_client = vector_store.client
        try:

            try:
                qdrant_client.create_payload_index(
                    collection_name=item.engine.index_name,
                    field_name="metadata.namespace",
                    field_schema="keyword"  # 'keyword' è buono per ID di tenant
                )
                print(f"Payload index created on 'tenant_id' for collection '{item.engine.index_name}'.")
            except Exception as e:
                if "already exists" in str(e).lower() or "already present" in str(e).lower():  # Adatta il controllo dell'errore
                    print(
                        f"Payload index on 'tenant_id' likely already exists for collection '{item.engine.index_name}'.")
                else:
                    print(f"Could not create payload index (it might already exist or another error): {e}")

            if item.type in ['url', 'pdf', 'docx', 'txt']:
                documents = await self.fetch_documents(type_source=item.type,
                                                       source=item.source,
                                                       scrape_type=item.scrape_type,
                                                       parameters_scrape_type_4=item.parameters_scrape_type_4)



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

            contents = [chunk.page_content for chunk in chunks]
            total_tokens, cost = calc_embedding_cost(chunks, item.embedding)

            sparse_encoder = TiledeskSparseEncoders(item.sparse_encoder)

            doc_sparse_vectors = sparse_encoder.encode_documents(contents, batch_size=item.hybrid_batch_size)


            #async with vector_store.async_index as indice:
            await self.upsert_vector_store_hybrid(vector_store,
                                                      contents,
                                                      chunks,
                                                      item.id,
                                                      namespace=item.namespace,
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

    @inject_embedding_qa()
    async def get_chunks_from_repo(self, question_answer:QuestionAnswer, embedding_obj=None, embedding_dimension=None):
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

            filter_qdrant = models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.namespace",
                        match=models.MatchValue(
                            value=question_answer.namespace
                        ),
                    ),
                ]
            )

            start_time = datetime.datetime.now() if question_answer.debug else 0

            if question_answer.search_type == 'hybrid':
                emb_dimension = self.get_embeddings_dimension(question_answer.embedding)
                sparse_encoder = TiledeskSparseEncoders(question_answer.sparse_encoder)
                index = vector_store.client
                sparse_vector = sparse_encoder.encode_queries(question_answer.question)
                dense_vector = await embedding_obj.aembed_query(question_answer.question)
                if question_answer.alpha == 0.5:
                    dense = dense_vector
                    sparse = sparse_vector
                else:
                    dense, sparse = hybrid_score_norm(dense_vector, sparse_vector, alpha=question_answer.alpha)

                search_result = index.query_points(
                    collection_name=question_answer.engine.index_name,
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF  # we are using reciprocal rank fusion here
                    ),
                    prefetch=[
                        models.Prefetch(
                            query=dense,
                            using="text-dense"
                        ),

                        models.Prefetch(
                            query=sparse,
                            using="text-sparse"
                        ),
                    ],
                    query_filter=filter_qdrant,  # If you don't want any filters for now
                    limit=question_answer.top_k,  # 5 the closest results
                ).points


                results = []
                for point in search_result:

                    if point.payload and "page_content" in point.payload:
                        document = Document(
                            id=point.id,
                            metadata=point.payload.get('metadata',''),
                            page_content=point.payload.pop('page_content')
                        )
                        results.append(document)

            else:
                results = await vector_store.asearch(query=question_answer.question,
                                                     k=question_answer.top_k,
                                                     search_type=question_answer.search_type,
                                                     filter=filter_qdrant)

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

    async def delete_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str):

        try:
            if engine.deployment == "local":
                qdrant_client = AsyncQdrantClient(host=engine.host, port=engine.port)
            else:
                qdrant_client = AsyncQdrantClient(api_key=engine.apikey)

            collection_name = engine.index_name

            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.id",
                        match=models.MatchValue(value=metadata_id)
                    ),
                    models.FieldCondition(
                        key="metadata.namespace",
                        match=models.MatchValue(value=namespace)
                    )
                ]
            )

            response = await qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=delete_filter
                )
            )

            if response.status == models.UpdateStatus.COMPLETED:
                logger.info(f"Eliminazione completata. Stato: {response.status}")
                logger.info(f"Punti eliminati con metadata_id = '{metadata_id}'.")
            else:
                logger.error(f"Eliminazione in corso o fallita. Stato: {response.status}")
                logger.error(f"Dettagli: {response.error}")

        except Exception as e:
            logger.error(f"Errore durante l'eliminazione dei punti: {e}")
            raise e

    async def delete_namespace(self, namespace_to_delete: RepositoryNamespace):
        """
        Delete namespace from Qdrant index
        :param namespace_to_delete:
        :return:
        """

        if namespace_to_delete.engine.deployment == "local":
            qdrant_client = AsyncQdrantClient(host=namespace_to_delete.engine.host, port=namespace_to_delete.engine.port)
        else:
            qdrant_client = AsyncQdrantClient(api_key=namespace_to_delete.engine.apikey)

        collection_name = namespace_to_delete.engine.index_name

        try:
            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.namespace",
                        match=models.MatchValue(value=namespace_to_delete.namespace)
                    )
                ]
            )

            response = await qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=delete_filter
                )
            )

            if response.status == models.UpdateStatus.COMPLETED:
                logger.info(f"Eliminazione completata. Stato: {response.status}")
            else:
                logger.error(f"Eliminazione in corso o fallita. Stato: {response.status}")
                logger.error(f"Dettagli: {response.error}")

        except Exception as e:
            logger.error(f"Errore durante l'eliminazione dei punti: {e}")
            raise e

    async def delete_chunk_id_namespace(self, engine: Engine, chunk_id: str, namespace: str):
        """
        delete chunk from pinecone
        :param engine: Engine
        :param chunk_id:
        :param namespace:
        :return:
        """

        try:
            if engine.deployment == "local":
                qdrant_client = AsyncQdrantClient(host=engine.host, port=engine.port)
            else:
                qdrant_client = AsyncQdrantClient(api_key=engine.apikey)

            collection_name = engine.index_name

            #delete_filter = models.Filter(
            #    must=[
            #        models.FieldCondition(
            #            key="metadata.id",
            #            match=models.MatchValue(value=metadata_id)
            #        ),
            #        models.FieldCondition(
            #            key="metadata.namespace",
            #            match=models.MatchValue(value=namespace)
            #        )
            #    ]
            #)

            response = await qdrant_client.delete(
                collection_name=collection_name,points_selector=[chunk_id]
            )

            if response.status == models.UpdateStatus.COMPLETED:
                logger.info(f"Eliminazione completata. Stato: {response.status}")
            else:
                logger.error(f"Eliminazione in corso o fallita. Stato: {response.status}")
                logger.error(f"Dettagli: {response.error}")

        except Exception as e:
            logger.error(f"Errore durante l'eliminazione dei punti: {e}")
            raise e

    async def get_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str) -> RepositoryItems:
        """
        Get from Qdrant all items from namespace given document id
        :param engine: Engine
        :param metadata_id:
        :param namespace:
        :return:
        """
        try:
            if engine.deployment == "local":
                client = AsyncQdrantClient(host=engine.host, port=engine.port)
            else:
                client = AsyncQdrantClient(api_key=engine.apikey)

            result = []
            all_object_filter = models.Filter(
                must=[models.FieldCondition(
                    key="metadata.id",
                    match=models.MatchValue(value=metadata_id)
                ),
                    models.FieldCondition(
                    key="metadata.namespace",
                    match=models.MatchValue(value=namespace)
                )
                ]
            )
            result = []
            if await client.collection_exists(engine.index_name):
                # collection = client.get_collection(namespace)

                all_points = []
                next_page_offset = None

                while True:
                    # Esegue la richiesta di scroll
                    # payload_selector include solo i campi specifici che ci interessano
                    # limit imposta il numero di punti da recuperare per ogni richiesta (puoi aumentarlo o diminuirlo)
                    # offset è utilizzato per la paginazione
                    scroll_result, next_page_offset = await client.scroll(
                        collection_name=engine.index_name,
                        scroll_filter=all_object_filter,
                        offset=next_page_offset,
                        limit=100,  # Puoi modificare il limite per recuperare più o meno punti per volta
                        with_payload=['page_content','metadata'],  # Specifica i campi di metadata da recuperare
                        with_vectors=False  # Non recuperare i vettori, a meno che non ti servano

                    )

                    all_points.extend(scroll_result)

                    # Se non ci sono più punti o l'offset è nullo, abbiamo finito
                    if next_page_offset is None:
                        break

                logger.debug(f"Qdrant total vector in {namespace}: {len(all_points)}")
                result = []
                for point in all_points:
                    metadata = point.payload.get("metadata")
                    result.append(RepositoryQueryResult(id=point.id,
                                                        metadata_id=metadata.get('id'),
                                                        metadata_source=metadata.get('source'),
                                                        metadata_type=metadata.get('type'),
                                                        date=metadata.get('date', 'Date not defined'),
                                                        text=point.payload.get("page_content")
                                                        # su pod content, su Serverless text
                                                        )
                                  )

            res = RepositoryItems(matches=result)
            logger.debug(res)
            return res
        except Exception as ex:

            logger.error(ex)

            raise ex

    async def get_all_obj_namespace(self, engine: Engine, namespace: str) -> RepositoryItems:
        """
        Query Qdrant to get all object
        :param engine: Engine
        :param namespace:
        :return:
        """


        try:
            if engine.deployment== "local":
                client = AsyncQdrantClient(host=engine.host, port=engine.port)
            else:
                client = AsyncQdrantClient(api_key=engine.apikey)

            result = []
            all_object_filter = models.Filter(
                must=[models.FieldCondition(
                        key="metadata.namespace",
                        match=models.MatchValue(value=namespace)
                    )
                ]
            )

            if await client.collection_exists(engine.index_name):
                #collection = client.get_collection(namespace)

                all_points = []
                next_page_offset = None

                while True:
                    # Esegue la richiesta di scroll
                    # payload_selector include solo i campi specifici che ci interessano
                    # limit imposta il numero di punti da recuperare per ogni richiesta (puoi aumentarlo o diminuirlo)
                    # offset è utilizzato per la paginazione
                    scroll_result, next_page_offset = await client.scroll(
                        collection_name=engine.index_name,
                        scroll_filter=all_object_filter,
                        offset=next_page_offset,
                        limit=100,  # Puoi modificare il limite per recuperare più o meno punti per volta
                        with_payload=['metadata'],  # Specifica i campi di metadata da recuperare
                        with_vectors=False  # Non recuperare i vettori, a meno che non ti servano

                    )

                    all_points.extend(scroll_result)

                    # Se non ci sono più punti o l'offset è nullo, abbiamo finito
                    if next_page_offset is None:
                        break

                logger.debug(f"Qdrant total vector in {namespace}: {len(all_points)}")


                for point in all_points:
                    logger.debug(point)
                    point_id = point.id
                    metadata = point.payload.get("metadata")

                    metadata_id = metadata.get('id') if metadata else None
                    metadata_source = metadata.get('source') if metadata else None
                    metadata_type = metadata.get('type') if metadata else None
                    metadata_date = metadata.get('date', 'Date not defined') if metadata else None

                    logger.debug(f"Point ID: {point_id}")
                    logger.debug(f"  Metadata Name: {metadata_id}")
                    logger.debug(f"  Metadata Source: {metadata_source}")
                    logger.debug(f"  Metadata Type: {metadata_type}")
                    logger.debug(f"  Metadata Type: {metadata_date}")
                    logger.debug("-" * 20)
                    result.append(RepositoryQueryResult(id=point_id,
                                                        metadata_id=metadata_id,
                                                        metadata_source=metadata_source,
                                                        metadata_type=metadata_type,
                                                        date=metadata_date,
                                                        text=None))

            res = RepositoryItems(matches=result)
            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    async def get_desc_namespace(self, engine: Engine, namespace: str) -> RepositoryDescNamespaceResult:
        """
        Query Qdrant to get all object
        :param engine: Engine
        :param namespace:
        :return: PineconeDescNamespaceResult
        """

        try:
            if engine.deployment == "local":
                client = AsyncQdrantClient(host=engine.host, port=engine.port)
            else:
                client = AsyncQdrantClient(api_key=engine.apikey)

            result = []
            all_object_filter = models.Filter(
                must=[models.FieldCondition(
                    key="metadata.namespace",
                    match=models.MatchValue(value=namespace)
                )
                ]
            )

            all_points = []
            if await client.collection_exists(engine.index_name):
                # collection = client.get_collection(namespace)


                next_page_offset = None

                while True:
                    # Esegue la richiesta di scroll
                    # payload_selector include solo i campi specifici che ci interessano
                    # limit imposta il numero di punti da recuperare per ogni richiesta (puoi aumentarlo o diminuirlo)
                    # offset è utilizzato per la paginazione
                    scroll_result, next_page_offset = await client.scroll(
                        collection_name=engine.index_name,
                        scroll_filter=all_object_filter,
                        offset=next_page_offset,
                        limit=100,  # Puoi modificare il limite per recuperare più o meno punti per volta
                        with_payload=['metadata.id',"metadata.source"],  # Specifica i campi di metadata da recuperare
                        with_vectors=False  # Non recuperare i vettori, a meno che non ti servano

                    )

                    all_points.extend(scroll_result)

                    # Se non ci sono più punti o l'offset è nullo, abbiamo finito
                    if next_page_offset is None:
                        break

                logger.debug(f"Qdrant total vector in {namespace}: {len(all_points)}")

            ids_count: Dict[str, RepositoryIdSummaryResult] = {}


            total=len(all_points)
            for obj in all_points:
                metadata_id = obj.payload.get('metadata').get('id')
                if metadata_id in ids_count:
                    ids_count[metadata_id].chunks_count += 1
                else:
                    ids_count[metadata_id] = RepositoryIdSummaryResult(metadata_id=metadata_id,
                                                                       source=obj.payload.get('metadata').get('source'),
                                                                       chunks_count=1)


            description = RepositoryItemNamespaceResult(namespace=namespace, vector_count=total)

            res = RepositoryDescNamespaceResult(namespace_desc=description, ids=list(ids_count.values()))

            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    async def get_sources_namespace(self, engine: Engine, source: str, namespace: str) -> RepositoryItems:
        """
        Get from Qdrant all items from namespace given source
        :param engine: Engine
        :param source:
        :param namespace:
        :return:
        """
        try:
            if engine.deployment == "local":
                client = AsyncQdrantClient(host=engine.host, port=engine.port)
            else:
                client = AsyncQdrantClient(api_key=engine.apikey)

            result = []
            all_object_filter = models.Filter(
                must=[models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source)
                ),
                    models.FieldCondition(
                        key="metadata.namespace",
                        match=models.MatchValue(value=namespace)
                    )
                ]
            )
            result = []
            if await client.collection_exists(engine.index_name):
                # collection = client.get_collection(namespace)

                all_points = []
                next_page_offset = None

                while True:
                    # Esegue la richiesta di scroll
                    # payload_selector include solo i campi specifici che ci interessano
                    # limit imposta il numero di punti da recuperare per ogni richiesta (puoi aumentarlo o diminuirlo)
                    # offset è utilizzato per la paginazione
                    scroll_result, next_page_offset = await client.scroll(
                        collection_name=engine.index_name,
                        scroll_filter=all_object_filter,
                        offset=next_page_offset,
                        limit=100,  # Puoi modificare il limite per recuperare più o meno punti per volta
                        with_payload=['page_content', 'metadata'],  # Specifica i campi di metadata da recuperare
                        with_vectors=False  # Non recuperare i vettori, a meno che non ti servano

                    )

                    all_points.extend(scroll_result)

                    # Se non ci sono più punti o l'offset è nullo, abbiamo finito
                    if next_page_offset is None:
                        break

                logger.debug(f"Qdrant total vector in {namespace}: {len(all_points)}")
                result = []
                for point in all_points:
                    metadata = point.payload.get("metadata")
                    result.append(RepositoryQueryResult(id=point.id,
                                                        metadata_id=metadata.get('id'),
                                                        metadata_source=metadata.get('source'),
                                                        metadata_type=metadata.get('type'),
                                                        date=metadata.get('date', 'Date not defined'),
                                                        text=point.payload.get("page_content")
                                                        # su pod content, su Serverless text
                                                        )
                                  )

            res = RepositoryItems(matches=result)
            logger.debug(res)
            return res
        except Exception as ex:

            logger.error(ex)

            raise ex


    async def list_namespaces(self, engine: Engine) -> RepositoryNamespaceResult:
        try:
            if engine.deployment== "local":
                client = AsyncQdrantClient(host=engine.host, port=engine.port)
            else:
                client = AsyncQdrantClient(api_key=engine.apikey)



            results = []

            qdrant_response = await client.facet(
                collection_name=engine.index_name,
                key="metadata.namespace"
            )

            for hit in qdrant_response.hits:

                pc_item_namespace = RepositoryItemNamespaceResult(namespace=hit.value, vector_count=hit.count)
                results.append(pc_item_namespace)
                logger.debug(f"{hit.value}, {hit.count}")


            logger.info(f"Collections {engine.index_name}")
            return RepositoryNamespaceResult(namespaces=results)
        except Exception as ex:
            logger.error(f"No Collections available '{ex}' does not exist. Creating it ...")
            raise ex

    @staticmethod
    async def list_all_collections(engine: Engine) -> RepositoryNamespaceResult:
        try:
            if engine.deployment== "local":
                client = AsyncQdrantClient(host=engine.host, port=engine.port)
            else:
                client = AsyncQdrantClient(api_key=engine.apikey)

            qdrant_collections = await client.get_collections()

            results = []
            for collection in qdrant_collections.collections:

                details = await client.get_collection(collection_name=collection.name)
                pc_item_namespace = RepositoryItemNamespaceResult(namespace=collection.name, vector_count=details.points_count)
                results.append(pc_item_namespace)
                logger.debug(f"{collection.name}, {details.points_count}")


            logger.info(f"Collections {qdrant_collections}")
            return RepositoryNamespaceResult(namespaces=results)
        except Exception as ex:
            logger.error(f"No Collections available '{ex}' does not exist. Creating it ...")
            raise ex


    @staticmethod
    async def create_index(engine, embeddings, emb_dimension) ->  QdrantVectorStore :
        """
        Create or return existing Qdrant collection and return a Qdrant vector store instance.
        :param engine: Configuration object containing host, port.
        :param collection: Collection name
        :param embeddings: An instance of SentenceTransformerEmbeddings for dense vectors.
        :param emb_dimension: The dimension of the dense embeddings.
        :return: An instance of Qdrant (LangChain VectorStore).
        """

        if engine.deployment== "local":
            client = QdrantClient(host=engine.host, port=engine.port)
        else:
            client = QdrantClient(api_key=engine.apikey)

        collection_name = engine.index_name
        metric_distance = models.Distance[engine.metric.upper()]

        # Verifica se la collection esiste
        try:
            qdrant_collection = client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists. Loading embeddings ... {qdrant_collection}")
        except Exception:
            logger.info(f"Collection '{collection_name}' does not exist. Creating it ...")
            # Crea la collection con supporto per la ricerca ibrida (dense e sparse)
            is_collection_exist = client.collection_exists(collection_name=collection_name)
            if not is_collection_exist:
                qdrant_collection = client.create_collection(
                    collection_name=collection_name,
                    vectors_config={"text-dense":models.VectorParams(
                        size=emb_dimension,
                        distance=metric_distance
                    )},
                    sparse_vectors_config={
                        "text-sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(
                                on_disk=True,
                                full_scan_threshold=10000  # Opzionale, per ottimizzare le scansioni
                            )
                        )
                    }
                )
            logger.info(f"Collection '{collection_name}' created. ")

        vector_store =  QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            vector_name="text-dense",
            sparse_vector_name="text-sparse"

        )


        return vector_store

    @staticmethod
    async def upsert_vector_store(vector_store:QdrantVectorStore, chunks, metadata_id, namespace):
        ids = [f"{uuid.uuid4().hex}" for _ in range(len(chunks))]
        returned_ids = await vector_store.aadd_documents(documents=chunks, ids=ids)


    @staticmethod
    def get_embeddings_dimension(embedding):
        """
        Get embedding dimension for OpenAI embedding model
        :param embedding:
        :return:
        """
        emb_dimension = 1536
        try:
            if embedding == "text-embedding-3-large":
                emb_dimension = 3072
            elif embedding == "text-embedding-3-small":
                emb_dimension = 1536
            else:
                embedding = "text-embedding-ada-002"
                emb_dimension = 1536

        except IndexError:
            embedding = "text-embedding-ada-002"
            emb_dimension = 1536

        return emb_dimension


    @staticmethod
    def chunk_data_extended(data, chunk_size=256, chunk_overlap=10, **kwargs):
        """
        Chunk document in small pieces. Semantic chunking is implemented too with
        percentile, standard_deviation, interquartile, gradient

        :param data:
        :param chunk_size:
        :param chunk_overlap:
        :param kwargs:
        :return:
        """

        use_semantic_chunk = kwargs['semantic']
        if use_semantic_chunk:
            embeddings = kwargs['embeddings']
            breakpoint_threshold_type = kwargs['breakpoint_threshold_type']
            logger.info(f"Semantic chunk with {breakpoint_threshold_type}")
            from langchain_experimental.text_splitter import SemanticChunker
            text_splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type
            )
            chunks = text_splitter.split_documents(data)
        else:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(data)

        return chunks


    async def chunk_documents(self, item, documents, embeddings):
    #    print(f"in chunk_documents item: {item} \n documents {documents} emb: {embeddings}")
        chunks = []
        for document in documents:

            document.metadata["id"] = item.id
            document.metadata["source"] = item.source
            document.metadata["type"] = item.type
            document.metadata["embedding"] = item.embedding
            document.metadata["namespace"] = item.namespace
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
    def process_document_metadata(document, metadata):
        document.metadata.update(metadata)
        document.metadata['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
        for key, value in document.metadata.items():
            if isinstance(value, list) and all(item is None for item in value):
                document.metadata[key] = [""]
            elif value is None:
                document.metadata[key] = ""
        return document

    @staticmethod
    async def process_contents(type_source, source, metadata, content):
        if type_source == 'urlbs':
            doc_array = get_content_by_url_with_bs(source)
            return [Document(page_content=doc, metadata=MetadataItem(**metadata).model_dump()) for doc in doc_array]

        document = Document(page_content=content, metadata=MetadataItem(**metadata).model_dump())
        return [document]

    @staticmethod
    async def delete_ids_from_namespace(client: QdrantClient, collection_name: str, metadata_id: str, namespace: str):
        try:
            # Crea una QueryFilter per specificare la condizione di eliminazione
            # Stiamo cercando payloads in cui il campo `metadata_id_field` è uguale a `id_to_delete`
            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.id",
                        match=models.MatchValue(value=metadata_id)
                    ),
                    models.FieldCondition(
                        key="metadata.namespace",
                        match=models.MatchValue(value=namespace)
                    )
                ]
            )

            response = client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=delete_filter
                )
            )

            if response.status == models.UpdateStatus.COMPLETED:
                logger.info(f"Eliminazione completata. Stato: {response.status}")
                logger.info(f"Punti eliminati con metadata_id = '{metadata_id}'.")
            else:
                logger.error(f"Eliminazione in corso o fallita. Stato: {response.status}")
                logger.error(f"Dettagli: {response.error}")

        except Exception as e:
            logger.error(f"Errore durante l'eliminazione dei punti: {e}")

    @staticmethod
    async def upsert_vector_store_hybrid(vector_store: QdrantVectorStore, contents, chunks, metadata_id, engine, namespace, embeddings,
                                         sparse_vectors):
        embedding_chunk_size = 1000
        batch_size: int = 32

        ids = [f"{uuid.uuid4().hex}" for _ in range(len(chunks))]
        metadatas = [{**chunk.metadata, "namespace": namespace} for chunk in chunks]


        for i in range(0, len(contents), embedding_chunk_size):
            chunk_texts = contents[i: i + embedding_chunk_size]
            chunk_ids = ids[i: i + embedding_chunk_size]
            chunk_metadatas = metadatas[i: i + embedding_chunk_size]
            embedding_values = await embeddings.aembed_documents(chunk_texts)  # embeddings[i: i + embedding_chunk_size]
            sparse_values = sparse_vectors[i: i + embedding_chunk_size]


            print(f" collection {engine.index_name}")
            resp = vector_store.client.upsert(collection_name=engine.index_name,
                                                    points=[
                                                        models.PointStruct(
                                                            id=idr, # Un ID univoco per ogni chunk
                                                            vector={
                                                                "text-dense": embedding,
                                                                "text-sparse": sparse_value
                                                            },
                                                            payload={"metadata":chunk,
                                                                     "page_content": page_content}
                                                        )
                                                        for idr, embedding, chunk, sparse_value, page_content in
                                                        zip(chunk_ids, embedding_values, chunk_metadatas, sparse_values, chunk_texts)])
            logger.info(f"response upsert: {resp}")

