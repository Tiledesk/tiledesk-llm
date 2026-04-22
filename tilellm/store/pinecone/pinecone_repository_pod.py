import datetime
import uuid
from typing import List, Optional, Union, Dict

from fastapi import HTTPException

from tilellm.models.schemas import (IndexingResult,
                                    RetrievalChunksResult
                                    )
from tilellm.models import (MetadataItem,
                            Engine,
                            QuestionAnswer
                            )
from tilellm.models.llm import TEIConfig

from tilellm.shared.embedding_factory import inject_embedding
from tilellm.shared.embeddings.embedding_client_manager import inject_embedding_qa_async_optimized, inject_embedding_async_optimized
from tilellm.shared.tags_query_parser import build_tags_filter
from tilellm.store.vector_store_repository import VectorStoreIndexingError
from tilellm.tools.document_tools import (get_content_by_url,
                                          load_document,
                                          handle_regex_custom_chunk
                                          )

from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase


from langchain_core.documents import Document


import logging

logger = logging.getLogger(__name__)


class PineconeRepositoryPod(PineconeRepositoryBase):



    async def perform_hybrid_search(self, question_answer, index, dense_vector, sparse_vector, filter: Optional[Dict] = None):
        # Pinecone Pod indices are always dense only
        if sparse_vector is not None:
            logger.warning("Pinecone Pod indices are dense only, ignoring sparse vector.")
        
        try:
            results = await index.query(
                top_k=question_answer.top_k,
                vector=dense_vector,
                namespace=question_answer.namespace,
                include_metadata=True,
                filter=filter
            )
            return results
        finally:
            await index.close()


    async def search_community_report(self, question_answer, index, dense_vector, sparse_vector):
        try:
            results = await index.query(
                top_k=question_answer.top_k,
                vector=dense_vector,
                namespace=question_answer.namespace,
                include_metadata=True
            )
            return results
        finally:
            await index.close()

    @inject_embedding_async_optimized()
    async def add_item(self, item, embedding_obj=None, embedding_dimension=None) -> IndexingResult:
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
        # gpt_key = item.gptkey.get_secret_value()
        embedding = item.embedding
        embedding_name = embedding if isinstance(embedding, str) else embedding.name
        namespace = item.namespace
        semantic_chunk = item.semantic_chunk
        breakpoint_threshold_type = item.breakpoint_threshold_type
        scrape_type = item.scrape_type
        chunk_size = item.chunk_size
        chunk_overlap = item.chunk_overlap
        parameters_scrape_type_4 = item.parameters_scrape_type_4
        engine = item.engine

        try:
            await self.delete_ids_namespace(engine=engine, metadata_id=metadata_id, namespace=namespace)
        except Exception as ex:
            logger.warning(ex)
            pass

        emb_dimension = embedding_dimension

        # default text-embedding-ada-002 1536, text-embedding-3-large 3072, text-embedding-3-small 1536
        oai_embeddings = embedding_obj # OpenAIEmbeddings(api_key=gpt_key, model=embedding)
        vector_store = await self.create_index(engine=engine, embeddings=oai_embeddings, emb_dimension=emb_dimension)

        # print(f"=========== POD {type(vector_store)}")
        chunks = []
        total_tokens = 0
        cost = 0

        try:
            if type_source in ['url', 'pdf', 'docx', 'txt', 'md', 'xlsx', 'xls', 'csv']:

                documents = []
                if type_source in ['url', 'txt', 'md']:
                    documents = await get_content_by_url(source,
                                                         scrape_type,
                                                         parameters_scrape_type_4=parameters_scrape_type_4,
                                                         browser_headers=item.browser_headers)
                else:
                    documents = load_document(source, type_source)

                if not documents:
                    raise ValueError(f"No documents retrieved from the source: {source} (source type: {type_source})")

                has_content = False
                for doc in documents:
                    if doc and doc.page_content and doc.page_content.strip():
                        has_content = True
                        break

                if not has_content:
                    raise ValueError(
                        f"Documents retrieved but source content is empty: {source} (source type: {type_source})")

                for single_document in documents:
                    single_document.metadata["id"] = metadata_id
                    single_document.metadata["source"] = source
                    single_document.metadata["type"] = type_source
                    single_document.metadata["embedding"] = embedding_name
                    if item.tags:
                        single_document.metadata["tags"] = item.tags

                    for key, value in single_document.metadata.items():
                        if isinstance(value, list) and all(item is None for item in value):
                            single_document.metadata[key] = [""]
                        elif value is None:
                            single_document.metadata[key] = ""

                    chunks.extend(self.chunk_data_extended(data=[single_document],
                                                           chunk_size=chunk_size,
                                                           chunk_overlap=chunk_overlap,
                                                           semantic=semantic_chunk,
                                                           embeddings=oai_embeddings,
                                                           breakpoint_threshold_type=breakpoint_threshold_type
                                                           )
                                  )

                if len(chunks) == 0:
                    raise Exception("No chunks generated from source")

            elif type_source == 'regex_custom':
                documents = await handle_regex_custom_chunk(source, item.chunk_regex, item.browser_headers)
                base_metadata = MetadataItem(
                    id=item.id,
                    source=item.source,
                    type=item.type,
                    embedding=str(item.embedding)
                ).model_dump()

                if item.tags:
                    base_metadata["tags"] = item.tags

                chunks = [
                    Document(
                        page_content=document.page_content,
                        metadata={**document.metadata, **base_metadata}
                    )
                    for document in documents
                ]
                if len(chunks) == 0:
                    raise Exception("No chunks generated from source")

            else:
                metadata = MetadataItem(id=metadata_id,
                                        source=source,
                                        type=type_source,
                                        embedding=embedding_name).model_dump(exclude_none=True)
                if item.tags:
                    metadata["tags"] = item.tags
                document = Document(page_content=content, metadata=metadata)

                chunks.extend(self.chunk_data_extended(data=[document],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       semantic=semantic_chunk,
                                                       embeddings=oai_embeddings,
                                                       breakpoint_threshold_type=breakpoint_threshold_type
                                                        )
                               )
                if len(chunks) == 0:
                    return IndexingResult(id=metadata_id,
                                          chunks=0,
                                          total_tokens=0,
                                          cost="0.000000",
                                          error="No chunks generated from source")

            # Situated context enrichment (unified for all branches)
            sc_token_usage = None
            if getattr(item, 'situated_context', None) and chunks:
                try:
                    from tilellm.shared.situated_context import enrich_chunks_with_situated_context, build_llm_from_item
                    situated_llm = await build_llm_from_item(item)
                    if situated_llm:
                        sc_result = await enrich_chunks_with_situated_context(chunks, situated_llm)
                        chunks = sc_result.documents
                        sc_token_usage = sc_result.token_usage
                        logger.info(f"Situated context applied to {len(chunks)} chunks. Tokens: {sc_token_usage}")
                except Exception as sc_err:
                    logger.warning(f"Situated context enrichment failed, continuing without: {sc_err}")

            total_tokens, cost = self.calc_embedding_cost(chunks, embedding_name)
            ids = await vector_store.aadd_documents(chunks, namespace=namespace)
            logger.debug(f"ids: {ids}")
            logger.info(f"chunks: {len(chunks)}, total_tokens: {total_tokens}, cost: {cost:.6f}")

            pinecone_result = IndexingResult(
                id=metadata_id,
                chunks=len(chunks),
                total_tokens=total_tokens,
                cost=f"{cost:.6f}",
                sc_input_tokens=sc_token_usage["input_tokens"] if sc_token_usage else None,
                sc_output_tokens=sc_token_usage["output_tokens"] if sc_token_usage else None,
                sc_total_tokens=sc_token_usage["total_tokens"] if sc_token_usage else None,
            )
            return pinecone_result

        except Exception as ex:
            logger.error(repr(ex))
            index_res = IndexingResult(id=metadata_id, chunks=len(chunks), total_tokens=total_tokens,
                                             status=400,
                                             cost=f"{cost:.6f}",
                                             error=str(ex))

            raise VectorStoreIndexingError(index_res.model_dump())


    @inject_embedding_async_optimized()
    async def add_item_hybrid(self, item, embedding_obj=None, embedding_dimension=None):
        pass

    async def initialize_embeddings_and_index(self, question_answer, llm_embeddings, emb_dimension=None, embedding_config_key=None, cache_suffix=None):
        emb_dimension = await self.get_embeddings_dimension(question_answer.embedding)
        
        # Pinecone Pod indices are always dense only
        # Normalize sparse_encoder (empty string treated as None)
        sparse_encoder_param = question_answer.sparse_encoder
        if sparse_encoder_param == "":
            sparse_encoder_param = None
        
        if sparse_encoder_param is not None:
            logger.warning("Pinecone Pod indices are dense only, ignoring sparse_encoder parameter.")
        
        sparse_encoder = None  # Always None for Pod
        
        vector_store = await self.create_index(question_answer.engine, llm_embeddings, emb_dimension, embedding_config_key, cache_suffix)
        index = await vector_store.async_index

        return emb_dimension, sparse_encoder, index

    #@inject_embedding()
    async def aadd_documents(self, engine: Engine, documents: List[Document], namespace: str, embedding_model: any, sparse_encoder: Union[str, TEIConfig, None] = None, **kwargs):
        # Pinecone Pod indices are always dense only
        # Normalize sparse_encoder (empty string treated as None)
        if sparse_encoder == "":
            sparse_encoder = None
        if sparse_encoder is not None:
            logger.warning(f"Pinecone Pod indices are dense only, ignoring sparse_encoder parameter.")
        
        logger.info(f"Adding {len(documents)} documents to namespace '{namespace}' with dense embeddings only (Pod).")
        
        # 1. Get Pinecone Index
        emb_dimension = await self.get_embeddings_dimension(embedding_model)
        vector_store_instance = await self.create_index(engine, embedding_model, emb_dimension)
        index = await vector_store_instance.async_index

        try:
            # 2. Clear namespace before adding new documents (handle missing namespace)
            try:
                logger.info(f"Clearing namespace '{namespace}' before upserting.")
                await index.delete(delete_all=True, namespace=namespace)
            except Exception as e:
                if "Namespace not found" in str(e):
                    logger.info(f"Namespace '{namespace}' does not exist, skipping deletion.")
                else:
                    raise

            # 3. Prepare data and embeddings in batches
            doc_batch_size = 100 # Pods can often handle larger batches
            contents = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            all_vector_ids = []
            for i in range(0, len(documents), doc_batch_size):
                batch_contents = contents[i: i + doc_batch_size]
                batch_metadatas = metadatas[i: i + doc_batch_size]
                batch_ids = [str(uuid.uuid4()) for _ in batch_contents]

                # Generate dense embeddings only
                dense_embeds = await embedding_model.aembed_documents(batch_contents)

                # 4. Upsert to Pinecone
                vectors_to_upsert = []
                for j, content in enumerate(batch_contents):
                    combined_metadata = {
                        **batch_metadatas[j],
                        engine.text_key: content,
                        "namespace": namespace
                    }
                    vector = {
                        "id": batch_ids[j],
                        "values": dense_embeds[j],
                        "metadata": combined_metadata
                    }
                    vectors_to_upsert.append(vector)
                
                await index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                logger.info(f"Upserted batch {i//doc_batch_size + 1} with {len(vectors_to_upsert)} vectors to namespace '{namespace}'.")
                all_vector_ids.extend(batch_ids)
            
            logger.info(f"Successfully added {len(documents)} documents to namespace '{namespace}'.")
            return all_vector_ids

        except Exception as e:
            logger.error(f"Error adding documents to Pinecone Pod: {e}")
            raise
        finally:
            if index:
                await index.close()

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

            # AGIUNTA: build filter from tags
            filter_dict = None
            if question_answer.tags:
                filter_dict = build_tags_filter(question_answer.tags, field="tags")

            if question_answer.search_type == 'hybrid':
                raise HTTPException(
                    status_code=403,
                    detail="Method not implemented in POD istance"
                )

            else:
                results = await vector_store.asearch(query=question_answer.question,
                                                     search_type=question_answer.search_type,
                                                     k=question_answer.top_k,
                                                     namespace=question_answer.namespace,
                                                     filter=filter_dict
                                                     )

            end_time = datetime.now() if question_answer.debug else 0
            duration = (end_time - start_time).total_seconds() if question_answer.debug else 0.0

            if not results:
                raise ValueError("No chunks found with the current filters.")

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
                api_key=engine.apikey.get_secret_value() # const.PINECONE_API_KEY
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.IndexAsyncio(name=engine.index_name, host=host)
            async with index as index:
                describe = await index.describe_index_stats()
                logger.debug(describe)
                namespaces = describe.get("namespaces", {})
                total_vectors = 1

                if namespaces:
                    if namespace in namespaces.keys():
                        total_vectors = namespaces.get(namespace).get('vector_count')

                logger.debug(total_vectors)

                await index.delete(
                    filter={"id": {"$eq": metadata_id}},
                    namespace=namespace)

        except Exception as ex:
            # logger.error(ex)
            raise ex


