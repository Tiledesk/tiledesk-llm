"""
Milvus Repository implementation using langchain-milvus for vector storage.
Provides async operations for dense and hybrid (dense+sparse) vector search.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Union, Any

from langchain_core.documents import Document
from langchain_milvus import Milvus

from tilellm.models.schemas import (
    RepositoryQueryResult,
    RepositoryItems,
    IndexingResult,
    RepositoryNamespaceResult,
    RepositoryItemNamespaceResult,
    RepositoryIdSummaryResult,
    RepositoryDescNamespaceResult,
    RepositoryNamespace,
    RetrievalChunksResult
)
from tilellm.models import (
    MetadataItem,
    Engine,
    ItemSingle,
    QuestionAnswer,
    LlmEmbeddingModel
)
from tilellm.models.llm import TEIConfig
from tilellm.shared.embeddings.embedding_client_manager import inject_embedding_async_optimized, \
    inject_embedding_qa_async_optimized
from tilellm.shared.sparse_util import hybrid_score_norm
from tilellm.shared.timed_cache import TimedCache
from tilellm.shared.utility import _hash_api_key
from tilellm.store.vector_store_repository import VectorStoreRepository, VectorStoreIndexingError
from tilellm.tools.document_tools import fetch_documents, calc_embedding_cost, \
    get_content_by_url, load_document, handle_regex_custom_chunk, _extract_file_name
from tilellm.tools.sparse_encoders import TiledeskSparseEncoders
from tilellm.modules.ingestion.text_processor import process_auto_detected_text

logger = logging.getLogger(__name__)


class CachedVectorStore:
    """
    Wrapper class to manage async Milvus vector store lifecycle.
    Similar to Qdrant implementation pattern.
    """
    def __init__(self, engine, embeddings, emb_dimension, drop_old=False):
        self.engine = engine
        self.embeddings = embeddings
        self.emb_dimension = emb_dimension
        self._drop_old = drop_old  # Option to drop and recreate collection

        self._vector_store: Optional[Milvus] = None
        self._lock = asyncio.Lock()
        self._loop = None

        # Throttling for connection tests
        self._last_check = 0.0
        self._check_every_sec = 60

    async def _ensure_client(self):
        """
        Ensure Milvus vector store is initialized, running on correct event loop,
        and collection exists.
        """
        cur_loop = asyncio.get_running_loop()
        async with self._lock:
            if self._vector_store is None or self._loop is not cur_loop:
                logger.info(f"Creating new Milvus vector store for host: {self.engine.host}:{self.engine.port}")
                
                # Determine connection URI based on deployment
                uri = self.engine.host
                if self.engine.deployment == "local":
                    if not uri.startswith("http"):
                        uri = f"http://{self.engine.host}:{self.engine.port}"
                else:  # cloud
                    if not uri.startswith("http") and not uri.startswith("https"):
                        uri = f"https://{self.engine.host}"
                    if self.engine.port and str(self.engine.port) not in uri:
                        uri = f"{uri}:{self.engine.port}"

                # Create vector store with LangChain Milvus
                # This will auto-create collection if it doesn't exist
                try:
                    connection_args = {
                        "uri": uri,
                        "db_name": getattr(self.engine, 'database', 'default'),
                    }
                    
                    # Add token only if apikey is present and not empty
                    if self.engine.apikey and self.engine.apikey.get_secret_value():
                        connection_args["token"] = self.engine.apikey.get_secret_value()
                    index_params = {
                        "index_type": "HNSW",  # Hierarchical Navigable Small World
                        "metric_type": self.engine.metric,  # o "IP", "L2" in base al caso
                        "params": {
                            "M": 16,  # Numero connessioni per nodo (4-64)
                            "efConstruction": 200,  # Parametro costruzione grafo (100-500)
                        }
                    }

                    self._vector_store = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=self.engine.index_name,
                        connection_args=connection_args,
                        vector_field="vector",
                        index_params=index_params,
                        text_field="page_content",
                        primary_field="id",
                        enable_dynamic_field=True,  # Use this instead of metadata_field (deprecated)
                        drop_old=self._drop_old,  # Use the configured value
                    )
                    logger.info(f"Milvus vector store initialized for collection: {self.engine.index_name}")
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to initialize Milvus vector store: {e}")
                    
                    # Check if this is the sparse_vector index error
                    if "sparse_vector" in error_msg and "index" in error_msg.lower():
                        logger.error(
                            f"\n{'='*80}\n"
                            f"COLLECTION ERROR: Collection '{self.engine.index_name}' has a sparse_vector field\n"
                            f"without a required index. This usually happens when the collection was\n"
                            f"created with an older version or different configuration.\n\n"
                            f"SOLUTIONS (choose one):\n"
                            f"1. DELETE AND RECREATE (easiest):\n"
                            f"   The collection will be recreated automatically on next attempt if you\n"
                            f"   temporarily set drop_old=True or delete it manually:\n"
                            f"   - Using Milvus CLI: `milvus_cli > drop collection {self.engine.index_name}`\n"
                            f"   - Or via Python: see delete_collection() method\n\n"
                            f"2. CREATE INDEX MANUALLY:\n"
                            f"   Create an index on the sparse_vector field using Milvus CLI or SDK\n\n"
                            f"3. USE DIFFERENT COLLECTION NAME:\n"
                            f"   Change the index_name in your engine configuration\n"
                            f"{'='*80}\n"
                        )
                    
                    raise
                
                self._loop = cur_loop

    async def get_vector_store(self) -> Milvus:
        """
        Return the Milvus vector store instance.
        """
        await self._ensure_client()
        return self._vector_store

    async def get_client(self):
        """
        Return a MilvusClient instance for direct operations.
        """
        await self._ensure_client()
        from pymilvus import MilvusClient

        # Determine connection URI
        uri = self.engine.host
        if self.engine.deployment == "local":
            if not uri.startswith("http"):
                uri = f"http://{self.engine.host}:{self.engine.port}"
        else:  # cloud
            if not uri.startswith("http") and not uri.startswith("https"):
                uri = f"https://{self.engine.host}"
            if self.engine.port and str(self.engine.port) not in uri:
                uri = f"{uri}:{self.engine.port}"

        # Create connection args
        connection_args = {"uri": uri}
        if self.engine.apikey and self.engine.apikey.get_secret_value():
            connection_args["token"] = self.engine.apikey.get_secret_value()

        return MilvusClient(**connection_args)

    async def test_connection(self) -> bool:
        """
        Test connection to Milvus with throttling.
        """
        now = time.time()
        if now - self._last_check < self._check_every_sec:
            return True

        try:
            vector_store = await self.get_vector_store()
            # Simple connection test - try to get collection info
            # Milvus will raise exception if connection fails
            self._last_check = now
            return True
        except Exception as e:
            logger.warning(f"Milvus connection test failed: {e}. Attempting recovery.")
            async with self._lock:
                self._vector_store = None

            return False

    async def close(self):
        """Close Milvus connection."""
        if self._vector_store:
            try:
                # Milvus doesn't have explicit close, but we clear the reference
                self._vector_store = None
                logger.info("Milvus vector store reference cleared.")
            finally:
                self._vector_store = None
                self._loop = None


async def _create_vector_store_instance(engine, embeddings, emb_dimension, drop_old=False) -> CachedVectorStore:
    """Factory function to create CachedVectorStore instance."""
    return CachedVectorStore(engine, embeddings, emb_dimension, drop_old)


class MilvusRepository(VectorStoreRepository):
    """
    Milvus repository implementation using langchain-milvus.
    Supports both dense-only and hybrid (dense+sparse) vector search.
    """
    sparse_enabled = False  # Disabled by default to avoid index issues with sparse vectors.
                           # Set to True only if you need hybrid search AND have proper index setup.

    @staticmethod
    async def _get_milvus_client(engine: Engine):
        """
        Helper method to create a MilvusClient instance.
        Returns the client that can be used for direct operations.
        """
        from pymilvus import MilvusClient

        # Determine connection URI
        uri = engine.host
        if engine.deployment == "local":
            if not uri.startswith("http"):
                uri = f"http://{engine.host}:{engine.port}"
        else:  # cloud
            if not uri.startswith("http") and not uri.startswith("https"):
                uri = f"https://{engine.host}"
            if engine.port and str(engine.port) not in uri:
                uri = f"{uri}:{engine.port}"

        # Create connection args
        connection_args = {"uri": uri}
        if engine.apikey and engine.apikey.get_secret_value():
            connection_args["token"] = engine.apikey.get_secret_value()

        return MilvusClient(**connection_args)

    @staticmethod
    async def create_index_cache_wrapper(engine, embeddings, emb_dimension, embedding_config_key=None, cache_suffix=None, drop_old=False) -> CachedVectorStore:
        """
        Get CachedVectorStore instance from cache or create new one.
        Similar to Qdrant implementation.
        """
        cache_key = (
            engine.host,
            engine.port,
            engine.index_name,
            _hash_api_key(engine.apikey.get_secret_value()) if engine.apikey else "",
            embedding_config_key if embedding_config_key is not None else "default",
            str(drop_old)  # Include drop_old in cache key
        )
        if cache_suffix is not None:
            cache_key = cache_key + (cache_suffix,)

        async def _wrapper_creator():
            return await _create_vector_store_instance(engine, embeddings, emb_dimension, drop_old)

        wrapper = await TimedCache.async_get(
            object_type="milvus_vector_store_wrapper",
            key=cache_key,
            constructor=_wrapper_creator
        )
        return wrapper

    async def create_index(self, engine, embeddings, emb_dimension, embedding_config_key=None, cache_suffix=None, drop_old=False):
        """
        Main method to get Milvus vector store.
        Uses cached wrapper for connection management.
        
        Args:
            engine: Engine configuration
            embeddings: Embedding model
            emb_dimension: Embedding dimension
            embedding_config_key: Optional config key for caching
            cache_suffix: Optional suffix for cache key
            drop_old: If True, drop and recreate the collection (USE WITH CAUTION!)
        """
        cached_vs_wrapper = await self.create_index_cache_wrapper(
            engine, embeddings, emb_dimension, embedding_config_key, cache_suffix, drop_old
        )
        # Return the vector store
        return await cached_vs_wrapper.get_vector_store()

    def build_filter(self, namespace: str, filter_dict: Optional[Dict] = None) -> str:
        """
        Build Milvus filter expression from namespace and optional filter dict.
        """
        # Start with namespace condition
        namespace_condition = f'metadata["namespace"] == "{namespace}"'

        if not filter_dict:
            return namespace_condition

        def convert_to_expression(filter_dict: Dict) -> str:
            """Recursively convert filter dict to Milvus expression."""
            # Handle $and
            if "$and" in filter_dict:
                sub_exprs = [convert_to_expression(sub) for sub in filter_dict["$and"]]
                return "(" + " and ".join(sub_exprs) + ")"

            # Handle $or
            if "$or" in filter_dict:
                sub_exprs = [convert_to_expression(sub) for sub in filter_dict["$or"]]
                return "(" + " or ".join(sub_exprs) + ")"

            # Handle $not
            if "$not" in filter_dict:
                sub_expr = convert_to_expression(filter_dict["$not"])
                return f"not ({sub_expr})"

            # Handle tags field condition
            if "tags" in filter_dict:
                tag_cond = filter_dict["tags"]
                if "$in" in tag_cond:
                    tags = tag_cond["$in"]
                    # Milvus JSON contains check
                    if len(tags) == 1:
                        return f'metadata["tags"] contains "{tags[0]}"'
                    else:
                        conditions = [f'metadata["tags"] contains "{tag}"' for tag in tags]
                        return "(" + " or ".join(conditions) + ")"
                elif "$nin" in tag_cond:
                    tags = tag_cond["$nin"]
                    if len(tags) == 1:
                        return f'not (metadata["tags"] contains "{tags[0]}")'
                    else:
                        conditions = [f'not (metadata["tags"] contains "{tag}")' for tag in tags]
                        return "(" + " and ".join(conditions) + ")"
                else:
                    logger.warning(f"Unsupported tag condition: {tag_cond}")
                    return "1 == 1"

            # Unknown structure
            logger.warning(f"Unsupported filter structure: {filter_dict}")
            return "1 == 1"

        tag_expression = convert_to_expression(filter_dict)

        # Combine namespace with tag expression using AND
        if tag_expression == "1 == 1":
            return namespace_condition

        return f"{namespace_condition} and {tag_expression}"

    async def delete_collection(self, engine: Engine) -> bool:
        """
        Delete a Milvus collection. Useful when you need to recreate a collection
        that has configuration issues (e.g., sparse_vector without index).

        Args:
            engine: Engine configuration with collection name to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            client = await self._get_milvus_client(engine)

            # Check if collection exists and drop it (wrapped in asyncio.to_thread for non-blocking)
            has_collection = await asyncio.to_thread(client.has_collection, engine.index_name)

            if has_collection:
                await asyncio.to_thread(client.drop_collection, engine.index_name)
                logger.info(f"Successfully dropped collection: {engine.index_name}")
                return True
            else:
                logger.warning(f"Collection {engine.index_name} does not exist")
                return False

        except Exception as e:
            logger.error(f"Failed to delete collection {engine.index_name}: {e}")
            return False

    @inject_embedding_async_optimized()
    async def add_item(self, item: ItemSingle, embedding_obj=None, embedding_dimension=None):
        """
        Add items to namespace into Milvus collection.
        Uses langchain-milvus async methods.
        """
        logger.info(f"Adding item: {item}")
        
        # Get vector store
        vector_store = await self.create_index(
            engine=item.engine,
            embeddings=embedding_obj,
            emb_dimension=embedding_dimension
        )

        try:
            # Delete existing items with same metadata.id and namespace
            await self.delete_ids_from_namespace(
                vector_store=vector_store,
                metadata_id=item.id,
                namespace=item.namespace
            )
        except Exception as ex:
            logger.warning(f"Could not delete existing items: {ex}")
            pass
        
        chunks = []
        total_tokens = 0
        cost = 0

        try:
            # Fetch and process documents based on item type
            if item.type in ['url', 'pdf', 'docx', 'txt', 'md', 'xlsx', 'xls', 'csv']:
                documents = await self.fetch_documents(
                    type_source=item.type,
                    source=item.source,
                    scrape_type=item.scrape_type,
                    parameters_scrape_type_4=item.parameters_scrape_type_4,
                    browser_headers=item.browser_headers
                )
                chunks = await self.chunk_documents(
                    item=item,
                    documents=documents,
                    embeddings=embedding_obj
                )
            elif item.type == 'regex_custom':
                documents = await self.fetch_documents(
                    type_source=item.type,
                    source=item.source,
                    scrape_type=item.scrape_type,
                    parameters_scrape_type_4=item.parameters_scrape_type_4,
                    browser_headers=item.browser_headers,
                    chunk_regex=item.chunk_regex
                )
                base_metadata = MetadataItem(
                    id=item.id,
                    source=item.source,
                    type=item.type,
                    embedding=str(item.embedding)
                ).model_dump()

                if item.tags:
                    base_metadata["tags"] = item.tags

                # Unisci i metadati del documento con i metadati base
                chunks = [
                    Document(
                        page_content=document.page_content,
                        metadata={**document.metadata, **base_metadata}  # Merge dei due dizionari
                    )
                    for document in documents
                ]
            else:
                # Direct text content — use auto-detection for format
                metadata = MetadataItem(
                    id=item.id,
                    source=item.source,
                    type=item.type,
                    embedding=item.embedding,
                    namespace=item.namespace
                ).model_dump()
                if item.tags:
                    metadata["tags"] = item.tags

                # Auto-detect format (markdown, tabular, or plain) and chunk appropriately
                chunks.extend(await process_auto_detected_text(
                    content=item.content,
                    source=item.source,
                    doc_id=item.id,
                    chunk_size=item.chunk_size,
                    chunk_overlap=item.chunk_overlap,
                    metadata=metadata,
                    semantic_chunk=item.semantic_chunk,
                    embeddings=embedding_obj
                ))

            if getattr(item, 'situated_context', None) and chunks:
                try:
                    from tilellm.shared.situated_context import enrich_chunks_with_situated_context, build_llm_from_item
                    situated_llm = await build_llm_from_item(item)
                    if situated_llm:
                        chunks = await enrich_chunks_with_situated_context(chunks, situated_llm)
                        logger.info(f"Situated context applied to {len(chunks)} chunks.")
                except Exception as sc_err:
                    logger.warning(f"Situated context enrichment failed, continuing without: {sc_err}")

            if len(chunks) == 0:
                raise Exception("No chunks generated from source")

            # Calculate embedding cost
            total_tokens, cost = calc_embedding_cost(chunks, item.embedding)

            # Upsert chunks into Milvus using async method
            returned_ids = await self.upsert_vector_store(
                vector_store=vector_store,
                chunks=chunks,
                metadata_id=item.id,
                namespace=item.namespace
            )

            logger.debug(f"Upserted IDs: {returned_ids}")

            return IndexingResult(
                id=item.id,
                chunks=len(chunks),
                total_tokens=total_tokens,
                cost=f"{cost:.6f}"
            )

        except Exception as ex:
            logger.error(f"Error adding item: {repr(ex)}")
            
            index_res = IndexingResult(
                id=item.id,
                chunks=len(chunks),
                total_tokens=total_tokens,
                status=400,
                cost=f"{cost:.6f}",
                error=str(ex)
            )
            raise VectorStoreIndexingError(index_res.model_dump())

    @inject_embedding_async_optimized()
    async def add_item_hybrid(self, item, embedding_obj=None, embedding_dimension=None):
        """
        Add item for hybrid search (dense + sparse vectors).
        Uses langchain-milvus with sparse vector support.
        
        NOTE: Hybrid search requires sparse_enabled=True and proper index setup.
        If you get index errors, delete and recreate the collection.
        """
        if not self.sparse_enabled:
            logger.warning("Hybrid search requested but sparse_enabled=False. Falling back to dense-only.")
            return await self.add_item(item, embedding_obj=embedding_obj, embedding_dimension=embedding_dimension)
        
        logger.info(f"Adding hybrid item: {item}")
        
        try:
            # Delete existing items
            await self.delete_ids_namespace(
                engine=item.engine,
                metadata_id=item.id,
                namespace=item.namespace
            )
        except Exception as ex:
            logger.warning(f"Could not delete existing items: {ex}")
            pass

        # Get vector store
        vector_store = await self.create_index(
            engine=item.engine,
            embeddings=embedding_obj,
            emb_dimension=embedding_dimension
        )

        chunks = []
        total_tokens = 0
        cost = 0

        try:
            # Fetch and process documents
            if item.type in ['url', 'pdf', 'docx', 'txt', 'md', 'xlsx', 'xls', 'csv']:
                documents = await self.fetch_documents(
                    type_source=item.type,
                    source=item.source,
                    scrape_type=item.scrape_type,
                    parameters_scrape_type_4=item.parameters_scrape_type_4,
                    browser_headers=item.browser_headers
                )
                chunks = await self.chunk_documents(
                    item=item,
                    documents=documents,
                    embeddings=embedding_obj
                )
            elif item.type == 'regex_custom':
                documents = await self.fetch_documents(
                    type_source=item.type,
                    source=item.source,
                    scrape_type=item.scrape_type,
                    parameters_scrape_type_4=item.parameters_scrape_type_4,
                    browser_headers=item.browser_headers,
                    chunk_regex=item.chunk_regex
                )
                base_metadata = MetadataItem(
                    id=item.id,
                    source=item.source,
                    type=item.type,
                    embedding=str(item.embedding)
                ).model_dump()

                if item.tags:
                    base_metadata["tags"] = item.tags

                # Unisci i metadati del documento con i metadati base
                chunks = [
                    Document(
                        page_content=document.page_content,
                        metadata={**document.metadata, **base_metadata}  # Merge dei due dizionari
                    )
                    for document in documents
                ]
            else:
                # Direct text content — use auto-detection for format
                metadata = MetadataItem(
                    id=item.id,
                    source=item.source,
                    type=item.type,
                    embedding=str(item.embedding)
                ).model_dump()
                if item.tags:
                    metadata["tags"] = item.tags

                # Auto-detect format (markdown, tabular, or plain) and chunk appropriately
                chunks.extend(await process_auto_detected_text(
                    content=item.content,
                    source=item.source,
                    doc_id=item.id,
                    chunk_size=item.chunk_size,
                    chunk_overlap=item.chunk_overlap,
                    metadata=metadata,
                    semantic_chunk=item.semantic_chunk,
                    embeddings=embedding_obj
                ))

            if len(chunks) == 0:
                raise Exception("No chunks generated from source")

            if getattr(item, 'situated_context', None) and chunks:
                try:
                    from tilellm.shared.situated_context import enrich_chunks_with_situated_context, build_llm_from_item
                    situated_llm = await build_llm_from_item(item)
                    if situated_llm:
                        chunks = await enrich_chunks_with_situated_context(chunks, situated_llm)
                        logger.info(f"Situated context applied to {len(chunks)} chunks.")
                except Exception as sc_err:
                    logger.warning(f"Situated context enrichment failed, continuing without: {sc_err}")

            contents = [chunk.page_content for chunk in chunks]
            total_tokens, cost = calc_embedding_cost(chunks, item.embedding)

            # Generate sparse vectors
            sparse_encoder = TiledeskSparseEncoders(item.sparse_encoder)
            doc_sparse_vectors = sparse_encoder.encode_documents(
                contents,
                batch_size=item.hybrid_batch_size
            )

            # Upsert with hybrid vectors
            await self.upsert_vector_store_hybrid(
                vector_store=vector_store,
                contents=contents,
                chunks=chunks,
                metadata_id=item.id,
                engine=item.engine,
                namespace=item.namespace,
                embeddings=embedding_obj,
                sparse_vectors=doc_sparse_vectors
            )

            return IndexingResult(
                id=item.id,
                chunks=len(chunks),
                total_tokens=total_tokens,
                cost=f"{cost:.6f}"
            )

        except Exception as ex:
            logger.error(f"Error adding hybrid item: {repr(ex)}")
            
            index_res = IndexingResult(
                id=item.id,
                chunks=len(chunks),
                total_tokens=total_tokens,
                status=400,
                cost=f"{cost:.6f}",
                error=str(ex)
            )
            raise VectorStoreIndexingError(index_res.model_dump())

    @inject_embedding_qa_async_optimized()
    async def get_chunks_from_repo(self, question_answer: QuestionAnswer, embedding_obj=None, embedding_dimension=None):
        """
        Retrieve chunks from Milvus based on question.
        Supports both dense and hybrid search.
        """
        try:
            # Get vector store
            vector_store = await self.create_index(
                engine=question_answer.engine,
                embeddings=embedding_obj,
                emb_dimension=embedding_dimension
            )

            from tilellm.shared.tags_query_parser import build_tags_filter
            filter_expr = self.build_filter(
                question_answer.namespace,
                build_tags_filter(question_answer.tags) if question_answer.tags else None
            )

            import datetime
            start_time = datetime.datetime.now() if question_answer.debug else 0

            if question_answer.search_type == 'hybrid' and self.sparse_enabled:
                # Hybrid search with dense + sparse vectors
                emb_dimension = await self.get_embeddings_dimension(question_answer.embedding)
                logger.debug(f"emb_dimension: {emb_dimension}")
                sparse_encoder = TiledeskSparseEncoders(question_answer.sparse_encoder)
                
                # Generate sparse and dense vectors for query
                sparse_vector = sparse_encoder.encode_queries(question_answer.question)
                dense_vector = await embedding_obj.aembed_query(question_answer.question)
                
                # Use hybrid search
                hybrid_result = await self.perform_hybrid_search(
                    question_answer=question_answer,
                    vector_store=vector_store,
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector
                )
                
                matches = hybrid_result.get("matches", [])
                
                # Convert matches to Document objects
                documents = []
                for match in matches:
                    metadata = match.get("metadata", {})
                    text = match.get("text", "")
                    doc_id = match.get("id", "")
                    
                    documents.append(Document(
                        id=doc_id,
                        metadata=metadata,
                        page_content=text
                    ))
                
            else:
                # Dense vector search using asearch
                if question_answer.search_type == 'hybrid' and not self.sparse_enabled:
                    logger.warning("Hybrid search requested but sparse_enabled=False. Using dense search.")
                
                documents = await vector_store.asearch(
                    query=question_answer.question,
                    k=question_answer.top_k,
                    filter=filter_expr
                )
            
            end_time = datetime.datetime.now() if question_answer.debug else 0
            duration = (end_time - start_time).total_seconds() if question_answer.debug else 0.0
            
            retrieval = RetrievalChunksResult(
                success=True,
                namespace=question_answer.namespace,
                chunks=[doc.page_content for doc in documents],
                metadata=[doc.metadata for doc in documents],
                error_message=None,
                duration=duration
            )
            
            return retrieval
            
        except Exception as ex:
            logger.error(f"Error retrieving chunks from Milvus: {ex}")
            raise ex

    async def delete_namespace(self, namespace_to_delete: RepositoryNamespace):
        """
        Delete all items in a namespace from Milvus collection.
        """
        try:
            engine = namespace_to_delete.engine
            client = await self._get_milvus_client(engine)

            # Build filter expression
            filter_expr = f'metadata["namespace"] == "{namespace_to_delete.namespace}"'

            # Delete using filter (wrapped in asyncio.to_thread for non-blocking)
            result = await asyncio.to_thread(
                client.delete,
                collection_name=engine.index_name,
                filter=filter_expr
            )
            logger.info(f"Deleted namespace '{namespace_to_delete.namespace}', result: {result}")

        except Exception as e:
            logger.error(f"Error deleting namespace from Milvus: {e}")
            raise e

    async def delete_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str):
        """
        Delete items with specific metadata.id from namespace.
        """
        try:
            client = await self._get_milvus_client(engine)

            # Build filter expression
            filter_expr = f'metadata["id"] == "{metadata_id}" and metadata["namespace"] == "{namespace}"'

            # Delete using filter (wrapped in asyncio.to_thread for non-blocking)
            result = await asyncio.to_thread(
                client.delete,
                collection_name=engine.index_name,
                filter=filter_expr
            )
            logger.info(f"Deleted items with metadata_id='{metadata_id}' from namespace='{namespace}', result: {result}")

        except Exception as e:
            logger.error(f"Error deleting items from Milvus: {e}")
            raise e

    async def delete_chunk_id_namespace(self, engine: Engine, chunk_id: str, namespace: str):
        """
        Delete a single chunk by its ID and namespace.
        """
        try:
            client = await self._get_milvus_client(engine)

            # Delete by IDs (wrapped in asyncio.to_thread for non-blocking)
            result = await asyncio.to_thread(
                client.delete,
                collection_name=engine.index_name,
                ids=[chunk_id]
            )
            logger.info(f"Deleted chunk with ID='{chunk_id}' from namespace='{namespace}', result: {result}")

        except Exception as e:
            logger.error(f"Error deleting chunk from Milvus: {e}")
            raise e

    async def delete_ids_from_namespace(self, vector_store: Milvus, metadata_id: str, namespace: str):
        """
        Helper method to delete items by metadata_id from namespace.
        Uses pymilvus client for deletion.
        """
        filter_expr = f'metadata["id"] == "{metadata_id}" and metadata["namespace"] == "{namespace}"'
        logger.info(f"Deleting items with metadata_id={metadata_id} from namespace={namespace}")

        try:
            from pymilvus import MilvusClient

            # Get connection info from vector_store
            collection_name = vector_store.collection_name
            connection_args = vector_store.connection_args

            # Create client
            client = MilvusClient(
                uri=connection_args.get("uri"),
                token=connection_args.get("token")
            )

            # Delete using filter expression (wrapped in asyncio.to_thread for non-blocking)
            result = await asyncio.to_thread(
                client.delete,
                collection_name=collection_name,
                filter=filter_expr
            )
            logger.info(f"Deleted items with filter: {filter_expr}, result: {result}")

        except Exception as e:
            logger.error(f"Error deleting items from Milvus: {e}")
            raise e

    async def get_by_doc_id(self, engine: Engine, namespace: str, doc_id: str) -> List[Document]:
        """
        Get chunks from doc_id and namespace in Milvus.
        """
        try:
            client = await self._get_milvus_client(engine)

            # Try doc_id first
            filter_expr = f'metadata["doc_id"] == "{doc_id}" and metadata["namespace"] == "{namespace}"'
            
            results = await asyncio.to_thread(
                client.query,
                collection_name=engine.index_name,
                filter=filter_expr,
                output_fields=["id", "metadata", "page_content"],
                limit=10000
            )

            # Fallback to id (metadata_id) if no results found
            if not results:
                logger.debug(f"No results found for doc_id='{doc_id}'. Trying metadata['id'].")
                filter_expr = f'metadata["id"] == "{doc_id}" and metadata["namespace"] == "{namespace}"'
                results = await asyncio.to_thread(
                    client.query,
                    collection_name=engine.index_name,
                    filter=filter_expr,
                    output_fields=["id", "metadata", "page_content"],
                    limit=10000
                )

            documents = []
            for result in results:
                documents.append(Document(
                    id=result.get("id", ""),
                    metadata=result.get("metadata", {}),
                    page_content=result.get("page_content", "")
                ))

            logger.info(f"Retrieved {len(documents)} chunks for doc_id='{doc_id}' in namespace='{namespace}'")
            return documents

        except Exception as e:
            logger.error(f"Error in Milvus get_by_doc_id: {e}")
            return []

    async def get_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str) -> RepositoryItems:
        """
        Get from Milvus all items from namespace given document id.
        """
        try:
            client = await self._get_milvus_client(engine)

            # Build filter expression
            filter_expr = f'metadata["id"] == "{metadata_id}" and metadata["namespace"] == "{namespace}"'

            # Query with filter (limit to max results) (wrapped in asyncio.to_thread for non-blocking)
            results = await asyncio.to_thread(
                client.query,
                collection_name=engine.index_name,
                filter=filter_expr,
                output_fields=["id", "metadata", "page_content"],
                limit=10000  # Max results similar to Pinecone
            )

            # Convert to RepositoryQueryResult
            matches = []
            for result in results:
                metadata = result.get("metadata", {})
                matches.append(RepositoryQueryResult(
                    id=result.get("id", ""),
                    metadata_id=metadata.get("id", ""),
                    metadata_source=metadata.get("source", ""),
                    metadata_type=metadata.get("type", ""),
                    date=metadata.get("date", "Date not defined"),
                    text=result.get("page_content", "")
                ))

            logger.debug(f"Retrieved {len(matches)} items with metadata_id='{metadata_id}' from namespace='{namespace}'")
            return RepositoryItems(matches=matches)

        except Exception as e:
            logger.error(f"Error getting items from Milvus: {e}")
            raise e

    async def list_namespaces(self, engine: Engine) -> RepositoryNamespaceResult:
        """
        List all unique namespaces in Milvus collection with vector counts.
        """
        try:
            client = await self._get_milvus_client(engine)

            # Query all records to get unique namespaces
            # Note: This is not optimal for large collections, but Milvus doesn't have a facet API
            # (wrapped in asyncio.to_thread for non-blocking)
            results = await asyncio.to_thread(
                client.query,
                collection_name=engine.index_name,
                filter="",  # No filter, get all
                output_fields=["metadata"],
                limit=10000  # Adjust based on expected size
            )

            # Count namespaces
            namespace_counts = {}
            for result in results:
                metadata = result.get("metadata", {})
                namespace = metadata.get("namespace", "")
                if namespace:
                    namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1

            # Convert to result format
            namespaces = [
                RepositoryItemNamespaceResult(namespace=ns, vector_count=count)
                for ns, count in namespace_counts.items()
            ]

            logger.info(f"Found {len(namespaces)} namespaces in collection '{engine.index_name}'")
            return RepositoryNamespaceResult(namespaces=namespaces)

        except Exception as e:
            logger.error(f"Error listing namespaces from Milvus: {e}")
            raise e

    async def get_all_obj_namespace(self, engine: Engine, namespace: str, with_text: bool = False) -> RepositoryItems:
        """
        Get all objects from a namespace in Milvus collection.
        """
        try:
            client = await self._get_milvus_client(engine)

            # Build filter expression
            filter_expr = f'metadata["namespace"] == "{namespace}"'

            # Query with filter
            output_fields = ["id", "metadata"]
            if with_text:
                output_fields.append("page_content")

            # (wrapped in asyncio.to_thread for non-blocking)
            results = await asyncio.to_thread(
                client.query,
                collection_name=engine.index_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=1000  # Max results
            )

            # Convert to RepositoryQueryResult
            matches = []
            for result in results:
                metadata = result.get("metadata", {})
                matches.append(RepositoryQueryResult(
                    id=result.get("id", ""),
                    metadata_id=metadata.get("id", ""),
                    metadata_source=metadata.get("source", ""),
                    metadata_type=metadata.get("type", ""),
                    date=metadata.get("date", "Date not defined"),
                    text=result.get("page_content", "") if with_text else None
                ))

            logger.debug(f"Retrieved {len(matches)} objects from namespace='{namespace}'")
            return RepositoryItems(matches=matches)

        except Exception as e:
            logger.error(f"Error getting all objects from namespace in Milvus: {e}")
            raise e

    async def get_desc_namespace(self, engine: Engine, namespace: str) -> RepositoryDescNamespaceResult:
        """
        Get namespace description with vector count and document ID summary.
        """
        try:
            client = await self._get_milvus_client(engine)

            # Build filter expression
            filter_expr = f'metadata["namespace"] == "{namespace}"'

            # Query with filter to get all items (wrapped in asyncio.to_thread for non-blocking)
            results = await asyncio.to_thread(
                client.query,
                collection_name=engine.index_name,
                filter=filter_expr,
                output_fields=["id", "metadata"],
                limit=10000  # Max results
            )

            # Count vectors and group by metadata_id
            total_vectors = len(results)
            ids_count: Dict[str, RepositoryIdSummaryResult] = {}

            for result in results:
                metadata = result.get("metadata", {})
                metadata_id = metadata.get("id", "")
                if metadata_id:
                    if metadata_id in ids_count:
                        ids_count[metadata_id].chunks_count += 1
                    else:
                        ids_count[metadata_id] = RepositoryIdSummaryResult(
                            metadata_id=metadata_id,
                            source=metadata.get("source", ""),
                            chunks_count=1
                        )

            # Create namespace description
            namespace_desc = RepositoryItemNamespaceResult(
                namespace=namespace,
                vector_count=total_vectors
            )

            logger.debug(f"Namespace '{namespace}' has {total_vectors} vectors and {len(ids_count)} unique IDs")
            return RepositoryDescNamespaceResult(
                namespace_desc=namespace_desc,
                ids=list(ids_count.values())
            )

        except Exception as e:
            logger.error(f"Error getting namespace description from Milvus: {e}")
            raise e

    async def get_sources_namespace(self, engine: Engine, source: str, namespace: str) -> RepositoryItems:
        """
        Get all items from namespace given source.
        """
        try:
            client = await self._get_milvus_client(engine)

            # Build filter expression
            filter_expr = f'metadata["source"] == "{source}" and metadata["namespace"] == "{namespace}"'

            # Query with filter (wrapped in asyncio.to_thread for non-blocking)
            results = await asyncio.to_thread(
                client.query,
                collection_name=engine.index_name,
                filter=filter_expr,
                output_fields=["id", "metadata", "page_content"],
                limit=10000  # Max results
            )

            # Convert to RepositoryQueryResult
            matches = []
            for result in results:
                metadata = result.get("metadata", {})
                matches.append(RepositoryQueryResult(
                    id=result.get("id", ""),
                    metadata_id=metadata.get("id", ""),
                    metadata_source=metadata.get("source", ""),
                    metadata_type=metadata.get("type", ""),
                    date=metadata.get("date", "Date not defined"),
                    text=result.get("page_content", "")
                ))

            logger.debug(f"Retrieved {len(matches)} items with source='{source}' from namespace='{namespace}'")
            return RepositoryItems(matches=matches)

        except Exception as e:
            logger.error(f"Error getting items by source from Milvus: {e}")
            raise e

    async def aadd_documents(self, engine: Engine, documents: list, namespace: str, embedding_model: any, sparse_encoder: Union[str, TEIConfig, None] = "splade", **kwargs):
        """
        Deletes all documents in a namespace and adds new ones.
        Handles dense and sparse vectors for hybrid search.
        Uses langchain-milvus async methods.

        NOTE: If you encounter index errors with sparse vectors, you may need to:
        1. Delete the collection using delete_collection() method
        2. Or set sparse_enabled=False to use dense-only mode
        """
        if self.sparse_enabled:
            logger.info(f"Adding {len(documents)} documents to namespace '{namespace}' with hybrid embeddings.")
        else:
            logger.info(f"Adding {len(documents)} documents to namespace '{namespace}' with dense embeddings only (sparse_enabled=False).")

        # Get vector store
        vector_store = await self.create_index(
            engine=engine,
            embeddings=embedding_model,
            emb_dimension=await self.get_embeddings_dimension(embedding_model)
        )

        collection_name = engine.index_name

        skip_delete = kwargs.get('skip_delete', False)
        metadata_id = kwargs.get('metadata_id')

        try:
            # Clear namespace before adding new documents
            if not skip_delete:
                logger.info(f"Clearing namespace '{namespace}' before upserting.")
                client = await self._get_milvus_client(engine)
                if metadata_id:
                    filter_expr = f'metadata["namespace"] == "{namespace}" && metadata["id"] == "{metadata_id}"'
                else:
                    filter_expr = f'metadata["namespace"] == "{namespace}"'
                try:
                    await asyncio.to_thread(
                        client.delete,
                        collection_name=collection_name,
                        filter=filter_expr
                    )
                    logger.info(f"Cleared documents in namespace '{namespace}' from collection '{collection_name}'.")
                except Exception as del_ex:
                    logger.warning(f"Could not clear namespace (may be empty): {del_ex}")

            # Prepare data
            doc_batch_size = 50

            # Process in batches and add using aadd_documents
            for i in range(0, len(documents), doc_batch_size):
                batch_docs = documents[i:i + doc_batch_size]

                # Ensure namespace in metadata
                for doc in batch_docs:
                    doc.metadata["namespace"] = namespace

                # Use aadd_documents from Milvus vector store
                ids = await vector_store.aadd_documents(batch_docs)

                logger.info(f"Added batch {i//doc_batch_size + 1} of {len(batch_docs)} documents to namespace '{namespace}'.")

            logger.info(f"Successfully added {len(documents)} documents to namespace '{namespace}'.")

        except Exception as e:
            logger.error(f"Error adding documents to Milvus: {e}")
            raise e

    async def initialize_embeddings_and_index(self, question_answer, llm_embeddings, emb_dimension=None, embedding_config_key=None, cache_suffix=None):
        """
        Initialize embeddings and vector store for QA operations.
        """
        logger.info(f"Embedding parameter type: {type(question_answer.embedding)}, value: {question_answer.embedding}")
        
        # Use provided dimension if available, otherwise compute from embedding model
        if emb_dimension is None:
            emb_dimension = await self.get_embeddings_dimension(question_answer.embedding)
        else:
            logger.info(f"Using provided embedding dimension: {emb_dimension}")
        
        logger.info(f"Embedding dimension for collection: {emb_dimension}")
        
        # Initialize sparse encoder only if sparse is enabled
        if self.sparse_enabled and question_answer.sparse_encoder:
            sparse_encoder = TiledeskSparseEncoders(question_answer.sparse_encoder)
        else:
            sparse_encoder = None
            if question_answer.search_type == 'hybrid':
                logger.warning("Hybrid search requested but sparse_enabled=False or no sparse_encoder provided.")
        
        # Get vector store
        vector_store = await self.create_index(
            question_answer.engine, 
            llm_embeddings, 
            emb_dimension, 
            embedding_config_key,
            cache_suffix
        )
        
        return emb_dimension, sparse_encoder, vector_store

    async def perform_hybrid_search(self, question_answer, vector_store, dense_vector, sparse_vector, filter=None):
        """
        Perform hybrid search (dense + sparse) in Milvus.
        Note: Full hybrid search support depends on Milvus and langchain-milvus capabilities.
        """
        if not self.sparse_enabled:
            logger.warning("Hybrid search requested but sparse_enabled=False. Using dense search only.")
            # Fall back to dense search
            filter_expr = self.build_filter(question_answer.namespace, filter)
            documents = await vector_store.asearch(
                query=question_answer.question,
                k=question_answer.top_k,
                filter=filter_expr
            )
            
            matches = []
            for doc in documents:
                matches.append({
                    "id": doc.metadata.get("id", ""),
                    "metadata": doc.metadata,
                    "text": doc.page_content,
                    "score": 0.0
                })
            
            return {"matches": matches}
        
        # Apply alpha weighting if not 0.5
        if question_answer.alpha == 0.5:
            dense = dense_vector
            sparse = sparse_vector
        else:
            dense, sparse = hybrid_score_norm(dense_vector, sparse_vector, alpha=question_answer.alpha)
        
        logger.info(f"Performing hybrid search with alpha={question_answer.alpha}")
        
        # Build filter expression
        filter_expr = self.build_filter(question_answer.namespace, filter)
        
        try:
            # For hybrid search, we might need to use Milvus-specific features
            # This is a simplified version - actual implementation may vary
            
            logger.warning("Full hybrid search (dense + sparse with RRF) may require Milvus-specific implementation. "
                          "Using dense search as primary with sparse fallback.")
            
            # Use asearch for dense search
            documents = await vector_store.asearch(
                query=question_answer.question,
                k=question_answer.top_k,
                filter=filter_expr
            )
            
            # Convert to match format
            matches = []
            for doc in documents:
                matches.append({
                    "id": doc.metadata.get("id", ""),
                    "metadata": doc.metadata,
                    "text": doc.page_content,
                    "score": 0.0  # Score not available from asearch
                })
            
            return {"matches": matches}
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {"matches": []}

    async def search_community_report(self, question_answer, vector_store, dense_vector, sparse_vector):
        """
        Search for community reports using hybrid search.
        """
        return await self.perform_hybrid_search(
            question_answer,
            vector_store,
            dense_vector,
            sparse_vector
        )

    @staticmethod
    async def upsert_vector_store(vector_store: Milvus, chunks: List[Document], metadata_id: str, namespace: str):
        """
        Upsert chunks into Milvus vector store using aadd_documents.
        """
        ids = [f"{metadata_id}#{uuid.uuid4().hex}" for _ in range(len(chunks))]
        
        # Ensure namespace is in metadata
        for chunk in chunks:
            chunk.metadata["namespace"] = namespace
        
        # Use aadd_documents (async)
        returned_ids = await vector_store.aadd_documents(chunks, ids=ids)
        
        logger.debug(f"upsert_vector_store: {returned_ids}")
        return returned_ids

    @staticmethod
    async def upsert_vector_store_hybrid(
        vector_store: Milvus,
        contents: List[str],
        chunks: List[Document],
        metadata_id: str,
        engine: Engine,
        namespace: str,
        embeddings: any,
        sparse_vectors: List[Dict]
    ):
        """
        Upsert chunks with hybrid (dense + sparse) vectors into Milvus.
        
        NOTE: Full hybrid vector support in Milvus through LangChain may require:
        1. Collection with both dense and sparse vector fields
        2. Proper indexing configuration on both fields
        3. sparse_enabled=True in the repository
        
        If you encounter 'no vector index on field: sparse_vector' errors:
        - Delete the collection and recreate it with proper configuration
        - Or set sparse_enabled=False to use dense-only mode
        """
        logger.info(f"Upserting {len(chunks)} chunks with hybrid vectors to namespace '{namespace}'")
        
        # Generate IDs
        ids = [f"{metadata_id}#{uuid.uuid4().hex}" for _ in range(len(chunks))]
        
        # Prepare documents with metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["namespace"] = namespace
            # Add sparse vector info to metadata if supported
            if sparse_vectors and i < len(sparse_vectors):
                # Note: Storing sparse vectors in metadata is a workaround
                # True hybrid search requires proper collection schema
                chunk.metadata["_sparse_vector"] = sparse_vectors[i]
        
        # Use aadd_documents for dense embeddings
        # Milvus will handle dense embedding generation via the embedding_function
        returned_ids = await vector_store.aadd_documents(chunks, ids=ids)
        
        logger.info(f"Upserted {len(returned_ids)} documents. "
                   f"Note: For true hybrid search, ensure sparse_enabled=True and proper index setup.")
        return returned_ids

    @staticmethod
    async def get_vector_store(engine: Engine, embeddings: any, emb_dimension: int) -> Milvus:
        """
        Get or create a Milvus vector store.
        Factory method for creating Milvus vector store instances.
        """
        # Determine connection URI
        uri = engine.host
        if engine.deployment == "local":
            if not uri.startswith("http"):
                uri = f"http://{engine.host}:{engine.port}"
        else:  # cloud
            if not uri.startswith("http") and not uri.startswith("https"):
                uri = f"https://{engine.host}"
            if engine.port and str(engine.port) not in uri:
                uri = f"{uri}:{engine.port}"
        
        # Create connection args
        connection_args = {
            "uri": uri,
            "db_name": getattr(engine, 'database', 'default'),
        }
        
        # Add token only if apikey is present
        if engine.apikey and engine.apikey.get_secret_value():
            connection_args["token"] = engine.apikey.get_secret_value()
        
        # Create vector store
        index_params = {
            "index_type": "HNSW",  # Hierarchical Navigable Small World
            "metric_type": engine.metric,  # o "IP", "L2" in base al caso
            "params": {
                "M": 16,  # Numero connessioni per nodo (4-64)
                "efConstruction": 200,  # Parametro costruzione grafo (100-500)
            }
        }

        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=engine.index_name,
            connection_args=connection_args,
            vector_field="vector",
            index_params=index_params,
            text_field="page_content",
            primary_field="id",
            enable_dynamic_field=True,
            drop_old=False,
        )
        
        return vector_store

    # Helper methods (same as base class)
    @staticmethod
    async def fetch_documents(type_source, source, scrape_type, parameters_scrape_type_4, browser_headers, chunk_regex=None):
        """Fetch documents from source."""
        if type_source in ['url', 'txt', 'md']:
            documents = await get_content_by_url(
                source,
                scrape_type,
                parameters_scrape_type_4=parameters_scrape_type_4,
                browser_headers=browser_headers
            )
        elif type_source == 'regex_custom':
            documents = await handle_regex_custom_chunk(source, chunk_regex, browser_headers)
        else:
            documents = load_document(source, type_source)
        
        # Validation
        if not documents:
            raise ValueError(f"No documents retrieved from the source: {source} (source type: {type_source})")
        
        has_content = any(doc and doc.page_content and doc.page_content.strip() for doc in documents)
        if not has_content:
            raise ValueError(f"Documents retrieved but source content is empty: {source} (source type: {type_source})")
        
        return documents

    @staticmethod
    def process_document_metadata(document, metadata):
        """Process and clean document metadata."""
        import datetime
        document.metadata.update(metadata)
        document.metadata['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
        for key, value in document.metadata.items():
            if isinstance(value, list) and all(item is None for item in value):
                document.metadata[key] = [""]
            elif value is None:
                document.metadata[key] = ""
        return document

    async def chunk_documents(self, item, documents, embeddings):
        """Chunk documents based on item configuration."""
        logger.debug(f"in chunk_documents item: {item} \n documents {documents} emb: {embeddings}")
        chunks = []
        for document in documents:
            document.metadata["id"] = item.id
            document.metadata["source"] = item.source
            document.metadata["type"] = item.type
            document.metadata["embedding"] = str(item.embedding)
            if item.tags:
                document.metadata["tags"] = item.tags
            # Ensure file_name and page are always present.
            if not document.metadata.get("file_name"):
                document.metadata["file_name"] = _extract_file_name(item.source or "")
            if "page" not in document.metadata:
                document.metadata["page"] = 1
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
        """Process content based on type."""
        document = Document(page_content=content, metadata=MetadataItem(**metadata).model_dump(exclude_none=True))
        return [document]

    # Abstract methods from VectorStoreRepository that need implementation
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
            from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(data)

        return chunks
    
    @staticmethod
    async def get_embeddings_dimension(embedding) -> int:
        """Get embedding dimension from model."""
        from tilellm.models import LlmEmbeddingModel
        
        # First check if the object has a dimension attribute (e.g., ResilientEmbeddings)
        if hasattr(embedding, 'dimension') and embedding.dimension is not None:
            return embedding.dimension
        
        # Handle LlmEmbeddingModel
        if isinstance(embedding, LlmEmbeddingModel):
            # Use the dimension from the model if provided
            if embedding.dimension is not None:
                return embedding.dimension
            # Otherwise fallback to default mapping
            embedding_str = embedding.name
        # Handle string
        elif isinstance(embedding, str):
            embedding_str = embedding
        # Handle LangChain embedding objects (ResilientEmbeddings, OpenAIEmbeddings, etc.)
        elif hasattr(embedding, 'model'):
            # Try to get model name from LangChain embedding
            embedding_str = str(embedding.model)
        elif hasattr(embedding, 'model_name'):
            embedding_str = embedding.model_name
        else:
            # Fallback to string representation
            embedding_str = str(embedding)
            logger.warning(f"Unknown embedding type: {type(embedding)}, using string representation: {embedding_str}")
        
        # Map known embedding model names to dimensions
        # OpenAI models
        if embedding_str == "text-embedding-3-large":
            return 3072
        elif embedding_str in ["text-embedding-3-small", "openai/text-embedding-3-small"]:
            return 1536
        elif embedding_str in ["text-embedding-ada-002", "openai/text-embedding-ada-002"]:
            return 1536
        # Azure OpenAI models
        elif isinstance(embedding_str, str) and "text-embedding-3" in embedding_str and "small" in embedding_str.lower():
            return 1536
        elif isinstance(embedding_str, str) and "text-embedding-ada" in embedding_str:
            return 1536
        # Sentence Transformers
        elif embedding_str == "sentence-transformers/all-MiniLM-L6-v2":
            return 384
        elif embedding_str == "sentence-transformers/all-mpnet-base-v2":
            return 768
        # BAAI models
        elif embedding_str == "BAAI/bge-m3":
            return 1024
        elif embedding_str == "BAAI/bge-large-en":
            return 1024
        # TEI models (commonly 1024 dimensions)
        elif isinstance(embedding_str, str) and ("e5-large" in embedding_str.lower() or "multilingual-e5" in embedding_str.lower()):
            return 1024
        # Cohere models
        elif isinstance(embedding_str, str) and "cohere" in embedding_str.lower() and "embed" in embedding_str.lower():
            return 1024
        # If model name contains "1536" assume it's a 1536-dim model
        elif isinstance(embedding_str, str) and "1536" in embedding_str:
            return 1536
        # If model name contains "3072" assume it's a 3072-dim model
        elif isinstance(embedding_str, str) and "3072" in embedding_str:
            return 3072
        # If model name contains "1024" assume it's a 1024-dim model
        elif isinstance(embedding_str, str) and "1024" in embedding_str:
            return 1024
        # Try to get dimension from the model by embedding a test query (fallback)
        if hasattr(embedding, 'embed_query'):
            try:
                # Test embedding
                test_vector = await embedding.aembed_query("test")
                return len(test_vector)
            except Exception:
                pass
        
        # Default fallback
        return 1536
