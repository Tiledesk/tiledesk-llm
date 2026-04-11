import logging
from typing import Optional, Dict, Any
from tilellm.modules.raptor.repository import RaptorRepository


import httpx


from taskiq import TaskiqDepends, TaskiqState
from tilellm.modules.task_executor.broker import broker




logger = logging.getLogger(__name__)

async def send_webhook(url: Optional[str], payload: dict):
    if not url:
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload)
            logger.info(f"Webhook sent to {url}")
    except Exception as e:
        logger.error(f"Failed to send webhook to {url}: {e}")



@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "neo4j_graph_create"})
async def task_graph_create(request_dict: dict, state: TaskiqState = TaskiqDepends()) -> dict:
    """
    Task to create/import a community graph.
    """
    from tilellm.modules.knowledge_graph import logic as kg_logic
    from tilellm.modules.knowledge_graph.models import GraphCreateRequest
    webhook_url = request_dict.get("webhook_url")
    task_id = state.task_id
    try:
        # Convert dict back to Pydantic model
        logger.info(f"TASK [{task_id}] => {request_dict}")
        request = GraphCreateRequest(**request_dict)
        logger.info(f"Starting graph_create task for namespace: {request.namespace}")
        # ✅ Send Initial heartbeat
        await state.send_heartbeat()

        #result = await kg_logic.create_graph(request)
        result = await kg_logic.create_graph(request)

        logger.info(f"Finished graph_create task for namespace: {request.namespace}")
        if webhook_url:
            await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in graph_create task: [{task_id}]: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "neo4j_add_document"})
async def task_add_document(request_dict: dict) -> dict:
    """
    Task to add a document to the graph.
    """
    from tilellm.modules.knowledge_graph import logic as kg_logic
    from tilellm.modules.knowledge_graph.models.schemas import AddDocumentRequest
    webhook_url = request_dict.get("webhook_url")
    try:
        # Convert dict back to Pydantic model
        request = AddDocumentRequest(**request_dict)
        logger.info(f"Starting add_document task for metadata_id: {request.metadata_id}")
        result = await kg_logic.add_document_to_graph(request)
        logger.info(f"Finished add_document task for metadata_id: {request.metadata_id}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in add_document task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "neo4_louvain_cluster"})
async def task_louvain_cluster(request_dict: dict) -> dict:
    """
    Task to perform Louvain clustering.
    """
    from tilellm.modules.knowledge_graph import logic as kg_logic
    from tilellm.modules.knowledge_graph.models.schemas import GraphClusterRequest
    webhook_url = request_dict.get("webhook_url")
    try:
        # Convert dict back to Pydantic model
        request = GraphClusterRequest(**request_dict)
        logger.info(f"Starting louvain_cluster task for namespace: {request.namespace}")
        result = await kg_logic.cluster_graph_louvain(request)
        logger.info(f"Finished louvain_cluster task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in louvain_cluster task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "neo4j_leiden_cluster"})
async def task_leiden_cluster(request_dict: dict) -> dict:
    """
    Task to perform Leiden clustering.
    """
    from tilellm.modules.knowledge_graph import logic as kg_logic
    from tilellm.modules.knowledge_graph.models.schemas import GraphClusterRequest

    webhook_url = request_dict.get("webhook_url")
    try:
        # Convert dict back to Pydantic model
        request = GraphClusterRequest(**request_dict)
        logger.info(f"Starting leiden_cluster task for namespace: {request.namespace}")
        result = await kg_logic.cluster_graph_leiden(request)
        logger.info(f"Finished leiden_cluster task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in leiden_cluster task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "neo4j_hierarchical_cluster"})
async def task_hierarchical_cluster(request_dict: dict) -> dict:
    """
    Task to perform Hierarchical clustering.
    """
    from tilellm.modules.knowledge_graph import logic as kg_logic
    from tilellm.modules.knowledge_graph.models.schemas import GraphClusterRequest

    webhook_url = request_dict.get("webhook_url")
    try:
        # Convert dict back to Pydantic model
        request = GraphClusterRequest(**request_dict)
        logger.info(f"Starting hierarchical_cluster task for namespace: {request.namespace}")
        result = await kg_logic.cluster_graph_hierarchical(request)
        logger.info(f"Finished hierarchical_cluster task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in hierarchical_cluster task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "neo4j_community_analysis"})
async def task_community_analysis(request_dict: dict) -> dict:
    """
    Task to analyze document collection for communities.
    """
    from tilellm.modules.knowledge_graph import logic as kg_logic
    from tilellm.modules.knowledge_graph.models.schemas import GraphClusterRequest

    webhook_url = request_dict.get("webhook_url")
    try:
        # Reuse GraphClusterRequest as it has namespace and engine
        request = GraphClusterRequest(**request_dict)
        logger.info(f"Starting community_analysis task for namespace: {request.namespace}")
        
        # Call logic function
        result = await kg_logic.analyze_community(request)
        
        logger.info(f"Finished community_analysis task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in community_analysis task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e


# ==================== FALKORDB TASKS ====================

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "falkor_graph_create"})
async def task_falkor_graph_create(request_dict: dict) -> dict:
    """
    Task to create/import a community graph using FalkorDB.
    """



    try:
        from tilellm.modules.knowledge_graph_falkor import logic as falkor_logic
        from tilellm.modules.knowledge_graph_falkor.models.schemas import GraphCreateRequest as FalkorGraphCreateRequest
        FALKORDB_AVAILABLE=True
    except ImportError:
        FALKORDB_AVAILABLE=False
        falkor_logic=None
        FalkorGraphCreateRequest=None

    webhook_url = request_dict.get("webhook_url")
    if not FALKORDB_AVAILABLE or falkor_logic is None:
        raise RuntimeError("FalkorDB module not available. Install with 'poetry install --extras graph'.")
    try:
        # Convert dict back to Pydantic model
        logger.info(f"FALKORDB TASK {request_dict}")
        request = FalkorGraphCreateRequest(**request_dict)
        logger.info(f"Starting falkor_graph_create task for namespace: {request.namespace}")
        result = await falkor_logic.create_graph(request)
        logger.info(f"Finished falkor_graph_create task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in falkor_graph_create task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "falkor_add_document"})
async def task_falkor_add_document(request_dict: dict) -> dict:
    """
    Task to add a document to the FalkorDB graph.
    """
    try:
        from tilellm.modules.knowledge_graph_falkor import logic as falkor_logic
        from tilellm.modules.knowledge_graph_falkor.models.schemas import AddDocumentRequest as FalkorAddDocumentRequest
        FALKORDB_AVAILABLE=True
    except ImportError:
        FALKORDB_AVAILABLE=False
        falkor_logic=None
        FalkorAddDocumentRequest=None


    webhook_url = request_dict.get("webhook_url")
    if not FALKORDB_AVAILABLE or falkor_logic is None:
        raise RuntimeError("FalkorDB module not available. Install with 'poetry install --extras graph'.")
    try:
        # Convert dict back to Pydantic model
        request = FalkorAddDocumentRequest(**request_dict)
        logger.info(f"Starting falkor_add_document task for metadata_id: {request.metadata_id}")
        result = await falkor_logic.add_document_to_graph(request)
        logger.info(f"Finished falkor_add_document task for metadata_id: {request.metadata_id}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in falkor_add_document task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "falkor_louvain_cluster"})
async def task_falkor_louvain_cluster(request_dict: dict) -> dict:
    """
    Task to perform Louvain clustering on FalkorDB graph.
    """
    try:
        from tilellm.modules.knowledge_graph_falkor import logic as falkor_logic
        from tilellm.modules.knowledge_graph_falkor.models.schemas import GraphClusterRequest as FalkorGraphClusterRequest
        FALKORDB_AVAILABLE=True
    except ImportError:
        FALKORDB_AVAILABLE=False
        falkor_logic=None
        FalkorGraphClusterRequest=None

    webhook_url = request_dict.get("webhook_url")
    if not FALKORDB_AVAILABLE or falkor_logic is None:
        raise RuntimeError("FalkorDB module not available. Install with 'poetry install --extras graph'.")
    try:
        # Convert dict back to Pydantic model
        request = FalkorGraphClusterRequest(**request_dict)
        logger.info(f"Starting falkor_louvain_cluster task for namespace: {request.namespace}")
        result = await falkor_logic.cluster_graph_louvain(request)
        logger.info(f"Finished falkor_louvain_cluster task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in falkor_louvain_cluster task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "falkor_leiden_cluster"})
async def task_falkor_leiden_cluster(request_dict: dict) -> dict:
    """
    Task to perform Leiden clustering on FalkorDB graph.
    """
    try:
        from tilellm.modules.knowledge_graph_falkor import logic as falkor_logic
        from tilellm.modules.knowledge_graph_falkor.models.schemas import \
            GraphClusterRequest as FalkorGraphClusterRequest
        FALKORDB_AVAILABLE = True
    except ImportError:
        FALKORDB_AVAILABLE = False
        falkor_logic = None
        FalkorGraphClusterRequest = None

    webhook_url = request_dict.get("webhook_url")
    if not FALKORDB_AVAILABLE or falkor_logic is None:
        raise RuntimeError("FalkorDB module not available. Install with 'poetry install --extras graph'.")
    try:
        # Convert dict back to Pydantic model
        request = FalkorGraphClusterRequest(**request_dict)
        logger.info(f"Starting falkor_leiden_cluster task for namespace: {request.namespace}")
        result = await falkor_logic.cluster_graph_leiden(request)
        logger.info(f"Finished falkor_leiden_cluster task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in falkor_leiden_cluster task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "falkor_hierarchical_cluster"})
async def task_falkor_hierarchical_cluster(request_dict: dict) -> dict:
    """
    Task to perform Hierarchical clustering on FalkorDB graph.
    """
    try:
        from tilellm.modules.knowledge_graph_falkor import logic as falkor_logic
        from tilellm.modules.knowledge_graph_falkor.models.schemas import \
            GraphClusterRequest as FalkorGraphClusterRequest
        FALKORDB_AVAILABLE = True
    except ImportError:
        FALKORDB_AVAILABLE = False
        falkor_logic = None
        FalkorGraphClusterRequest = None

    webhook_url = request_dict.get("webhook_url")
    if not FALKORDB_AVAILABLE or falkor_logic is None:
        raise RuntimeError("FalkorDB module not available. Install with 'poetry install --extras graph'.")
    try:
        # Convert dict back to Pydantic model
        request = FalkorGraphClusterRequest(**request_dict)
        logger.info(f"Starting falkor_hierarchical_cluster task for namespace: {request.namespace}")
        result = await falkor_logic.cluster_graph_hierarchical(request)
        logger.info(f"Finished falkor_hierarchical_cluster task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in falkor_hierarchical_cluster task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e

@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "falkor_community_analysis"})
async def task_falkor_community_analysis(request_dict: dict) -> dict:
    """
    Task to analyze document collection for communities in FalkorDB graph.
    """
    try:
        from tilellm.modules.knowledge_graph_falkor import logic as falkor_logic
        from tilellm.modules.knowledge_graph_falkor.models.schemas import \
            GraphClusterRequest as FalkorGraphClusterRequest
        FALKORDB_AVAILABLE = True
    except ImportError:
        FALKORDB_AVAILABLE = False
        falkor_logic = None
        FalkorGraphClusterRequest = None

    webhook_url = request_dict.get("webhook_url")
    if not FALKORDB_AVAILABLE or falkor_logic is None:
        raise RuntimeError("FalkorDB module not available. Install with 'poetry install --extras graph'.")
    try:
        # Reuse GraphClusterRequest as it has namespace and engine
        request = FalkorGraphClusterRequest(**request_dict)
        logger.info(f"Starting falkor_community_analysis task for namespace: {request.namespace}")
        
        # Call logic function
        result = await falkor_logic.analyze_community(request)
        
        logger.info(f"Finished falkor_community_analysis task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in falkor_community_analysis task: {e}")
        if webhook_url:
             await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e


EXPIRATION_SECONDS = 48 * 60 * 60


@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "raptor_build"})
async def task_raptor_build(request_dict: dict) -> dict:
    """
    Task to build RAPTOR tree.
    """
    from tilellm.modules.raptor.services.raptor_service import RaptorService
    from tilellm.modules.raptor.repository import RaptorRepository
    from tilellm.modules.raptor.config_loader import get_raptor_config_from_env
    from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async
    import redis.asyncio as redis
    
    webhook_url = request_dict.get("webhook_url")
    try:
        # Convert dict back to Pydantic model
        from tilellm.modules.raptor.models.models import RaptorRequest
        request = RaptorRequest(**request_dict)
        logger.info(f"Starting raptor_build task for doc_id: {request.doc_id}")
        
        # Setup dependencies using injection pattern similar to controllers
        @inject_llm_chat_async
        @inject_repo_async
        async def _build_raptor_tree_injected(
            req: RaptorRequest,
            repo=None,
            llm=None,
            llm_embeddings=None,
            **kwargs,
        ):
            from tilellm.modules.raptor.services.raptor_service import RaptorService
            from tilellm.modules.raptor.config_loader import get_raptor_config_from_env
            
            config = req.config or get_raptor_config_from_env()
            
            # Get RAPTOR repository
            raptor_repo = await get_raptor_repo()
            service = RaptorService(repo=raptor_repo)
            
            # Retrieve chunks from vector store (similar to controller logic)
            from tilellm.modules.raptor.controllers import _retrieve_document_chunks, _to_documents
            chunks = await _retrieve_document_chunks(
                namespace=req.namespace,
                doc_id=req.doc_id,
                chunk_ids=req.chunk_ids,
                vector_repo=repo,
                engine=req.engine,
            )
            
            # Ensure chunks is a List[Document] (defensive programming)
            if not isinstance(chunks, list):
                chunks = _to_documents(chunks)
            
            return await service.build_raptor_tree(
                chunks=chunks,
                namespace=req.namespace,
                doc_id=req.doc_id,
                llm=llm,
                embeddings=llm_embeddings,
                vector_repo=repo,
                engine=req.engine,
                config=config,
                sparse_encoder=req.sparse_encoder,
            )
        
        # Execute the injected function
        result = await _build_raptor_tree_injected(request)
        
        logger.info(f"Finished raptor_build task for doc_id: {request.doc_id}")
        if webhook_url:
            # Convert RaptorResponse to dict for webhook payload
            if hasattr(result, 'model_dump'):
                payload = result.model_dump()
            elif hasattr(result, 'dict'):
                payload = result.dict()
            else:
                payload = dict(result)
            await send_webhook(webhook_url, payload)
        # Return dict for Taskiq result_backend
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        elif hasattr(result, 'dict'):
            return result.dict()
        else:
            return dict(result)
    except Exception as e:
        logger.error(f"Error in raptor_build task: {e}")
        if webhook_url:
            await send_webhook(webhook_url, {"error": str(e), "status": "failed"})
        raise e


# Helper function to get RAPTOR repository (similar to controllers.py)
async def get_raptor_repo() -> RaptorRepository:
    """Get RaptorRepository instance backed by Redis."""
    import redis.asyncio as redis
    from tilellm.shared.utility import get_service_config

    config = get_service_config()
    redis_url = f"redis://{config['redis']['host']}:{config['redis']['port']}/{config['redis']['db']}"
    redis_client = redis.from_url(redis_url)
    return RaptorRepository(redis_client=redis_client)


@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "scraping"})
async def task_scrape_item_single(item_dict: dict) -> dict:
    """
    Task to scrape and index a single item.
    Handles both standard and hybrid indexing.
    """
    from tilellm.models import ItemSingle
    from tilellm.controller.controller import add_item, add_item_hybrid
    from tilellm.models.schemas.repository_schemas import IndexingResult

    item = ItemSingle(**item_dict)
    webhook = ""
    token = ""

    try:
        raw_webhook = item.webhook
        if raw_webhook and '?' in raw_webhook:
            webhook, raw_token = raw_webhook.split('?')
            if raw_token.startswith('token='):
                _, token = raw_token.split('=')
        else:
            webhook = raw_webhook or ""

        logger.info(f"Starting scrape task for item {item.id}, hybrid={item.hybrid} ")

        if item.hybrid:
            pc_result = await add_item_hybrid(item)
        else:
            pc_result = await add_item(item)



        if webhook:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        webhook,
                        json=pc_result.model_dump(exclude_none=True),
                        headers={
                            "Content-Type": "application/json",
                            "X-Auth-Token": token
                        }
                    )
                    logger.info(f"Webhook sent to {webhook}")
            except Exception as e:
                logger.error(f"Webhook error: {e}")

        logger.info(f"Finished scrape task for item {item.id}")
        return pc_result.model_dump(exclude_none=True)

    except Exception as e:
        logger.error(f"Error in scrape task for item {item.id}: {e}")

        if webhook:
            try:
                res = IndexingResult(id=item.id, status=400, error=repr(e))
                async with httpx.AsyncClient() as client:
                    await client.post(
                        webhook,
                        json=res.model_dump(exclude_none=True),
                        headers={
                            "Content-Type": "application/json",
                            "X-Auth-Token": token
                        }
                    )
            except Exception as we:
                logger.error(f"Webhook error notification failed: {we}")

        raise e


@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "pdf_ocr"})
async def process_pdf_document_task(
        doc_id: str,
        file_path: Optional[str] = None,
        bucket_name: Optional[str] = None,
        object_name: Optional[str] = None,
        webhook_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Taskiq task to process a PDF document using the advanced logic layer.
    Can process from local path OR MinIO bucket/object.
    
    Returns a lightweight result for Taskiq result_backend (no large mark down content).
    Full result is sent to webhook if configured.
    """
    logger.info(f"Task started: process_pdf_document_task for {doc_id}")

    try:
        # Convert config dict to PDFScrapingRequest
        if config is None:
            config = {}

        # Ensure config has required id field (use doc_id if missing)
        if 'id' not in config:
            config['id'] = doc_id

        # Create request object
        from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest
        request = PDFScrapingRequest(**config)

        # Import logic function
        from tilellm.modules.pdf_ocr.logic import process_pdf_document_with_embeddings

        # Process using logic layer with dependency injection
        full_result = await process_pdf_document_with_embeddings(
            question=request,
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path
            # repo, llm, llm_embeddings will be injected by decorators
        )

        # Build lightweight result for Taskiq result_backend
        # Excludes large fields like Markdown_preview, Markdown_content, etc.
        light_result = {
            "job_id": doc_id,
            "status": "completed",
            "progress": 100,
            "message": "Job completed successfully",
            "result": {
                "markdown_length": full_result.get("markdown_length", 0),
                "num_chunks": full_result.get("num_chunks", 0),
                "num_images": full_result.get("num_images", 0),
                "num_tables": full_result.get("num_tables", 0),
                "metadata": full_result.get("metadata", {})
            },
            "error_message": None
        }

        # Cache invalidation (strategy B): invalidate namespace after pdf_ocr completes
        namespace = config.get("namespace") if config else None
        if namespace:
            try:
                from tilellm.shared.cache import SemanticCache
                await SemanticCache.invalidate_namespace(namespace)
                logger.info(f"Cache invalidated for namespace={namespace} after pdf_ocr task")
            except Exception as cache_exc:
                logger.warning(f"Cache invalidation failed after pdf_ocr task: {cache_exc}")

        # Notify webhook with FULL result (if configured)
        if webhook_url:
            import httpx
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(webhook_url, json={
                        "status": "completed",
                        "doc_id": doc_id,
                        "result": full_result  # Full result to webhook
                    })

                response.raise_for_status()
                logger.info(f"Task finished: process_pdf_document_task for {doc_id}")
            except httpx.ConnectError:
                logger.warning(f"Webhook unreachable, skipping: {webhook_url}")
            except httpx.TimeoutException:
                logger.warning(f"Webhook timed out, skipping: {webhook_url}")
            except httpx.HTTPStatusError as e:
                logger.warning(f"Webhook returned error {e.response.status_code}, skipping")
            except Exception as e:
                logger.warning(f"Webhook failed unexpectedly, skipping: {e}")


        
        # Return lightweight result - Taskiq will save this to result_backend
        return light_result

    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        
        # Build lightweight error result for Taskiq
        light_result = {
            "job_id": doc_id,
            "status": "failed",
            "progress": 100,
            "message": "Job failed",
            "result": None,
            "error_message": str(e)
        }
        
        # Notify webhook with error
        if webhook_url:
            import httpx
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(webhook_url, json={
                        "status": "failed",
                        "doc_id": doc_id,
                        "error": str(e)
                    })
                response.raise_for_status()
                logger.error(f"Task finished with error: process_pdf_document_task for {doc_id}")
            except httpx.ConnectError:
                logger.warning(f"Webhook unreachable, skipping: {webhook_url}")
            except httpx.TimeoutException:
                logger.warning(f"Webhook timed out, skipping: {webhook_url}")
            except httpx.HTTPStatusError as e:
                logger.warning(f"Webhook returned error {e.response.status_code}, skipping")
            except Exception as e:
                logger.warning(f"Webhook failed unexpectedly, skipping: {e}")

        
        # Return error result - Taskiq will save this to result_backend
        return light_result