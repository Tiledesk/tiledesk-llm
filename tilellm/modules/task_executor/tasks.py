import os
import logging
from typing import Optional, Dict, Any


import httpx


from taskiq import Context, TaskiqDepends, TaskiqState
from tilellm.modules.task_executor.broker import broker
from tilellm.modules.knowledge_graph import logic as kg_logic
from tilellm.modules.knowledge_graph.models.schemas import (
    GraphCreateRequest, GraphCreateResponse,
    AddDocumentRequest, AddDocumentResponse,
    GraphClusterRequest, GraphClusterResponse
)

# FalkorDB imports
try:
    from tilellm.modules.knowledge_graph_falkor import logic as falkor_logic
    from tilellm.modules.knowledge_graph_falkor.models.schemas import (
        GraphCreateRequest as FalkorGraphCreateRequest,
        AddDocumentRequest as FalkorAddDocumentRequest,
        GraphClusterRequest as FalkorGraphClusterRequest,
        GraphCreateResponse as FalkorGraphCreateResponse,
        AddDocumentResponse as FalkorAddDocumentResponse,
        GraphClusterResponse as FalkorGraphClusterResponse
    )
    FALKORDB_AVAILABLE = True
except ImportError:
    falkor_logic = None
    FalkorGraphCreateRequest = None
    FalkorAddDocumentRequest = None
    FalkorGraphClusterRequest = None
    FalkorGraphCreateResponse = None
    FalkorAddDocumentResponse = None
    FalkorGraphClusterResponse = None
    FALKORDB_AVAILABLE = False

logger = logging.getLogger(__name__)

async def send_webhook(url: str, payload: dict):
    if not url:
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload)
            logger.info(f"Webhook sent to {url}")
    except Exception as e:
        logger.error(f"Failed to send webhook to {url}: {e}")



@broker.task
async def task_graph_create(request_dict: dict, state: TaskiqState = TaskiqDepends()) -> dict:
    """
    Task to create/import a community graph.
    """
    webhook_url = request_dict.get("webhook_url")
    task_id = state.task_id
    try:
        # Convert dict back to Pydantic model
        logger.info(f"TASK [{task_id}] => {request_dict}")
        request = GraphCreateRequest(**request_dict)
        logger.info(f"Starting graph_create task for namespace: {request.namespace}")
        # ✅ Invia heartbeat iniziale
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

@broker.task
async def task_add_document(request_dict: dict) -> dict:
    """
    Task to add a document to the graph.
    """
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

@broker.task
async def task_louvain_cluster(request_dict: dict) -> dict:
    """
    Task to perform Louvain clustering.
    """
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

@broker.task
async def task_leiden_cluster(request_dict: dict) -> dict:
    """
    Task to perform Leiden clustering.
    """
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

@broker.task
async def task_hierarchical_cluster(request_dict: dict) -> dict:
    """
    Task to perform Hierarchical clustering.
    """
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

@broker.task
async def task_community_analysis(request_dict: dict) -> dict:
    """
    Task to analyze document collection for communities.
    """
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

@broker.task
async def task_falkor_graph_create(request_dict: dict) -> dict:
    """
    Task to create/import a community graph using FalkorDB.
    """
    webhook_url = request_dict.get("webhook_url")
    if not FALKORDB_AVAILABLE or falkor_logic is None:
        raise RuntimeError("FalkorDB module not available. Install with 'poetry install --extras graph'.")
    try:
        # Convert dict back to Pydantic model
        logger.info(f"FALKORDB TASK =======================> {request_dict}")
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

@broker.task
async def task_falkor_add_document(request_dict: dict) -> dict:
    """
    Task to add a document to the FalkorDB graph.
    """
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

@broker.task
async def task_falkor_louvain_cluster(request_dict: dict) -> dict:
    """
    Task to perform Louvain clustering on FalkorDB graph.
    """
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

@broker.task
async def task_falkor_leiden_cluster(request_dict: dict) -> dict:
    """
    Task to perform Leiden clustering on FalkorDB graph.
    """
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

@broker.task
async def task_falkor_hierarchical_cluster(request_dict: dict) -> dict:
    """
    Task to perform Hierarchical clustering on FalkorDB graph.
    """
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

@broker.task
async def task_falkor_community_analysis(request_dict: dict) -> dict:
    """
    Task to analyze document collection for communities in FalkorDB graph.
    """
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


@broker.task(retry_on_error=True, max_retries=3, labels={"task_type": "scraping"})
async def task_scrape_item_single(item_dict: dict, state: Context = TaskiqDepends()) -> dict:
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


@broker.task
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
        result = await process_pdf_document_with_embeddings(
            question=request,
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path
            # repo, llm, llm_embeddings will be injected by decorators
        )

        # Notify webhook if provided
        if webhook_url:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(webhook_url, json={
                    "status": "completed",
                    "doc_id": doc_id,
                    "result": result
                })

        logger.info(f"Task finished: process_pdf_document_task for {doc_id}")
        return result

    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        if webhook_url:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(webhook_url, json={
                    "status": "failed",
                    "doc_id": doc_id,
                    "error": str(e)
                })
        raise e