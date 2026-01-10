import logging
from typing import Optional, Dict, Any

import httpx
from tilellm.modules.task_executor.broker import broker
from tilellm.modules.knowledge_graph import logic as kg_logic
from tilellm.modules.knowledge_graph.models.schemas import (
    GraphCreateRequest, GraphCreateResponse,
    AddDocumentRequest, AddDocumentResponse,
    GraphClusterRequest, GraphClusterResponse
)

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
async def task_graph_create(request_dict: dict) -> dict:
    """
    Task to create/import a community graph.
    """
    webhook_url = request_dict.get("webhook_url")
    try:
        # Convert dict back to Pydantic model
        logger.info(f"TASK =======================> {request_dict}")
        request = GraphCreateRequest(**request_dict)
        logger.info(f"Starting graph_create task for namespace: {request.namespace}")
        result = await kg_logic.create_graph(request)
        logger.info(f"Finished graph_create task for namespace: {request.namespace}")
        await send_webhook(webhook_url, result)
        return result
    except Exception as e:
        logger.error(f"Error in graph_create task: {e}")
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