import io
import os
import uuid
import base64
import requests
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import SecretStr

from tilellm.modules.pdf_ocr.models.pdf_scraping import (
    PDFScrapingRequest, PDFScrapingAcceptResponse, PDFScrapingStatusResponse
)
from tilellm.modules.pdf_ocr.services.taskiq_service import get_taskiq_service
from tilellm.modules.pdf_ocr.services.redis_queue_service import get_redis_queue_service
from tilellm.modules.pdf_ocr.services.job_service import get_job_service

logger = logging.getLogger(__name__)

# Create router for this module
router = APIRouter(
    prefix="/api/pdf",
    tags=["PDF OCR"]
)


@router.post("/scrape", response_model=PDFScrapingAcceptResponse, tags=["PDF OCR"])
async def scrape_pdf(request: PDFScrapingRequest):
    """
    Submit PDF document for scraping. The job is added to a queue for background processing.

    - **file_name**: Original file name with extension (e.g., 'document.pdf')
    - **file_content**: 
      - For Docling pipeline (use_docling=true): Must be a public URL (http/https)
      - For Legacy pipeline (use_docling=false): Base64 or URL (requires MinIO)
    - **webhook_url**: Optional URL to notify when processing is complete
    - **use_docling**: If True, uses the new advanced Docling-based pipeline.
      This pipeline does NOT require MinIO - the worker downloads from URL.
    - **id**: Optional custom job ID (will be used as document reference ID)

    Returns a job ID for tracking the processing status.
    """
    try:
        # doc_id is the stable document identity in the vector store.
        # If the caller provides request.id we honour it so that re-submitting
        # the same document replaces the previous version (aadd_documents deletes
        # the namespace before upserting).
        doc_id = request.id if request.id else str(uuid.uuid4())
        request.id = doc_id

        job_service = get_job_service()

        if request.use_docling:
            # New Advanced Pipeline (Docling-based)
            # This pipeline does NOT require MinIO

            # Validate that file_content is a URL (not Base64)
            if not request.is_url():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "For Docling pipeline (use_docling=true), file_content must be a public URL (http/https). "
                        "Base64 content is not supported. Please upload the PDF to a publicly accessible location "
                        "and provide the URL, or use the legacy pipeline (use_docling=false) with MinIO."
                    )
                )

            logger.info(f"PDF content is a URL: {request.file_content}")

            taskiq_service = get_taskiq_service()

            if taskiq_service.is_available():
                # Submit to Taskiq; task_id is the Taskiq-generated key for status polling.
                result = await taskiq_service.submit(doc_id=doc_id, request=request)
                job_id = result["task_id"]
                message = result.get("message", "PDF processing job (Advanced Docling) has been successfully queued.")
                job_service.register_task(task_id=job_id, doc_id=doc_id, file_name=request.file_name)
            else:
                # Fallback: Process synchronously – use doc_id directly as job_id.
                logger.info(f"Taskiq disabled or unavailable, processing PDF doc_id={doc_id} synchronously")
                result = await taskiq_service.process_synchronously(request=request)
                job_id = doc_id
                message = result.get("message", "PDF processing completed synchronously (Taskiq disabled).")

        else:
            # Legacy Pipeline (requires MinIO)
            queue_service = get_redis_queue_service()

            if not queue_service.is_minio_available():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Legacy PDF pipeline requires MinIO. Error: {queue_service.get_minio_error()}. "
                        "Consider using use_docling=true with a public URL for the new pipeline that works without MinIO."
                    )
                )

            await queue_service.submit(job_id=doc_id, request=request)
            job_id = doc_id
            message = "PDF processing job has been successfully queued."

        return PDFScrapingAcceptResponse(
            job_id=job_id,
            message=message,
            estimated_time=120
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        # This will catch Redis/MinIO connection errors from the service constructor
        raise HTTPException(status_code=503, detail=f"Job queuing service is unavailable: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.get("/status/{job_id}", response_model=PDFScrapingStatusResponse, tags=["PDF OCR"])
async def get_scraping_status(job_id: str):
    """
    Get the status of a PDF scraping job.

    - **job_id**: The job identifier returned from the /scrape endpoint.

    Returns the current job status. Note: status is updated by an external worker.
    """
    try:
        job_service = get_job_service()
        status_response = await job_service.get_status_response(job_id)

        if not status_response:
            raise HTTPException(
                status_code=404,
                detail=f"Job with ID '{job_id}' not found"
            )

        return status_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")


@router.get("/health", tags=["PDF OCR"])
async def health_check():
    """
    Check if the PDF OCR queuing service and its dependencies are healthy.
    
    Returns health status for both the new Taskiq-based pipeline and the legacy Redis/MinIO pipeline.
    """
    try:
        # Check Taskiq service (new pipeline)
        taskiq_service = get_taskiq_service()
        taskiq_healthy = taskiq_service.is_available()
        
        # Check job service (used by both pipelines)
        get_job_service()
        
        # Check legacy Redis queue service (only for old pipeline)
        redis_healthy = True
        redis_error = None
        try:
            queue_service = get_redis_queue_service()
            redis_healthy = queue_service.is_minio_available()
            if not redis_healthy:
                redis_error = queue_service.get_minio_error()
        except RuntimeError as e:
            redis_healthy = False
            redis_error = str(e)

        health_status = {
            "status": "healthy" if taskiq_healthy else "degraded",
            "message": "PDF OCR service is running.",
            "pipelines": {
                "docling": {
                    "status": "healthy" if taskiq_healthy else "unavailable",
                    "message": "Taskiq-based pipeline (URL only, no MinIO required)" + (" (enabled)" if taskiq_healthy else " (disabled)")
                },
                "legacy": {
                    "status": "healthy" if redis_healthy else "unavailable",
                    "message": "Redis/MinIO-based pipeline (deprecated)",
                    "error": redis_error if not redis_healthy else None
                }
            }
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"PDF queuing service health check failed: {e}"
        }


@router.get("/sections/{doc_id}", tags=["PDF OCR"])
async def get_document_sections(doc_id: str):
    """
    Get document structure (sections) for a processed PDF.

    Args:
        doc_id: Document identifier returned from /scrape endpoint

    Returns:
        Document structure including sections, hierarchy, and metadata
    """
    try:
        from tilellm.modules.knowledge_graph.repository.repository import GraphRepository
        from tilellm.modules.pdf_ocr.services.document_structure_extractor import DocumentStructureExtractor

        # Initialize graph repository
        graph_repo = GraphRepository()

        # Query for Document node
        query = """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:CONTAINS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(sub:Section)
        OPTIONAL MATCH (s)-[:CONTAINS]->(e)
        WITH d, s, sub, e
        ORDER BY s.level, sub.level, s.page
        RETURN d.id as doc_id,
               s.id as section_id,
               s.title as title,
               s.level as level,
               s.page as page,
               collect(DISTINCT sub.id) as subsections,
               collect(DISTINCT e.id) as elements
        """

        try:
            result = graph_repo.execute_query(query, {'doc_id': doc_id})

            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {doc_id} not found or not processed"
                )

            # Build hierarchical structure
            sections = []
            for row in result:
                section = {
                    'section_id': row.get('section_id'),
                    'title': row.get('title'),
                    'level': row.get('level'),
                    'page': row.get('page'),
                    'subsections': row.get('subsections', []),
                    'elements': row.get('elements', [])
                }
                sections.append(section)

            return {
                'doc_id': doc_id,
                'sections': sections,
                'metadata': {
                    'total_sections': len(sections)
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Error retrieving document structure: {str(e)}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge graph service unavailable: {str(e)}"
        )


@router.get("/search", tags=["PDF OCR"])
async def search_document_elements(
    doc_id: str,
    query: str,
    element_type: Optional[str] = None,
    top_k: int = 5
):
    """
    Search for elements within a document using vector similarity.

    Args:
        doc_id: Document identifier
        query: Search query text
        element_type: Filter by element type (text, table, image)
        top_k: Number of results to return

    Returns:
        List of matching elements with scores
    """
    try:
        from tilellm.modules.pdf_ocr.logic import process_pdf_document_with_embeddings

        # For now, return a placeholder response
        # Full implementation would use vector store search with doc_id filter

        return {
            'doc_id': doc_id,
            'query': query,
            'results': [],
            'message': 'Search functionality requires vector store integration'
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching document elements: {str(e)}"
        )
