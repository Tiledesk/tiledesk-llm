import io
import os
import base64
import requests
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import SecretStr

from tilellm.modules.pdf_ocr.models.pdf_scraping import (
    PDFScrapingRequest, PDFScrapingAcceptResponse, PDFScrapingStatusResponse
)
from tilellm.modules.pdf_ocr.services.redis_queue_service import get_redis_queue_service
from tilellm.modules.pdf_ocr.services.job_service import get_job_service
#from tilellm.modules.pdf_ocr.tasks import process_pdf_document_task
from tilellm.modules.task_executor.tasks import process_pdf_document_task
from tilellm.modules.pdf_ocr.logic import process_pdf_document_with_embeddings
from tilellm.shared.llm_config import serialize_with_secrets

ENABLE_TASKIQ = os.environ.get("ENABLE_TASKIQ", "false").lower() == "true"
TASKIQ_AVAILABLE = True  # Assuming True since we import it, or could use try-except block like in KG

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
    - **file_content**: File content encoded as Base64 or a public URL (http/https)
    - **webhook_url**: Optional URL to notify when processing is complete
    - **use_docling**: If True, uses the new advanced Docling-based pipeline.
    
    Returns a job ID for tracking the processing status.
    """
    try:
        # Get the required services
        queue_service = get_redis_queue_service()
        job_service = get_job_service()
        
        # 1. Create a job entry to get a unique ID
        job_id = job_service.create_job(file_name=request.file_name)
        
        if request.use_docling:
            # New Advanced Pipeline
            
            # Prepare content
            if request.is_url():
                response = requests.get(request.file_content, timeout=30)
                response.raise_for_status()
                pdf_content = response.content
            else:
                pdf_content = base64.b64decode(request.file_content)
                
            # Upload to MinIO (reusing queue service's client for convenience)
            object_name = f"{job_id}.pdf"
            bucket_name = queue_service.minio_bucket
            
            queue_service.minio_client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=io.BytesIO(pdf_content),
                length=len(pdf_content),
                content_type='application/pdf'
            )
            
            # Prepare config with job ID as document ID
            config = request.model_dump(mode='python') # Original dump, serialize later if needed
            config['id'] = job_id
            
            if ENABLE_TASKIQ and TASKIQ_AVAILABLE:
                # Trigger Taskiq task
                task_payload = serialize_with_secrets(config)
                await process_pdf_document_task.kiq(
                    doc_id=job_id,
                    bucket_name=bucket_name,
                    object_name=object_name,
                    webhook_url=request.webhook_url,
                    config=task_payload
                )
                message = "PDF processing job (Advanced Docling) has been successfully queued."
            else:
                # Fallback: Process synchronously (blocking)
                # Note: This will block the API response until processing finishes!
                # We need to construct the request object again or pass config
                # process_pdf_document_with_embeddings expects 'request' object
                
                # Update request object with ID if not present
                request.id = job_id
                
                # We already uploaded to MinIO, so use that
                await process_pdf_document_with_embeddings(
                    question=request,
                    bucket_name=bucket_name,
                    object_name=object_name
                )
                message = "PDF processing completed synchronously (Taskiq disabled)."
            
        else:
            # Old Pipeline
            # 2. Submit the job to the Redis queue for the worker to process
            await queue_service.submit(job_id=job_id, request=request)
            message = "PDF processing job has been successfully queued."
        
        return PDFScrapingAcceptResponse(
            job_id=job_id,
            message=message,
            estimated_time=120  # A rough estimate in seconds
        )
        
    except RuntimeError as e:
        # This will catch Redis connection errors from the service constructor
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
        status_response = job_service.get_status_response(job_id)
        
        if not status_response:
            raise HTTPException(
                status_code=404,
                detail=f"Job with ID '{job_id}' not found"
            )
        
        return status_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")


@router.get("/health", tags=["PDF OCR"])
async def health_check():
    """
    Check if the PDF OCR queuing service and its dependencies are healthy.
    """
    try:
        # Check connection to Redis via the queue service
        get_redis_queue_service()
        # Check in-memory job service
        get_job_service()
        
        return {
            "status": "healthy",
            "message": "PDF queuing service is running and connected to Redis."
        }
    except RuntimeError as e:
        return {
            "status": "unhealthy",
            "message": f"PDF queuing service is not available: {e}"
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