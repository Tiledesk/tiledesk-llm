"""
Taskiq service for PDF processing with Docling pipeline.
This service handles PDF submission for the new Docling-based pipeline.

Flow:
1. Client submits PDF URL (not Base64)
2. Taskiq message contains the URL
3. Worker downloads the file to temp
4. If MinIO is available, upload original PDF to MinIO
5. Start Docling ingestion pipeline
"""

import os
import logging
from typing import Optional, Dict, Any


from tilellm.shared.llm_config import serialize_with_secrets
from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest

try:
    from tilellm.modules.task_executor.broker import broker as _broker  # noqa: F401
    TASKIQ_AVAILABLE = True
except Exception:
    TASKIQ_AVAILABLE = False

ENABLE_TASKIQ = os.environ.get("ENABLE_TASKIQ", "false").lower() == "true"
ENABLE_TASKIQ = ENABLE_TASKIQ and TASKIQ_AVAILABLE

logger = logging.getLogger(__name__)


class TaskiqService:
    """
    Service for submitting PDF processing jobs to Taskiq queue.
    
    This service is used by the new Docling-based pipeline.
    
    IMPORTANT: PDFs must be submitted as URLs (not Base64).
    The worker will download the file and optionally upload to MinIO.
    """

    def __init__(self):
        """Initialize the Taskiq service."""
        self.taskiq_available = TASKIQ_AVAILABLE
        self.enabled = ENABLE_TASKIQ
        
        if self.enabled and not self.taskiq_available:
            logger.warning("Taskiq is enabled but not available")
        
        logger.info(f"TaskiqService initialized: enabled={self.enabled}, available={self.taskiq_available}")



    async def submit(
        self,
        doc_id: str,
        request: PDFScrapingRequest
    ) -> Dict[str, Any]:
        """
        Submit a PDF processing job to Taskiq queue.

        Args:
            doc_id: Document identifier used in the vector store (from request.id).
                    This is stable across re-submissions so that the old version of
                    the document is replaced (same namespace → aadd_documents deletes
                    the previous chunks before inserting the new ones).
            request: PDFScrapingRequest with document configuration (must have URL in file_content)

        Returns:
            Dict with ``task_id`` (Taskiq task ID – use this for /status polling)
            and ``doc_id`` (vector store document ID).

        Raises:
            RuntimeError: If Taskiq is not available
            ValueError: If request is invalid (e.g., not a URL)
        """
        if not self.taskiq_available:
            raise RuntimeError("Taskiq is not available")

        if not self.enabled:
            raise RuntimeError("Taskiq is not enabled")

        # Validate that file_content is a URL (not Base64)
        if not request.is_url():
            raise ValueError(
                "PDF file_content must be a URL (http:// or https://) for Taskiq pipeline. "
                "Base64 content is not supported."
            )

        # Import here to avoid circular imports
        from tilellm.modules.task_executor.tasks import process_pdf_document_task

        # doc_id is the stable document identity in the vector store.
        config = request.model_dump(mode='python')
        config['id'] = doc_id

        logger.info(f"Submitting PDF URL to Taskiq: {request.file_content}, doc_id={doc_id}")

        # Submit and capture the AsyncTaskiqTask so we can expose its auto-generated
        # task_id as the status-tracking key stored in the result backend.
        task_payload = serialize_with_secrets(config)

        task = await process_pdf_document_task.kiq(
            doc_id=doc_id,
            bucket_name=None,  # Worker will handle MinIO upload
            object_name=None,
            webhook_url=request.webhook_url,
            config=task_payload
        )

        task_id = task.task_id
        logger.info(f"Queued Taskiq task_id={task_id} for doc_id={doc_id}")

        return {
            "task_id": task_id,   # Taskiq task ID – use this for /status polling
            "doc_id": doc_id,     # Vector store document ID
            "message": "PDF processing job (Advanced Docling) has been successfully queued.",
            "estimated_time": 120,
        }

    async def process_synchronously(
        self,
        request: PDFScrapingRequest,
        bucket_name: Optional[str] = None,
        object_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF synchronously (fallback when Taskiq is not available).

        Args:
            request: PDFScrapingRequest with document configuration
            bucket_name: Optional MinIO bucket name (if already uploaded)
            object_name: Optional MinIO object name (if already uploaded)

        Returns:
            Dict with processing result
        """
        logger.info(f"Processing PDF synchronously (Taskiq disabled/unavailable) for job {request.id}")

        # Import here to avoid circular imports
        from tilellm.modules.pdf_ocr.logic import process_pdf_document_with_embeddings

        result = await process_pdf_document_with_embeddings(
            question=request,
            bucket_name=bucket_name,
            object_name=object_name
        )

        return {
            "job_id": request.id,
            "message": "PDF processing completed synchronously (Taskiq disabled).",
            "result": result
        }

    def is_available(self) -> bool:
        """Check if Taskiq is available and enabled."""
        return self.taskiq_available and self.enabled


# --- Singleton instance ---
_taskiq_service: Optional[TaskiqService] = None


def get_taskiq_service() -> TaskiqService:
    """Get or create a singleton instance of the TaskiqService."""
    global _taskiq_service
    if _taskiq_service is None:
        _taskiq_service = TaskiqService()
    return _taskiq_service
