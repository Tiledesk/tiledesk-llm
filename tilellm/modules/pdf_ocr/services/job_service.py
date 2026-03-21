"""
Job service for tracking PDF processing jobs.
This service manages job tracking and status updates, integrating with Taskiq Redis result backend.
"""

import uuid
import time
import logging
from typing import Dict, Optional

from tilellm.modules.pdf_ocr.models.pdf_scraping import (
    PDFScrapingJob, PDFScrapingStatus, PDFScrapingStatusResponse
)

logger = logging.getLogger(__name__)


class PDFJobService:
    """
    Service for managing and tracking PDF processing jobs.
    Integrates with Taskiq's Redis result backend for status tracking.
    """

    def __init__(self):
        self.jobs: Dict[str, PDFScrapingJob] = {}
        self._result_backend = None

    def _get_result_backend(self):
        """Lazy initialization - get result_backend from Taskiq broker."""
        if self._result_backend is None:
            try:
                # Import broker to access result_backend
                from tilellm.modules.task_executor.broker import broker
                
                if broker and broker.result_backend:
                    self._result_backend = broker.result_backend
                    logger.info("PDFJobService: Using Taskiq broker result_backend")
                else:
                    logger.warning("PDFJobService: Taskiq broker result_backend not available")
                    self._result_backend = None
            except Exception as e:
                logger.warning(f"PDFJobService: Could not access result_backend: {e}")
                self._result_backend = None
        
        return self._result_backend

    async def get_status_response(self, job_id: str) -> Optional[PDFScrapingStatusResponse]:
        """
        Get job status response, checking Taskiq result backend for updates.

        Args:
            job_id: Job identifier (same as task_id)

        Returns:
            Status response or None if job not found
        """
        result_backend = self._get_result_backend()

        if result_backend is None:
            logger.warning(f"No result_backend available for job {job_id}")
            return None

        try:
            # Check if result is ready using Taskiq result_backend
            is_ready = await result_backend.is_result_ready(job_id)

            if is_ready:
                result = await result_backend.get_result(job_id)

                if result.is_err:
                    # Task failed - extract error from the light_result structure
                    error_result = result.return_value if hasattr(result, 'return_value') else None
                    error_msg = None
                    if isinstance(error_result, dict):
                        error_msg = error_result.get("error_message", str(error_result.get("result", "Unknown error")))
                    if not error_msg:
                        error_msg = str(result.error) if hasattr(result, 'error') else str(error_result)
                    
                    return PDFScrapingStatusResponse(
                        job_id=job_id,
                        status=PDFScrapingStatus.FAILED,
                        progress=100,
                        message="Processing failed.",
                        result=None,
                        error_message=error_msg
                    )
                else:
                    # Task completed - result.return_value is our light_result structure
                    task_result = result.return_value
                    
                    # Extract fields from light_result format
                    status = PDFScrapingStatus.COMPLETED
                    progress = task_result.get("progress", 100) if isinstance(task_result, dict) else 100
                    message = task_result.get("message", "Processing completed successfully.") if isinstance(task_result, dict) else "Processing completed successfully."
                    result_data = task_result.get("result") if isinstance(task_result, dict) else None
                    error_msg = task_result.get("error_message") if isinstance(task_result, dict) else None
                    
                    # Check if task_result indicates failure
                    if task_result.get("status") == "failed" if isinstance(task_result, dict) else False:
                        status = PDFScrapingStatus.FAILED
                    
                    return PDFScrapingStatusResponse(
                        job_id=job_id,
                        status=status,
                        progress=progress,
                        message=message,
                        result=result_data,
                        error_message=error_msg
                    )
            else:
                # Result not ready - task is still in progress
                return PDFScrapingStatusResponse(
                    job_id=job_id,
                    status=PDFScrapingStatus.PROCESSING,
                    progress=50,
                    message="Job is being processed by a worker.",
                    result=None,
                    error_message=None
                )

        except Exception as e:
            logger.error(f"Error getting task status for {job_id}: {e}", exc_info=True)
            return None

    def register_task(self, task_id: str, doc_id: str, file_name: str) -> None:
        """
        Register a Taskiq task in local memory for tracking.

        Args:
            task_id: Taskiq task ID (returned by .kiq()) – used as the status-polling key.
            doc_id:  Vector store document ID (from request.id) – stable identifier.
            file_name: Original file name.
        """
        job = PDFScrapingJob(
            job_id=task_id,
            file_name=file_name,
            status=PDFScrapingStatus.PENDING,
            created_at=time.time()
        )
        self.jobs[task_id] = job
        logger.info(f"Registered task_id={task_id} doc_id={doc_id} file={file_name}")

    def get_job(self, job_id: str) -> Optional[PDFScrapingJob]:
        """
        Get job by ID from local storage.
        """
        return self.jobs.get(job_id)

    def _get_status_message(self, status: PDFScrapingStatus) -> str:
        """Get status message for job."""
        messages = {
            PDFScrapingStatus.PENDING: "Job has been queued for processing.",
            PDFScrapingStatus.PROCESSING: "Job is being processed by a worker.",
            PDFScrapingStatus.COMPLETED: "Processing completed successfully.",
            PDFScrapingStatus.FAILED: "Processing failed."
        }
        return messages.get(status, "Unknown status")


# --- Singleton instance ---
_job_service: Optional[PDFJobService] = None


def get_job_service() -> PDFJobService:
    """Get or create a singleton job service instance."""
    global _job_service
    if _job_service is None:
        _job_service = PDFJobService()
    return _job_service
