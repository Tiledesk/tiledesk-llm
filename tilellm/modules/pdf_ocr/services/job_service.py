"""
Job service for tracking PDF processing jobs.
This service ONLY manages job tracking and status updates. It does not execute jobs.
"""

import uuid
import time
from typing import Dict, Optional

from tilellm.modules.pdf_ocr.models.pdf_scraping import (
    PDFScrapingJob, PDFScrapingStatus, PDFScrapingStatusResponse
)

class PDFJobService:
    """
    Service for managing and tracking PDF processing jobs in memory.
    """
    
    def __init__(self):
        self.jobs: Dict[str, PDFScrapingJob] = {}
    
    def create_job(self, file_name: str) -> str:
        """
        Create a new PDF processing job and store it in memory.
        
        Args:
            file_name: The name of the file for the job.
            
        Returns:
            The generated Job ID.
        """
        job_id = str(uuid.uuid4())
        
        job = PDFScrapingJob(
            job_id=job_id,
            file_name=file_name,
            status=PDFScrapingStatus.PENDING,
            created_at=time.time()
        )
        
        self.jobs[job_id] = job
        print(f"Created job {job_id} for file {file_name} with status PENDING.")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[PDFScrapingJob]:
        """
        Get job by ID.
        """
        return self.jobs.get(job_id)

    def get_status_response(self, job_id: str) -> Optional[PDFScrapingStatusResponse]:
        """
        Get job status response.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Status response or None if job not found
        """
        job = self.get_job(job_id)
        if not job:
            return None
        
        # In this new architecture, this service won't know about completion.
        # The status will remain PENDING unless updated by an external mechanism (e.g., a webhook).
        progress = 5 # Represents "queued"
        if job.status == PDFScrapingStatus.PROCESSING:
            progress = 50
        elif job.status == PDFScrapingStatus.COMPLETED:
            progress = 100
        
        return PDFScrapingStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=progress,
            message=self._get_status_message(job.status),
            result=job.result,
            error_message=job.error_message
        )
    
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