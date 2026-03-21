"""
PDF OCR module for document parsing using Dolphin/Docling.
Provides PDF scraping functionality for text, tables, and images.

Two pipelines are supported:
1. Docling/Taskiq pipeline (recommended): Does not require MinIO
2. Legacy Redis/MinIO pipeline (deprecated): Requires MinIO
"""

from .controllers import router
from .models.pdf_scraping import (
    PDFScrapingRequest, PDFScrapingResponse, PDFPage, PDFElement, PDFElementType,
    PDFScrapingAcceptResponse, PDFScrapingStatusResponse, PDFScrapingJob, PDFScrapingStatus
)
from .services.job_service import PDFJobService, get_job_service
from .services.taskiq_service import TaskiqService, get_taskiq_service
from .services.redis_queue_service import RedisQueueService, get_redis_queue_service

__all__ = [
    "router",
    "PDFScrapingRequest",
    "PDFScrapingResponse",
    "PDFPage",
    "PDFElement",
    "PDFElementType",
    "PDFScrapingAcceptResponse",
    "PDFScrapingStatusResponse",
    "PDFScrapingJob",
    "PDFScrapingStatus",
    "PDFJobService",
    "get_job_service",
    "TaskiqService",
    "get_taskiq_service",
    "RedisQueueService",
    "get_redis_queue_service"
]