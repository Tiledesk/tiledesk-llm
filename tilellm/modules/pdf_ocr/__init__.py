"""
PDF OCR module for document parsing using Dolphin.
Provides PDF scraping functionality for text, tables, and images.
"""

from .controllers import router
from .models.pdf_scraping import (
    PDFScrapingRequest, PDFScrapingResponse, PDFPage, PDFElement, PDFElementType,
    PDFScrapingAcceptResponse, PDFScrapingStatusResponse, PDFScrapingJob, PDFScrapingStatus
)
from .services.job_service import PDFJobService, get_job_service

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
    "get_job_service"
]