"""
Models for PDF scraping functionality using Dolphin.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, SecretStr

from tilellm.models import LlmEmbeddingModel
from tilellm.models.llm import ItemSingle


class PDFScrapingRequest(ItemSingle):
    """
    Model for PDF scraping request. Inherits from ItemSingle for RAG integration.
    """
    llm: Optional[str] = Field(default="openai")
    gptkey: Optional[SecretStr] = "sk"
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.7)
    top_k: int = Field(default=5)
    max_tokens: int = Field(default=512)
    top_p: Optional[float] = Field(default=1.0)
    debug: bool = Field(default_factory=lambda: False)
    file_name: str = Field(..., description="Original file name with extension (e.g., 'document.pdf').")
    file_content: str = Field(..., description="File content encoded as Base64 or URL (http/https).")
    include_images: bool = Field(True, description="Whether to include image descriptions.")
    include_tables: bool = Field(True, description="Whether to include table extraction.")
    include_text: bool = Field(True, description="Whether to include text extraction.")
    include_formulas: bool = Field(True, description="Whether to include formula extraction.")
    index_tables_to_vector_store: bool = Field(True, description="Whether to index table descriptions to vector store.")
    index_images_to_vector_store: bool = Field(True, description="Whether to index image captions to vector store.")
    max_batch_size: Optional[int] = Field(16, description="Maximum batch size for processing elements.")
    webhook_url: Optional[str] = Field(None, description="URL to notify when processing is complete.")
    callback_token: Optional[str] = Field(None, description="Token for webhook authentication.")
    unstructured_config: Optional[Dict[str, Any]] = Field(None, description="Unstructured configuration overrides.")
    use_docling: bool = Field(False, description="Use new Docling-based advanced pipeline.")
    extract_entities: bool = Field(False, description="Whether to extract semantic entities using GraphRAG.")
    extract_structure: bool = Field(True, description="Whether to extract hierarchical document structure.")

    def is_url(self) -> bool:
        """Check if file_content is a URL."""
        return self.file_content.startswith(('http://', 'https://'))


class PDFElementType(str, Enum):
    """
    Enum for PDF element types.
    """
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    CODE = "code"
    HEADING = "heading"
    LIST = "list"


class PDFElement(BaseModel):
    """
    Model for a single PDF element.
    """
    element_type: PDFElementType = Field(..., description="Type of the element.")
    content: str = Field(..., description="Extracted content of the element.")
    page_number: int = Field(..., description="Page number where the element was found.")
    reading_order: int = Field(..., description="Reading order on the page.")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates [x1, y1, x2, y2].")
    confidence: Optional[float] = Field(None, description="Confidence score for the extraction.")


class PDFPage(BaseModel):
    """
    Model for a single PDF page.
    """
    page_number: int = Field(..., description="Page number.")
    elements: List[PDFElement] = Field(..., description="Elements extracted from this page.")
    markdown_content: str = Field(..., description="Markdown representation of the page content.")


class PDFScrapingResponse(BaseModel):
    """
    Model for PDF scraping response.
    """
    file_name: str = Field(..., description="Original file name.")
    total_pages: int = Field(..., description="Total number of pages processed.")
    pages: List[PDFPage] = Field(..., description="List of processed pages.")
    markdown_content: str = Field(..., description="Complete markdown representation of the document.")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the processing.")


class PDFScrapingStatus(str, Enum):
    """
    Enum for PDF scraping status.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PDFScrapingJob(BaseModel):
    """
    Model for PDF scraping job tracking.
    """
    job_id: str = Field(..., description="Unique job identifier.")
    file_name: str = Field(..., description="Original file name.")
    status: PDFScrapingStatus = Field(..., description="Current job status.")
    created_at: float = Field(..., description="Job creation timestamp.")
    started_at: Optional[float] = Field(None, description="Job start timestamp.")
    completed_at: Optional[float] = Field(None, description="Job completion timestamp.")
    error_message: Optional[str] = Field(None, description="Error message if job failed.")
    result: Optional[PDFScrapingResponse] = Field(None, description="Processing result if completed.")


class PDFScrapingAcceptResponse(BaseModel):
    """
    Model for PDF scraping acceptance response.
    """
    job_id: str = Field(..., description="Unique job identifier.")
    status: str = Field("accepted", description="Job acceptance status.")
    message: str = Field(..., description="Acceptance message.")
    estimated_time: Optional[int] = Field(None, description="Estimated processing time in seconds.")


class PDFScrapingStatusResponse(BaseModel):
    """
    Model for PDF scraping status response.
    """
    job_id: str = Field(..., description="Unique job identifier.")
    status: PDFScrapingStatus = Field(..., description="Current job status.")
    progress: Optional[float] = Field(None, description="Processing progress (0-100).")
    message: Optional[str] = Field(None, description="Status message.")
    result: Optional[PDFScrapingResponse] = Field(None, description="Processing result if completed.")
    error_message: Optional[str] = Field(None, description="Error message if failed.")