"""
CommonChunkMetadata: Shared metadata schema for all indexed chunks.
Used by both pdf_ocr and scrape/single pipelines to ensure consistent metadata fields.
"""
from typing import List, Optional
from pydantic import BaseModel


class CommonChunkMetadata(BaseModel):
    """
    Canonical metadata schema shared across all pipelines and vector stores.
    All fields either required or have safe defaults so retrieval never hits KeyError.
    Extra fields are allowed for pipeline-specific additions.
    """
    model_config = {"extra": "allow"}

    # --- Identity (required) ---
    id: str
    metadata_id: str
    doc_id: str
    namespace: str

    # --- Source info ---
    source: str = ""
    file_name: str = ""
    file_content: str = ""

    # --- Position ---
    page: int = 1
    heading_path: str = ""

    # --- Classification ---
    type: str = "text"
    chunk_type: str = "text"

    # --- Cross-modal refs (pdf_ocr) ---
    ref_tables: str = "[]"
    ref_images: str = "[]"
    surrounding_text: str = ""

    # --- Asset refs (pdf_ocr) ---
    table_id: str = ""
    image_id: str = ""
    parquet_path: str = ""
    md_path: str = ""
    path: str = ""

    # --- Quality flags ---
    has_situated_context: bool = False

    # --- Optional / list fields ---
    tags: Optional[List[str]] = None
    answerable_questions: str = "[]"
    columns: str = "[]"

    def to_metadata_dict(self) -> dict:
        """Return a flat dict suitable for vector store metadata, excluding None values."""
        return self.model_dump(exclude_none=True)
