"""
Canonical model for the export/md redesign of `tilellm.modules.ingestion`.

`ExtractedDocument` is the single internal representation every converter (PDF,
DOCX, CSV, ...) produces and both serializers (Markdown+frontmatter, JSON) read
from — one model, two projections, per the "entrambi paritari" decision without
duplicating parsing logic.

Frontmatter fields follow the OKF vocabulary (type/title/description/resource/
tags/timestamp + free `extra` extension keys) — curated, doc-level, stable.
`Block.page` / `heading_path` / `block_type` are structural: derived by the
converter at extraction time, volatile, never mixed into frontmatter.

See memory/ingestion_md_redesign.md for the design decisions.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tilellm.models.document_type import DocumentType


class Block(BaseModel):
    """A single content unit of an extracted document (a page, a table, a row...)."""

    content: str
    block_type: str = "text"
    page: Optional[int] = None
    position: Optional[int] = Field(
        default=None,
        description=(
            "Structural position for non-paginated formats (e.g. DOCX paragraph/table "
            "index). Distinct from `page`: never a fabricated page number."
        ),
    )
    heading_path: Optional[str] = None
    order: int = 0


class ExtractedDocument(BaseModel):
    """OKF-vocabulary frontmatter + ordered content blocks."""

    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    resource: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    timestamp: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    blocks: List[Block] = Field(default_factory=list)

    def body_text(self) -> str:
        """Concatenate block contents in `order`, double-newline separated."""
        ordered = sorted(self.blocks, key=lambda b: b.order)
        return "\n\n".join(b.content for b in ordered)


class ExportMdRequest(BaseModel):
    """Request for POST /api/export/md — extraction only, no vector store."""

    type: Optional[DocumentType] = Field(default=DocumentType.AUTO)
    source: Optional[str] = None
    content: Optional[str] = None
    file_name: Optional[str] = None
    file_content: Optional[str] = Field(
        default=None, description="Base64-encoded binary content, or a URL string.",
    )
    format: str = Field(default="md", description="'md' (frontmatter+markdown) or 'json'.")

    # type=url only — same scraping strategy knobs as ItemSingle/add_item
    scrape_type: int = Field(default=0, description="Web-page scraping strategy (see ScrapeType).")
    parameters_scrape_type_4: Optional[Any] = Field(default=None)
    browser_headers: Optional[Dict[str, str]] = Field(default=None)

    # Frontmatter overrides applied after extraction (curated, doc-level)
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
