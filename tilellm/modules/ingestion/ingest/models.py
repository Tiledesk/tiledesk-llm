"""
Request/response models for POST /api/ingest/md — the F2 half of the redesign:
frontmatter (from export/md) -> real vector-store write via `aadd_documents`.

`IngestConfig` holds every "how to ingest" field and is also reused directly
by /api/v2/ingestion (api_v2/services/ingestion_v2_service.py), which already
has an in-memory ExtractedDocument (from export_document) and has no md/json
text to round-trip through — only IngestMdRequest (the public /api/ingest/md
contract) adds the mutually-exclusive source fields on top.

See memory/ingestion_md_redesign.md for the `aadd_documents` vs `add_item`
decision (additional_metadata gap on the standard path) and the per-backend
hybrid-behavior caveat.
"""
from typing import List, Optional, Union

from pydantic import BaseModel, Field, SecretStr, model_validator

from tilellm.models import Engine, LlmEmbeddingModel
from tilellm.models.llm import SituatedContextConfig, TEIConfig

_SOURCE_FIELDS = ("md", "md_url", "json_content", "json_url")


class IngestConfig(BaseModel):
    """How to ingest an already-produced ExtractedDocument — no opinion on
    where the document comes from."""

    id: str = Field(..., description="Document id — used as metadata_id for dedup/re-ingest.")
    namespace: str
    engine: Engine

    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    hybrid: bool = Field(default=False, description="If True, also generates sparse vectors.")
    sparse_encoder: Union[str, TEIConfig, None] = Field(default="splade")

    tags: Optional[List[str]] = Field(
        default=None,
        description="Overrides the frontmatter 'tags' when provided; otherwise the document's own tags are used.",
    )
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=400)
    table_strategy: str = Field(default="adaptive")

    situated_context: Optional[SituatedContextConfig] = Field(
        default=None,
        description="Dedicated LLM configuration for Contextual Retrieval (situated context), same as ItemSingle.situated_context.",
    )

    gptkey: Optional[SecretStr] = Field(default=None)
    id_project: Optional[str] = Field(default=None)
    request_id: Optional[str] = Field(default=None)
    debug: bool = Field(default=False)

    # Required by @inject_llm_chat_async (builds llm_embeddings — the only thing F2
    # actually uses — but the decorator unconditionally also builds a chat model;
    # no embeddings-only injector exists in this codebase, see memory/ingestion_md_redesign).
    llm: Optional[str] = Field(default="openai")
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0)
    top_p: Optional[float] = Field(default=1.0)
    max_tokens: int = Field(default=512)


class IngestMdRequest(IngestConfig):
    """Ingest a previously-exported document (Markdown+frontmatter or JSON)."""

    # Exactly one of these four
    md: Optional[str] = Field(default=None, description="Inline Markdown+frontmatter.")
    md_url: Optional[str] = Field(default=None, description="URL to a Markdown+frontmatter document.")
    json_content: Optional[str] = Field(default=None, description="Inline JSON (ExtractedDocument).")
    json_url: Optional[str] = Field(default=None, description="URL to a JSON (ExtractedDocument) document.")

    @model_validator(mode="after")
    def _validate_source(self) -> "IngestMdRequest":
        provided = sum(bool(getattr(self, f) and getattr(self, f).strip()) for f in _SOURCE_FIELDS)
        if provided > 1:
            raise ValueError("'md', 'md_url', 'json' e 'json_url' sono mutuamente esclusivi.")
        if provided == 0:
            raise ValueError("Fornire uno tra 'md', 'md_url', 'json' o 'json_url'.")
        return self


class IngestMdResult(BaseModel):
    """Response of POST /api/ingest/md."""

    id: str
    namespace: str
    chunks_indexed: int
    chunk_ids: List[str] = Field(default_factory=list)
