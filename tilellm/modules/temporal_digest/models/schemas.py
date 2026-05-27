"""
Temporal Digest — Pydantic models.

Generic module for time-based aggregation of document streams.
Use cases: PA administrative acts, legal documents, financial reports, news.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, SecretStr

from tilellm.models import Engine
from tilellm.models.embedding import LlmEmbeddingModel
from tilellm.models.llm import TEIConfig, PineconeRerankerConfig


class DigestGenerationRequest(BaseModel):
    """Request to generate one or more temporal digests for a namespace."""

    namespace: str = Field(..., description="Vector store namespace (e.g. ASL identifier).")
    date_from: date = Field(..., description="Start date (inclusive) for chunk retrieval.")
    date_to: Optional[date] = Field(
        default=None,
        description="End date (inclusive). Defaults to date_from (single-day digest).",
    )
    granularity: Literal["daily", "weekly", "monthly"] = Field(
        default="daily",
        description="Aggregation granularity. 'daily' produces one digest per day in the range.",
    )
    engine: Engine
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    gptkey: Optional[SecretStr] = Field(default=None)
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    llm: str = Field(default="openai")
    temperature: float = Field(default=0.0)
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=2048)
    debug: bool = Field(default=False)
    top_k: int = Field(
        default=1000,
        description="Max chunks to retrieve per date window. Increase for very active namespaces.",
    )
    force_regenerate: bool = Field(
        default=False,
        description="Re-generate even if a digest for this period already exists.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for digest generation. Overrides domain default.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Pre-built domain prompt key (e.g. 'pa_italiana', 'legal', 'generic').",
    )
    date_metadata_field: str = Field(
        default="date",
        description="Name of the metadata field that stores the document date in the vector store.",
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="URL notified on completion (or failure) when the task runs asynchronously via TaskIQ.",
    )
    tags: Optional[List[str]] = Field(default=None)
    # lgraph integration — augment chunk retrieval with DATE_IT entities from FalkorDB
    use_lgraph: bool = Field(
        default=False,
        description=(
            "When True, also retrieves chunks linked to DATE_IT entities in the lgraph "
            "that match the date window, plus community-sibling chunks (LinearRAG-style). "
            "Also injects community summary context from {namespace}__lgraph_communities. "
            "Requires a built lgraph for this namespace+index."
        ),
    )
    lgraph_spacy_model: str = Field(
        default="it_core_news_lg",
        description="spaCy model used when building the lgraph (must match build phase).",
    )
    classify_act_types: bool = Field(
        default=True,
        description=(
            "When True, run an LLM classifier to infer act_type for chunks that do not "
            "have it populated in metadata. Improves digest categorization quality. "
            "Disable to reduce LLM cost on high-volume batches."
        ),
    )


class DigestGenerationResult(BaseModel):
    """Result of a digest generation for a single time window."""

    namespace: str
    date_from: str
    date_to: str
    granularity: str
    content: str = Field(description="Generated digest text.")
    chunk_count: int = Field(description="Number of source chunks aggregated.")
    act_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Count per act_type extracted from chunk metadata (if available).",
    )
    total_amount: Optional[float] = Field(
        default=None,
        description="Sum of 'amount' metadata field across chunks (if available).",
    )
    digest_vector_id: Optional[str] = Field(
        default=None,
        description="Vector ID of the indexed digest (for retrieval).",
    )
    already_existed: bool = Field(
        default=False,
        description="True if a digest for this window already existed and was not regenerated.",
    )


class DigestGenerationResponse(BaseModel):
    """Aggregated response for a DigestGenerationRequest (may span multiple windows)."""

    namespace: str
    digests: List[DigestGenerationResult]
    total_chunks_processed: int
    total_windows: int
    skipped_windows: int = Field(
        default=0,
        description="Windows skipped because a digest already existed and force_regenerate=False.",
    )


class DigestQueryRequest(BaseModel):
    """Request to query across digests and/or raw chunks."""

    question: str
    namespace: str
    date_from: Optional[date] = Field(
        default=None,
        description="If set, restricts retrieval to this date range.",
    )
    date_to: Optional[date] = Field(default=None)
    engine: Engine
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    gptkey: Optional[SecretStr] = Field(default=None)
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    llm: str = Field(default="openai")
    temperature: float = Field(default=0.3)
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=1024)
    debug: bool = Field(default=False)
    top_k: int = Field(default=5)
    sparse_encoder: Union[str, TEIConfig, None] = Field(
        default=None,
        description=(
            "Sparse encoder for hybrid search. "
            "String: model name (e.g. 'splade', 'bge-m3'). "
            "TEIConfig: remote TEI endpoint ({provider:'tei', name:'...', url:'http://...'})."
        ),
    )
    search_type: str = Field(
        default="similarity",
        description=(
            "Vector search type: 'similarity' (dense only) or 'hybrid' (dense + sparse). "
            "Set to 'hybrid' when sparse_encoder is provided."
        ),
    )
    reranking: Union[bool, TEIConfig, PineconeRerankerConfig] = Field(
        default=False,
        description=(
            "Reranking strategy. "
            "False: disabled. "
            "True: use local CrossEncoder (reranker_model). "
            "TEIConfig: remote TEI reranker endpoint. "
            "PineconeRerankerConfig: Pinecone inference reranker."
        ),
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Local CrossEncoder model name. Used only when reranking=True (not TEI/Pinecone).",
    )
    reranking_multiplier: int = Field(
        default=3,
        description=(
            "When reranking is enabled, retrieve top_k × reranking_multiplier candidates "
            "before reranking down to top_k."
        ),
    )
    query_mode: Literal["auto", "temporal", "semantic"] = Field(
        default="auto",
        description=(
            "'auto' classifies the query and routes accordingly. "
            "'temporal' forces digest retrieval. "
            "'semantic' forces vector search on raw chunks."
        ),
    )
    date_metadata_field: str = Field(default="date")
    tags: Optional[List[str]] = Field(default=None)
    chat_history_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Conversation history keyed by turn index. Each entry: {question, answer}.",
    )
    max_history_messages: int = Field(default=10)


class DigestAgentRequest(BaseModel):
    """
    Agentic digest query — the LLM extracts date range and query_mode from the
    free-form question and conversation history, then executes the optimal retrieval path.
    """
    question: str
    namespace: str
    engine: Engine
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    gptkey: Optional[SecretStr] = Field(default=None)
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    llm: str = Field(default="openai")
    temperature: float = Field(default=0.3)
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=1024)
    debug: bool = Field(default=False)
    top_k: int = Field(default=5)
    sparse_encoder: Union[str, TEIConfig, None] = Field(default=None)
    search_type: str = Field(default="similarity")
    reranking: Union[bool, TEIConfig, PineconeRerankerConfig] = Field(default=False)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranking_multiplier: int = Field(default=3)
    date_metadata_field: str = Field(default="date")
    tags: Optional[List[str]] = Field(default=None)
    chat_history_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Conversation history keyed by turn index. Each entry: {question, answer}.",
    )
    max_history_messages: int = Field(default=10)
    today: Optional[date] = Field(
        default=None,
        description="Reference date for relative expressions ('ieri', 'la settimana scorsa'). Defaults to server date.",
    )


class DigestQueryResponse(BaseModel):
    """Response to a digest query."""

    answer: str
    query_mode: str = Field(description="Resolved mode: 'temporal' or 'semantic'.")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source metadata for each retrieved chunk/digest.",
    )
    digests_used: List[str] = Field(
        default_factory=list,
        description="Date strings of digests consulted (temporal mode).",
    )
    chunk_count: int = Field(default=0)


class DigestAgentResponse(DigestQueryResponse):
    """Response to an agentic digest query — extends DigestQueryResponse with extraction details."""

    extracted_date_from: Optional[str] = Field(
        default=None, description="Date extracted by the agent (ISO format)."
    )
    extracted_date_to: Optional[str] = Field(
        default=None, description="Date extracted by the agent (ISO format)."
    )
    extracted_query_mode: str = Field(
        default="auto", description="Query mode chosen by the agent."
    )
    agent_reasoning: Optional[str] = Field(
        default=None, description="Agent's reasoning for parameter extraction."
    )
