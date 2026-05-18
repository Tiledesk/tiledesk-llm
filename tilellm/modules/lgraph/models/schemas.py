from pydantic import BaseModel, Field, SecretStr
from typing import List, Optional, Dict, Any, Union

from tilellm.models.vector_store import Engine
from tilellm.models.embedding import LlmEmbeddingModel


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

class LGraphBuildRequest(BaseModel):
    """Request to build a light graph over an existing namespace in the vector store."""
    namespace: str = Field(..., description="Tenant namespace in the vector store")
    engine: Engine = Field(..., description="Vector store engine configuration")
    spacy_model: str = Field(default="it_core_news_lg", description="spaCy model for entity extraction")
    use_noun_chunks: bool = Field(default=True, description="Include noun chunks as CONCEPT entities")
    include_entity_types: List[str] = Field(
        default=["PER", "ORG", "LOC", "MISC", "CIG", "CUP", "DATE_IT", "MONEY", "QUANTITY"],
        description="NER entity types to include (empty = all types).",
    )
    npmi_threshold: float = Field(default=0.1, description="Minimum NPMI for entity co-occurrence edges")
    npmi_min_count: int = Field(default=2, description="Minimum co-occurrence count before NPMI is computed")
    overwrite: bool = Field(default=True, description="Delete existing graph before rebuilding")
    # Percorso A: sub-windowing for entity extraction precision
    sub_window_size: int = Field(
        default=300,
        description="Char size of sub-windows for entity extraction (0 = disabled). "
                    "Each vector-store chunk is split into overlapping sub-windows; entities "
                    "from all sub-windows are unioned and attributed to the original LChunk.",
    )
    sub_window_overlap: int = Field(
        default=50,
        description="Char overlap between consecutive sub-windows.",
    )
    webhook_url: Optional[str] = Field(default=None, description="Optional callback URL on completion")


class LGraphBuildResponse(BaseModel):
    status: str
    namespace: str
    graph_name: str
    chunks_processed: int
    entities_created: int
    entity_chunk_edges: int
    entity_entity_edges: int
    message: str = ""


# ---------------------------------------------------------------------------
# Search (PPR)
# ---------------------------------------------------------------------------

class LGraphSearchRequest(BaseModel):
    """Query the light graph using Personalized PageRank."""
    question: str = Field(..., description="Natural language query")
    namespace: str = Field(..., description="Tenant namespace")
    engine: Engine = Field(..., description="Vector store engine (index_name used as graph key)")
    spacy_model: str = Field(default="it_core_news_lg", description="spaCy model matching the build phase")
    use_noun_chunks: bool = Field(default=True)
    include_entity_types: List[str] = Field(
        default=["PER", "ORG", "LOC", "MISC", "CIG", "CUP", "DATE_IT", "MONEY", "QUANTITY"],
    )
    top_k: int = Field(default=5, ge=1, le=50)
    ppr_alpha: float = Field(default=0.85, description="PageRank damping factor")
    ppr_seed_k: int = Field(default=10, description="Number of entity seeds from query")
    ppr_max_iter: int = Field(default=100)


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    metadata_id: str
    source: str
    ppr_score: float


class LGraphSearchResponse(BaseModel):
    chunks: List[ChunkResult]
    entities_found: List[str]
    graph_name: str
    query: str


# ---------------------------------------------------------------------------
# QA (PPR + LLM)
# ---------------------------------------------------------------------------

class LGraphQARequest(BaseModel):
    """PPR retrieval + LLM answer for Italian PA documents."""
    question: str
    namespace: str
    engine: Engine
    # LLM config — same fields expected by inject_llm_chat_async
    gptkey: Optional[SecretStr] = Field(default=None)
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    llm: str = Field(default="openai")
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    temperature: float = Field(default=0.3)
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=2048)
    # PPR params
    spacy_model: str = Field(default="it_core_news_lg")
    use_noun_chunks: bool = Field(default=True)
    include_entity_types: List[str] = Field(
        default=["PER", "ORG", "LOC", "MISC", "CIG", "CUP", "DATE_IT", "MONEY", "QUANTITY"],
    )
    top_k: int = Field(default=10, ge=1, le=100)
    ppr_alpha: float = Field(default=0.85)
    ppr_seed_k: int = Field(default=10)
    ppr_max_iter: int = Field(default=100)
    # Temporal filter (ISO dates "YYYY-MM-DD")
    date_from: Optional[str] = Field(default=None)
    date_to: Optional[str] = Field(default=None)
    system_context: str = Field(default="")
    debug: bool = Field(default=False)


class LGraphQAResponse(BaseModel):
    answer: str
    entities_found: List[str]
    graph_name: str
    chunk_count: int
    chunks_used: Optional[List[ChunkResult]] = Field(default=None, description="Populated when debug=True")


# ---------------------------------------------------------------------------
# Leiden clustering
# ---------------------------------------------------------------------------

class LGraphLeidenRequest(BaseModel):
    """Run Leiden community detection on LEntity CO_OCCURS graph."""
    namespace: str
    engine: Engine
    resolution: float = Field(default=1.0, description="Leiden resolution (higher → more, smaller communities)")
    min_community_size: int = Field(default=3, description="Skip communities smaller than this")
    webhook_url: Optional[str] = Field(default=None)


class LGraphLeidenResponse(BaseModel):
    status: str
    graph_name: str
    community_count: int
    entities_updated: int
    message: str = ""


class LGraphLeidenAsyncTaskResponse(BaseModel):
    task_id: str
    status: str = "queued"


# ---------------------------------------------------------------------------
# Misc (reused across endpoints)
# ---------------------------------------------------------------------------

class LGraphDeleteResponse(BaseModel):
    status: str
    graph_name: str
    message: str


class LGraphNetworkResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]


class LGraphAsyncTaskResponse(BaseModel):
    task_id: str
    status: str = "queued"


class LGraphTaskPollResponse(BaseModel):
    task_id: str
    status: str  # queued | in_progress | success | failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
