"""
API request and response schemas for Knowledge Graph endpoints.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, model_validator
from tilellm.models import Engine, QuestionAnswer
from tilellm.models.schemas.multimodal_content import MultimodalContent


class GraphQARequest(QuestionAnswer):
    """
    Request model for Graph QA endpoint.
    Extends QuestionAnswer to include LLM configuration for GraphRAG QA.
    """
    max_results: Optional[int] = 10
    index_name: Optional[str] = None  # Optional index_name for graph partition
    graph_db_name: Optional[str] = Field(default=None, description="Graph name")
    creation_prompt: str = Field(default="generic", description="Optional prompt for creation")
    # Note: llm_key, llm, model, etc. are inherited from QuestionAnswer
    @model_validator(mode='after')
    def set_graph_db_name(self) -> 'GraphQARequest':
        if not self.graph_db_name:
            parts = [self.namespace]
            if self.creation_prompt:
                parts.append(self.creation_prompt)
            # Not including index_name in graph_db_name to maintain consistency with report namespace
            self.graph_db_name = "-".join(parts)
        return self


class GraphQAResponse(BaseModel):
    """Response model for Graph QA endpoint"""
    answer: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    query_used: str
    chat_history_dict: Optional[Dict[str, Any]] = None


class GraphCreateRequest(QuestionAnswer):
    """
    Request model for Graph creation endpoint.
    Extends QuestionAnswer to include LLM configuration for GraphRAG extraction.
    """
    namespace: str
    index_name: Optional[str] = None  # Optional index_name for graph partition
    engine: Optional[Engine] = None  # Engine configuration for vector store
    limit: Optional[int] = 100
    overwrite: Optional[bool] = False
    creation_prompt: str = Field(default="generic", description="Optional prompt for creation")
    webhook_url: Optional[str] = Field(default=None, description="URL to call when task is finished")


    # Keep same type as parent but with default empty string (not needed for creation)
    question: Optional[Union[str, List[MultimodalContent]]] = Field(default="", description="Optional prompt for extraction guidance")
    graph_db_name: Optional[str] = Field(default=None, description="Graph name")

    # Batch/parallelism tuning (vLLM-optimized defaults)
    extraction_concurrency: int = Field(default=10, description="Max concurrent LLM extraction calls (raise for vLLM, lower for cloud APIs)")
    chunk_window_size: int = Field(default=500, description="Number of chunks processed per window before writing to graph")
    batch_size: int = Field(default=100, description="Number of nodes/relationships per UNWIND batch write")

    @model_validator(mode='after')
    def set_graph_db_name(self) -> 'GraphCreateRequest':
        if not self.graph_db_name:
            parts = [self.namespace]
            if self.creation_prompt:
                parts.append(self.creation_prompt)
            # Not including index_name in graph_db_name to maintain consistency with report namespace
            self.graph_db_name = "-".join(parts)
        return self
    # Note: llm_key, llm, model, etc. are inherited from QuestionAnswer
    

class GraphCreateResponse(BaseModel):
    """Response model for Graph creation endpoint"""
    namespace: str
    chunks_processed: int
    nodes_created: int
    relationships_created: int
    status: str


class AsyncTaskResponse(BaseModel):
    """Response model for Async Task submission"""
    task_id: str
    status: str = "queued"
    message: str = "Task submitted successfully"


class TaskPollResponse(BaseModel):
    """Response model for Task Polling"""
    task_id: str
    status: str  # queued, in_progress, success, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GraphClusterRequest(QuestionAnswer):
    """
    Request model for Graph clustering endpoint.
    Uses QuestionToLLM for LLM configuration to generate community reports.
    """
    question: Union[str, List[MultimodalContent]] = Field(default="", description="Optional prompt for clustering guidance")
    level: Optional[int] = 0
    namespace: Optional[str] = None
    index_name: Optional[str] = None
    creation_prompt: str = Field(default="generic", description="Optional prompt for creation")
    graph_db_name: Optional[str] = Field(default=None, description="Graph name")
    engine: Optional[Engine] = None  # Added for report indexing
    overwrite: Optional[bool] = Field(
        default=True,
        description="If True, removes existing community reports before regeneration (default: True)"
    )
    webhook_url: Optional[str] = Field(default=None, description="URL to call when task is finished")
    resolutions: Optional[List[float]] = Field(
        default=None,
        description="Leiden resolution per level [L0, L1, L2]. Lower = fewer, larger communities. "
                    "Defaults to [0.05, 0.15, 0.35] (recommended for sparse graphs)."
    )
    min_community_size: int = Field(
        default=8,
        description="Minimum number of nodes for a community to generate a report. "
                    "Communities smaller than this are discarded."
    )
    max_community_prompt_chars: int = Field(
        default=18000,
        description="Maximum characters for entities+relationships in each community report prompt. "
                    "~4 chars/token → 18000 ≈ 4500 tokens. Lower this for small-context models."
    )

    @model_validator(mode='after')
    def set_graph_db_name_cluster(self) -> 'GraphClusterRequest':
        if not self.graph_db_name:
            parts = [self.namespace] if self.namespace else []
            if self.creation_prompt:
                parts.append(self.creation_prompt)
            # Not including index_name in graph_db_name to maintain consistency with report namespace
            self.graph_db_name = "-".join(parts) if parts else "knowledge_graph"
        return self

class GraphClusterResponse(BaseModel):
    """Response model for Graph clustering endpoint"""
    status: str
    communities_detected: int
    reports_created: int
    message: Optional[str] = None


class CommunityQAResponse(BaseModel):
    """Response model for Community/Global search QA"""
    answer: str
    reports_used: int = 0
    chat_history_dict: Optional[Dict[str, Any]] = None




class GraphQAAdvancedRequest(GraphQARequest):
    """
    Advanced Graph QA request with retrieval parameters.
    """
    vector_weight: Optional[float] = 1.0
    keyword_weight: Optional[float] = 1.0
    graph_weight: Optional[float] = 1.0
    query_type: Optional[str] = None  # "exploratory", "technical", "relational"
    use_community_search: Optional[bool] = False
    max_expansion_nodes: Optional[int] = 20
    use_reranking: Optional[bool] = True
    creation_prompt: Optional[str] = Field(
        default=None,
        description="Domain identifier for knowledge graph schema (e.g., 'debt_recovery', 'generic'). "
                   "If None, uses 'generic' domain. Must match the creation_prompt used when creating the graph."
    )



class GraphQAAdvancedResponse(BaseModel):
    """Advanced Graph QA response"""
    answer: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    query_used: str
    retrieval_strategy: str
    scores: Dict[str, Any]
    expanded_nodes: List[Dict[str, Any]]
    expanded_relationships: List[Dict[str, Any]]
    chat_history_dict: Optional[Dict[str, Any]] = None


class AddDocumentRequest(QuestionAnswer):
    """
    Request model for adding a document to the knowledge graph by metadata_id.
    Retrieves all chunks of the document from vector store and extracts entities/relationships.
    Enables incremental graph updates without reimporting the entire namespace.
    """
    metadata_id: str = Field(..., description="Unique identifier for the document in vector store")
    namespace: str = Field(..., description="Namespace for the graph (e.g., 'bancaitalia')")
    engine: Engine = Field(..., description="Engine configuration (must include name, index_name, and type/deployment)")
    deduplicate_entities: Optional[bool] = Field(default=True, description="If True, reuses existing entity nodes")
    webhook_url: Optional[str] = Field(default=None, description="URL to call when task is finished")
    creation_prompt: str = Field(default="generic", description="Must match the creation_prompt used in /create")
    graph_db_name: Optional[str] = Field(default=None, description="Explicit FalkorDB graph name (computed if not provided)")

    # Keep same type as parent but with default empty string (not needed for chunk addition)
    question: Optional[Union[str, List[MultimodalContent]]] = Field(default="", description="Not used for chunk addition")

    @model_validator(mode='after')
    def set_graph_db_name(self) -> 'AddDocumentRequest':
        if not self.graph_db_name:
            parts = [self.namespace]
            if self.creation_prompt:
                parts.append(self.creation_prompt)
            # Not including index_name in graph_db_name to maintain consistency with report namespace
            self.graph_db_name = "-".join(parts)
        return self



class AddDocumentResponse(BaseModel):
    """Response model for add chunk endpoint"""
    metadata_id: str
    chunks_processed: int
    entities_extracted: int
    entities_new: int
    entities_reused: int
    relationships_created: int
    status: str


class GraphNetworkResponse(BaseModel):
    """Response model for graph network endpoint"""
    nodes: List[Dict[str, Any]] = Field(description="List of nodes with id, label, properties")
    relationships: List[Dict[str, Any]] = Field(description="List of relationships with source_id, target_id, type")
    stats: Dict[str, Any] = Field(description="Network statistics and filter info")


class GraphOptimizeRequest(QuestionAnswer):
    """
    Request model for graph optimization via embedding-based entity deduplication.
    Finds near-duplicate entity nodes (cosine similarity >= threshold), merges them
    in a DuckDB merge plan, and reimports the cleaned graph into FalkorDB.
    Community reports are preserved as-is.
    """
    namespace: str = Field(..., description="Namespace of the graph to optimize")
    engine: Engine = Field(..., description="Engine configuration (for vector store references)")
    graph_db_name: Optional[str] = Field(default=None, description="FalkorDB graph name (computed if not provided)")
    creation_prompt: str = Field(default="generic", description="Must match the creation_prompt used in /create")
    similarity_threshold: float = Field(
        default=0.92, ge=0.5, le=1.0,
        description="Cosine similarity threshold above which two entities are considered duplicates. "
                    "Higher = more conservative (fewer merges). Recommended: 0.90–0.95."
    )
    embedding_batch_size: int = Field(
        default=256,
        description="Number of entities to embed per LLM call. Raise for TEI/vLLM, lower for cloud APIs."
    )
    dry_run: bool = Field(
        default=False,
        description="If True, compute and return the merge plan without modifying FalkorDB."
    )
    webhook_url: Optional[str] = Field(default=None, description="URL to call when task is finished")
    question: Optional[Union[str, List[MultimodalContent]]] = Field(default="", description="Not used")

    @model_validator(mode='after')
    def set_graph_db_name(self) -> 'GraphOptimizeRequest':
        if not self.graph_db_name:
            parts = [self.namespace]
            if self.creation_prompt:
                parts.append(self.creation_prompt)
            self.graph_db_name = "-".join(parts)
        return self


class GraphOptimizeResponse(BaseModel):
    """Response model for graph optimization endpoint"""
    status: str
    nodes_before: int
    nodes_after: int
    nodes_merged: int
    relationships_before: int
    relationships_after: int
    merge_pairs: int
    dry_run: bool
    snapshot_timestamp: Optional[str] = None


class GraphReimportRequest(QuestionAnswer):
    """
    Request model for reimporting a graph from a previously saved MinIO snapshot.
    Useful after external Parquet manipulation or as disaster recovery.
    """
    namespace: str = Field(..., description="Namespace of the graph to reimport")
    engine: Engine = Field(..., description="Engine configuration")
    graph_db_name: Optional[str] = Field(default=None, description="FalkorDB graph name (computed if not provided)")
    creation_prompt: str = Field(default="generic", description="Must match the original creation_prompt")
    snapshot_timestamp: Optional[str] = Field(
        default=None,
        description="Snapshot timestamp to reimport (e.g. '20260426_143000'). If None, uses the latest snapshot."
    )
    preserve_community_reports: bool = Field(
        default=True,
        description="If True, community reports saved in MinIO are re-imported into FalkorDB."
    )
    webhook_url: Optional[str] = Field(default=None, description="URL to call when task is finished")
    question: Optional[Union[str, List[MultimodalContent]]] = Field(default="", description="Not used")

    @model_validator(mode='after')
    def set_graph_db_name(self) -> 'GraphReimportRequest':
        if not self.graph_db_name:
            parts = [self.namespace]
            if self.creation_prompt:
                parts.append(self.creation_prompt)
            self.graph_db_name = "-".join(parts)
        return self


class GraphReimportResponse(BaseModel):
    """Response model for graph reimport endpoint"""
    status: str
    nodes_imported: int
    relationships_imported: int
    community_reports_restored: int
    snapshot_timestamp: str


class MultimodalSearchResponse(BaseModel):
    """Response model for Multimodal Search"""
    answer: str
    sources: Dict[str, Any]
    query_used: str


# Risolvi forward references per i modelli che ereditano da QuestionAnswer
def rebuild_graph_schemas():
    """Rebuild models to resolve forward references."""
    from tilellm.models.chat import ChatEntry
    from tilellm.models.llm import TEIConfig, PineconeRerankerConfig
    GraphQARequest.model_rebuild()
    GraphOptimizeRequest.model_rebuild()
    GraphReimportRequest.model_rebuild()
    GraphQAAdvancedRequest.model_rebuild()
    GraphCreateRequest.model_rebuild()
    GraphClusterRequest.model_rebuild()
    AddDocumentRequest.model_rebuild()

