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
            if self.index_name:
                parts.append(self.index_name)
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

    @model_validator(mode='after')
    def set_graph_db_name(self) -> 'GraphCreateRequest':
        if not self.graph_db_name:
            parts = [self.namespace]
            if self.creation_prompt:
                parts.append(self.creation_prompt)
            if self.index_name:
                parts.append(self.index_name)
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
    graph_db_name: Optional[str] = Field(default=None, description="Graph name")
    engine: Optional[Engine] = None  # Added for report indexing
    overwrite: Optional[bool] = Field(
        default=True,
        description="If True, removes existing community reports before regeneration (default: True)"
    )
    webhook_url: Optional[str] = Field(default=None, description="URL to call when task is finished")

    @model_validator(mode='after')
    def set_graph_db_name_cluster(self) -> 'GraphClusterRequest':
        if not self.graph_db_name:
            parts = [self.namespace] if self.namespace else []
            if self.index_name:
                parts.append(self.index_name)
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
            index_name = self.engine.index_name if self.engine and self.engine.index_name else None
            if index_name:
                parts.append(index_name)
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
    GraphQAAdvancedRequest.model_rebuild()
    GraphCreateRequest.model_rebuild()
    GraphClusterRequest.model_rebuild()
    AddDocumentRequest.model_rebuild()

