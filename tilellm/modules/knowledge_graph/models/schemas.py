"""
API request and response schemas for Knowledge Graph endpoints.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from tilellm.models import Engine, QuestionToLLM, QuestionAnswer
from tilellm.models.schemas.multimodal_content import MultimodalContent


class GraphQARequest(QuestionAnswer):
    """
    Request model for Graph QA endpoint.
    Extends QuestionToLLM to include LLM configuration for GraphRAG QA.
    """
    namespace: str
    max_results: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.3
    index_name: Optional[str] = None  # Optional index_name for graph partition
    
    # Keep same type as parent but with default (though parent requires it)
    question: Union[str, List[MultimodalContent]] = Field(default="", description="Question to answer using knowledge graph")
    
    # Note: llm_key, llm, model, etc. are inherited from QuestionToLLM


class GraphQAResponse(BaseModel):
    """Response model for Graph QA endpoint"""
    answer: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    query_used: str


class GraphCreateRequest(QuestionToLLM):
    """
    Request model for Graph creation endpoint.
    Extends QuestionToLLM to include LLM configuration for GraphRAG extraction.
    """
    namespace: str
    index_name: Optional[str] = None  # Optional index_name for graph partition
    engine: Optional[Engine] = None  # Engine configuration for vector store
    limit: Optional[int] = 100
    overwrite: Optional[bool] = False
    
    # Keep same type as parent but with default empty string (not needed for creation)
    question: Union[str, List[MultimodalContent]] = Field(default="", description="Optional prompt for extraction guidance")
    
    # Note: llm_key, llm, model, etc. are inherited from QuestionToLLM


class GraphCreateResponse(BaseModel):
    """Response model for Graph creation endpoint"""
    namespace: str
    chunks_processed: int
    nodes_created: int
    relationships_created: int
    status: str


class GraphClusterRequest(QuestionToLLM):
    """
    Request model for Graph clustering endpoint.
    Uses QuestionToLLM for LLM configuration to generate community reports.
    """
    level: Optional[int] = 0
    namespace: Optional[str] = None
    index_name: Optional[str] = None


class GraphClusterResponse(BaseModel):
    """Response model for Graph clustering endpoint"""
    status: str
    communities_detected: int
    reports_created: int
    message: Optional[str] = None


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
    engine: Optional[Engine] = None  # Engine configuration for vector store access


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