"""
Data models for RAPTOR module.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, SecretStr

from tilellm.models.vector_store import Engine
from tilellm.models.embedding import LlmEmbeddingModel
from tilellm.models.llm import TEIConfig


class RaptorLevel(int, Enum):
    """RAPTOR tree levels."""
    LEVEL_0 = 0  # Raw chunks
    LEVEL_1 = 1  # First-level summaries (groups of 5-10 chunks)
    LEVEL_2 = 2  # Second-level summaries (groups of level-1 summaries)
    LEVEL_3 = 3  # Third-level summaries (for very long documents)


class RaptorRetrievalStrategy(str, Enum):
    """Retrieval strategies for RAPTOR."""
    COLLAPSED_TREE = "collapsed_tree"  # All nodes in same vector space (simpler)
    TREE_TRAVERSAL = "tree_traversal"  # Agent decides which level to search (dynamic)


class RaptorConfig(BaseModel):
    """Configuration for RAPTOR summarization."""
    
    # Clustering configuration
    cluster_size: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of chunks to group together for summarization (5-10)"
    )
    
    # Level configuration
    max_levels: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of summary levels to create"
    )
    
    # Document activation thresholds
    min_pages_for_raptor: int = Field(
        default=10,
        ge=1,
        description="Minimum document pages to activate RAPTOR"
    )
    
    # Document types that benefit from RAPTOR
    enabled_doc_types: List[str] = Field(
        default=["accademico", "tecnico", "legale", "scientifico"],
        description="Document types that should use RAPTOR summarization"
    )
    
    # Summary generation
    summary_max_tokens: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Maximum tokens for each summary"
    )
    
    summary_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for summary generation LLM"
    )
    
    # Retrieval configuration
    retrieval_strategy: RaptorRetrievalStrategy = Field(
        default=RaptorRetrievalStrategy.COLLAPSED_TREE,
        description="Strategy for retrieving from RAPTOR tree"
    )
    
    top_k_per_level: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Top-K results to retrieve per level"
    )
    
    # Fusion configuration (for collapsed tree)
    use_rrf_fusion: bool = Field(
        default=True,
        description="Use Reciprocal Rank Fusion for merging results from multiple levels"
    )
    
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=100,
        description="RRF constant k for rank fusion"
    )


class RaptorNode(BaseModel):
    """A node in the RAPTOR tree."""
    
    node_id: str = Field(description="Unique identifier for the node")
    level: RaptorLevel = Field(description="Level in the RAPTOR tree")
    
    # Content
    content: str = Field(description="Text content (chunk or summary)")
    summary: Optional[str] = Field(default=None, description="Summary text (for non-leaf nodes)")
    
    # References
    child_ids: List[str] = Field(
        default_factory=list,
        description="IDs of child nodes (chunks or lower-level summaries)"
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="ID of parent summary node (None for root/level-0 nodes)"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (source doc, position, etc.)"
    )
    
    # Embeddings (stored separately in vector store, but referenced here)
    embedding_ref: Optional[str] = Field(
        default=None,
        description="Reference to dense embedding in vector store"
    )
    sparse_embedding_ref: Optional[str] = Field(
        default=None,
        description="Reference to sparse embedding in vector store"
    )
    
    # Creation info
    created_at: Optional[str] = Field(default=None, description="ISO timestamp of creation")
    model_used: Optional[str] = Field(default=None, description="LLM model used for summarization")


class RaptorTree(BaseModel):
    """Complete RAPTOR tree structure for a document."""
    
    tree_id: str = Field(description="Unique identifier for the tree")
    namespace: str = Field(description="Namespace/document collection")
    doc_id: str = Field(description="Source document ID")
    
    # Tree structure
    nodes: Dict[str, RaptorNode] = Field(
        default_factory=dict,
        description="All nodes in the tree indexed by node_id"
    )
    
    # Level organization
    levels: Dict[int, List[str]] = Field(
        default_factory=dict,
        description="Node IDs organized by level"
    )
    
    # Root nodes (highest level summaries)
    root_ids: List[str] = Field(
        default_factory=list,
        description="IDs of root nodes (highest level summaries)"
    )
    
    # Leaf nodes (original chunks)
    leaf_ids: List[str] = Field(
        default_factory=list,
        description="IDs of leaf nodes (original chunks)"
    )
    
    # Metadata
    config: RaptorConfig = Field(
        default_factory=RaptorConfig,
        description="Configuration used to build this tree"
    )
    
    created_at: Optional[str] = Field(default=None, description="ISO timestamp of creation")
    total_nodes: int = Field(default=0, description="Total number of nodes in tree")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "tree_id": "raptor_tree_doc123",
                "namespace": "my-documents",
                "doc_id": "doc123",
                "nodes": {},
                "levels": {"0": ["chunk1", "chunk2"], "1": ["summary1"]},
                "root_ids": ["summary1"],
                "leaf_ids": ["chunk1", "chunk2"],
                "config": {"cluster_size": 5, "max_levels": 3},
                "total_nodes": 3
            }
        }
    }


class RaptorRequest(BaseModel):
    """Request to build RAPTOR tree for a document."""

    namespace: str = Field(description="Namespace containing the document")
    doc_id: str = Field(description="Document ID (metadata.id)")

    # Optional: explicit chunk IDs to use (if not provided, all chunks for doc_id are used)
    chunk_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific chunk IDs to include. If None, all chunks for doc_id are used."
    )

    # Configuration overrides
    config: Optional[RaptorConfig] = Field(
        default=None,
        description="Optional configuration overrides"
    )

    # Vector store engine (required for @inject_repo_async)
    engine: Engine

    # LLM credentials (required for @inject_llm_chat_async)
    gptkey: Optional[SecretStr] = None
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini", description="LLM model name")
    llm: Optional[str] = Field(default="openai", description="LLM provider")
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    temperature: float = Field(default=0.0)
    top_p: Optional[float] = Field(default=1.0)
    max_tokens: int = Field(default=512)
    debug: bool = Field(default=False)

    # Sparse encoder for hybrid search
    sparse_encoder: Union[str, TEIConfig, None] = Field(
        default=None,
        description="Sparse encoder for hybrid search (e.g., 'splade', 'bm25')"
    )

    # Optional: document type hint
    doc_type: Optional[str] = Field(
        default=None,
        description="Document type (accademico, tecnico, etc.)"
    )

    # Optional: page count hint
    page_count: Optional[int] = Field(default=None, description="Number of pages in document")


class RaptorResponse(BaseModel):
    """Response from RAPTOR tree building."""
    
    success: bool = Field(description="Whether the operation was successful")
    tree_id: Optional[str] = Field(default=None, description="ID of the created tree")
    
    # Tree statistics
    total_chunks: int = Field(default=0, description="Number of leaf chunks processed")
    total_summaries: int = Field(default=0, description="Number of summary nodes created")
    levels_created: int = Field(default=0, description="Number of levels in the tree")
    
    # Level breakdown
    level_stats: Dict[int, int] = Field(
        default_factory=dict,
        description="Number of nodes per level"
    )
    
    # Processing info
    processing_time_seconds: float = Field(default=0.0, description="Time taken to build tree")
    model_used: Optional[str] = Field(default=None, description="LLM model used")
    
    # Error info
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Tree reference (optional, can be retrieved separately)
    tree: Optional[RaptorTree] = Field(default=None, description="Full tree structure (optional)")


class RaptorSummaryRequest(BaseModel):
    """Request to generate summaries for chunks."""

    namespace: str = Field(description="Namespace containing chunks")
    doc_id: Optional[str] = Field(default=None, description="Document ID (optional, can be extracted from chunk_ids)")
    chunk_ids: List[str] = Field(description="Chunk IDs to summarize")

    # Grouping configuration
    cluster_size: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of chunks to group for each summary"
    )

    # Vector store engine (required for @inject_repo_async)
    engine: Engine

    # LLM credentials (required for @inject_llm_chat_async)
    gptkey: Optional[SecretStr] = None
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini", description="LLM model name")
    llm: Optional[str] = Field(default="openai", description="LLM provider")
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0)
    max_tokens: int = Field(default=512, ge=128, le=2048)
    debug: bool = Field(default=False)
    sparse_encoder: Union[str, TEIConfig, None] = Field(default=None)


class RaptorSummaryResponse(BaseModel):
    """Response from summary generation."""
    
    success: bool = Field(description="Whether the operation was successful")
    
    # Generated summaries
    summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of generated summaries with metadata"
    )
    
    # Statistics
    total_groups: int = Field(default=0, description="Number of chunk groups summarized")
    processing_time_seconds: float = Field(default=0.0, description="Time taken")
    
    # Error info
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RaptorRetrievalRequest(BaseModel):
    """Request for RAPTOR-based retrieval."""

    question: str = Field(description="User question/query")
    namespace: str = Field(description="Namespace to search")

    # Optional: specific document
    doc_id: Optional[str] = Field(
        default=None,
        description="Specific document ID to search within"
    )

    # Retrieval strategy
    strategy: RaptorRetrievalStrategy = Field(
        default=RaptorRetrievalStrategy.COLLAPSED_TREE,
        description="Retrieval strategy to use"
    )

    # Search configuration
    top_k: int = Field(default=5, ge=1, le=20, description="Total results to return")
    top_k_per_level: int = Field(default=3, ge=1, le=10)

    # Vector store engine (required for @inject_repo_async)
    engine: Engine

    # LLM credentials (required for @inject_llm_chat_async)
    gptkey: Optional[SecretStr] = None
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    llm: Optional[str] = Field(default="openai")
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    temperature: float = Field(default=0.0)
    top_p: Optional[float] = Field(default=1.0)
    max_tokens: int = Field(default=512)
    debug: bool = Field(default=False)
    sparse_encoder: Union[str, TEIConfig, None] = Field(default=None, description="Sparse encoder for hybrid search")

    # Use hybrid search (dense + sparse)
    use_hybrid: bool = Field(
        default=False,
        description="Use hybrid search (dense + sparse vectors)"
    )
    alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Hybrid weight: 1.0 = dense only, 0.0 = sparse only"
    )


class RaptorRetrievalResult(BaseModel):
    """Result from RAPTOR retrieval."""
    
    success: bool = Field(description="Whether retrieval was successful")
    
    # Retrieved items
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved nodes with scores and metadata"
    )
    
    # Strategy info
    strategy_used: str = Field(description="Strategy that was actually used")
    levels_searched: List[int] = Field(
        default_factory=list,
        description="Which levels were searched"
    )
    
    # For tree traversal: agent decisions
    traversal_path: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Path taken through tree (for tree_traversal strategy)"
    )
    
    # Statistics
    total_results: int = Field(default=0, description="Total results found")
    processing_time_seconds: float = Field(default=0.0)

    # Error info
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RaptorTraversalDecision(BaseModel):
    """Decision made by tree traversal agent."""

    current_level: int = Field(description="Current level in tree")
    action: Literal["search_this_level", "go_deeper", "go_higher", "stop"] = Field(
        description="Action to take"
    )
    reasoning: str = Field(description="Reasoning for the decision")
    next_level: Optional[int] = Field(default=None, description="Next level to search (if applicable)")


class RaptorQARequest(BaseModel):
    """Request for RAPTOR-based Q&A (retrieve + answer generation)."""

    question: str = Field(description="User question/query")
    namespace: str = Field(description="Namespace to search")

    # Optional: specific document
    doc_id: Optional[str] = Field(
        default=None,
        description="Specific document ID to search within"
    )

    # Retrieval strategy
    strategy: RaptorRetrievalStrategy = Field(
        default=RaptorRetrievalStrategy.COLLAPSED_TREE,
        description="Retrieval strategy to use"
    )

    # Search configuration
    top_k: int = Field(default=5, ge=1, le=20, description="Results to retrieve for context")
    top_k_per_level: int = Field(default=3, ge=1, le=10)

    # Vector store engine (required for @inject_repo_async)
    engine: Engine

    # LLM credentials (required for @inject_llm_chat_async)
    gptkey: Optional[SecretStr] = None
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini", description="LLM model for QA")
    llm: Optional[str] = Field(default="openai", description="LLM provider")
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    temperature: float = Field(default=0.0, description="Temperature for answer generation")
    top_p: Optional[float] = Field(default=1.0)
    max_tokens: int = Field(default=512, description="Max tokens for answer")
    debug: bool = Field(default=False)
    sparse_encoder: Union[str, TEIConfig, None] = Field(default=None)

    # Hybrid search
    use_hybrid: bool = Field(default=False, description="Use hybrid search (dense + sparse)")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Hybrid weight")


class RaptorQAResponse(BaseModel):
    """Response from RAPTOR Q&A."""

    success: bool = Field(description="Whether Q&A was successful")

    # Answer
    answer: str = Field(description="Generated answer to the question")

    # Retrieved context
    retrieved_chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chunks used to generate the answer"
    )

    # Retrieval details
    strategy_used: str = Field(description="Retrieval strategy that was used")
    levels_searched: List[int] = Field(
        default_factory=list,
        description="Which RAPTOR levels were searched"
    )

    # Statistics
    total_chunks_retrieved: int = Field(default=0, description="Total chunks retrieved")
    processing_time_seconds: float = Field(default=0.0, description="Total processing time")
    retrieval_time_seconds: Optional[float] = Field(default=None, description="Time for retrieval")
    answer_time_seconds: Optional[float] = Field(default=None, description="Time for answer generation")

    # Optional tree traversal info
    traversal_path: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Path taken through tree (for tree_traversal strategy)"
    )

    # Error info
    error: Optional[str] = Field(default=None, description="Error message if failed")