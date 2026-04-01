"""
State definition for RAPTOR LangGraph workflow.
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class RaptorGraphState(TypedDict):
    """
    State for RAPTOR workflow.
    
    Tracks the recursive summarization process through LangGraph nodes.
    """
    
    # Input
    chunks: Optional[List[Dict[str, Any]]]  # Leaf chunks to process
    namespace: str  # Vector store namespace
    doc_id: str  # Document identifier
    
    # Configuration
    cluster_size: int  # Chunks per summary (default: 5)
    max_levels: int  # Maximum tree depth (default: 3)
    
    # Current processing state
    current_level: int  # Current level being processed
    nodes_at_level: Optional[List[Dict[str, Any]]]  # Nodes at current level
    
    # Summarization
    clusters: Optional[List[List[Dict[str, Any]]]]  # Grouped nodes
    summaries: Optional[List[Dict[str, Any]]]  # Generated summaries
    
    # Tree structure
    tree_nodes: Optional[Dict[str, Dict[str, Any]]]  # All nodes indexed by ID
    tree_levels: Optional[Dict[int, List[str]]]  # Node IDs by level
    root_ids: Optional[List[str]]  # Highest level node IDs
    leaf_ids: Optional[List[str]]  # Leaf node IDs
    
    # LLM and embeddings (injected via closure)
    llm: Optional[Any]
    embeddings: Optional[Any]
    
    # Control flow
    should_continue: bool  # Continue to next level?
    retry_count: int  # Retry counter for failed operations
    max_retries: int  # Maximum retries allowed
    error_message: Optional[str]  # Error message if failed
    
    # Output
    tree_id: Optional[str]  # Generated tree ID
    success: bool  # Overall success flag
    processing_time: float  # Total processing time
    
    # Metadata
    metadata: Optional[Dict[str, Any]]  # Additional metadata


class RaptorTraversalState(TypedDict):
    """
    State for RAPTOR tree traversal (retrieval phase).
    
    Used by the tree traversal agent to decide which levels to search.
    """
    
    # Input
    question: str  # User question
    namespace: str  # Namespace to search
    doc_id: Optional[str]  # Optional document filter
    
    # Tree structure
    tree: Optional[Dict[str, Any]]  # RAPTOR tree structure
    available_levels: Optional[List[int]]  # Available levels in tree
    
    # Current traversal state
    current_level: int  # Current level being considered
    search_history: Optional[List[Dict[str, Any]]]  # Levels already searched
    retrieved_results: Optional[List[Dict[str, Any]]]  # Results so far
    
    # Agent decision
    action: Optional[Literal[
        "search_this_level",
        "go_deeper",
        "go_higher",
        "stop"
    ]]  # Current action
    reasoning: Optional[str]  # Agent's reasoning
    
    # Configuration
    top_k: int  # Total results target
    top_k_per_level: int  # Results per level
    max_iterations: int  # Maximum traversal steps
    
    # Control
    iteration_count: int  # Current iteration
    should_continue: bool  # Continue traversal?
    
    # Output
    final_results: Optional[List[Dict[str, Any]]]  # Final retrieved results
    traversal_path: Optional[List[Dict[str, Any]]]  # Path taken through tree
    success: bool  # Success flag
