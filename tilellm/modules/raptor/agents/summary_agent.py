"""
RAPTOR Summary Agent - LangGraph workflow for hierarchical summarization.

Creates and manages the RAPTOR tree building process using LangGraph StateGraph.
"""

import logging
from typing import Any, Optional, Dict

from langgraph.graph import StateGraph, END

from tilellm.modules.raptor.agents.state import RaptorGraphState
from tilellm.modules.raptor.agents.nodes import create_raptor_nodes
from tilellm.modules.raptor.repository import RaptorRepository

logger = logging.getLogger(__name__)


def create_summary_agent_workflow(
    repo: RaptorRepository,
    llm: Any,
    embeddings: Any,
    vector_repo: Any,
) -> StateGraph:
    """
    Create LangGraph workflow for RAPTOR summarization.
    
    Args:
        repo: RaptorRepository instance
        llm: LLM instance for summarization
        embeddings: Embeddings model
        vector_repo: Vector store repository
        
    Returns:
        Compiled LangGraph StateGraph
    """
    
    # Create nodes with dependency injection
    nodes = create_raptor_nodes(repo, llm, embeddings, vector_repo)
    
    # Build workflow
    workflow = StateGraph(RaptorGraphState)
    
    # Add nodes
    workflow.add_node("initialize_tree", nodes["initialize_tree"])
    workflow.add_node("cluster_nodes", nodes["cluster_nodes"])
    workflow.add_node("generate_summaries", nodes["generate_summaries"])
    workflow.add_node("index_summaries", nodes["index_summaries"])
    workflow.add_node("update_tree", nodes["update_tree"])
    workflow.add_node("finalize_tree", nodes["finalize_tree"])
    
    # Set entry point
    workflow.set_entry_point("initialize_tree")
    
    # Define edges
    workflow.add_edge("initialize_tree", "cluster_nodes")
    workflow.add_edge("cluster_nodes", "generate_summaries")
    workflow.add_edge("generate_summaries", "index_summaries")
    workflow.add_edge("index_summaries", "update_tree")
    
    # Conditional edge: continue or finalize
    def should_continue(state: RaptorGraphState) -> str:
        """Decide whether to continue to next level or finalize."""
        if state.get("should_continue", False):
            return "cluster_nodes"
        else:
            return "finalize_tree"
    
    workflow.add_conditional_edges(
        "update_tree",
        should_continue,
        {
            "cluster_nodes": "cluster_nodes",
            "finalize_tree": "finalize_tree",
        }
    )
    
    # End after finalize
    workflow.add_edge("finalize_tree", END)
    
    # Compile workflow
    app = workflow.compile()
    
    logger.info("RAPTOR summary agent workflow created")
    return app


async def run_raptor_summarization(
    chunks: list,
    namespace: str,
    doc_id: str,
    llm: Any,
    embeddings: Any,
    vector_repo: Any,
    repo: Optional[RaptorRepository] = None,
    cluster_size: int = 5,
    max_levels: int = 3,
) -> Dict[str, Any]:
    """
    Run RAPTOR summarization workflow.
    
    Args:
        chunks: List of chunk documents/data to summarize
        namespace: Vector store namespace
        doc_id: Document identifier
        llm: LLM instance
        embeddings: Embeddings model
        vector_repo: Vector store repository
        repo: Optional RaptorRepository (creates default if None)
        cluster_size: Chunks per summary group
        max_levels: Maximum tree levels
        
    Returns:
        Dictionary with tree_id and statistics
    """
    if repo is None:
        repo = RaptorRepository()
    
    # Create workflow
    workflow = create_summary_agent_workflow(repo, llm, embeddings, vector_repo)
    
    # Prepare initial state
    initial_state: RaptorGraphState = {
        "chunks": chunks,
        "namespace": namespace,
        "doc_id": doc_id,
        "cluster_size": cluster_size,
        "max_levels": max_levels,
        "current_level": 0,
        "nodes_at_level": None,
        "clusters": None,
        "summaries": None,
        "tree_nodes": None,
        "tree_levels": None,
        "root_ids": None,
        "leaf_ids": None,
        "llm": llm,
        "embeddings": embeddings,
        "should_continue": True,
        "retry_count": 0,
        "max_retries": 3,
        "error_message": None,
        "tree_id": None,
        "success": False,
        "processing_time": 0.0,
        "metadata": {},
    }
    
    # Run workflow
    try:
        result = await workflow.ainvoke(initial_state)
        
        logger.info(
            f"RAPTOR summarization complete: tree_id={result.get('tree_id')}, "
            f"success={result.get('success')}"
        )
        
        return {
            "success": result.get("success", False),
            "tree_id": result.get("tree_id"),
            "processing_time": result.get("processing_time", 0.0),
            "levels_created": len(result.get("tree_levels", {})),
            "total_nodes": len(result.get("tree_nodes", {})),
        }
        
    except Exception as e:
        logger.error(f"RAPTOR workflow failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }
