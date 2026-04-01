"""
LangGraph nodes for RAPTOR summarization workflow.

Implements the recursive summarization process using LangGraph StateGraph.
Uses closure pattern for dependency injection (repository, llm, embeddings).
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any

from tilellm.modules.raptor.agents.state import RaptorGraphState
from tilellm.modules.raptor.models.models import RaptorNode, RaptorLevel
from tilellm.modules.raptor.prompts import (
    RAPTOR_SUMMARY_PROMPT,
    RAPTOR_HIERARCHICAL_SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


def create_raptor_nodes(repo, llm, embeddings, vector_repo, engine=None):
    """
    Factory function to create RAPTOR workflow nodes with dependency injection.

    Args:
        repo: RaptorRepository instance
        llm: LLM instance for summarization
        embeddings: Embeddings model
        vector_repo: Vector store repository
        engine: Engine configuration object (required for indexing)

    Returns:
        Dictionary of node functions
    """
    
    async def initialize_tree_node(state: RaptorGraphState) -> Dict[str, Any]:
        """
        Initialize tree structure from input chunks.
        
        Creates Level 0 leaf nodes from raw chunks.
        """
        chunks = state.get("chunks", [])
        doc_id = state["doc_id"]
        namespace = state["namespace"]
        
        logger.info(f"Initializing RAPTOR tree with {len(chunks)} chunks")
        
        # Create leaf nodes
        leaf_nodes = []
        leaf_ids = []
        
        for idx, chunk_data in enumerate(chunks):
            node_id = f"chunk_{doc_id}_{idx}_{uuid.uuid4().hex[:8]}"
            
            node = RaptorNode(
                node_id=node_id,
                level=RaptorLevel.LEVEL_0,
                content=chunk_data.get("content", ""),
                metadata={
                    "doc_id": doc_id,
                    "namespace": namespace,
                    "chunk_index": idx,
                    "source": chunk_data.get("source", ""),
                    "page": chunk_data.get("page", 0),
                },
                created_at=datetime.utcnow().isoformat(),
            )
            
            leaf_nodes.append(node.model_dump(mode='json'))
            leaf_ids.append(node_id)
        
        # Initialize tree structure
        tree_levels = {0: leaf_ids}
        tree_nodes = {node["node_id"]: node for node in leaf_nodes}
        
        return {
            "nodes_at_level": leaf_nodes,
            "current_level": 0,
            "tree_nodes": tree_nodes,
            "tree_levels": tree_levels,
            "leaf_ids": leaf_ids,
            "should_continue": len(leaf_nodes) >= state.get("cluster_size", 5),
        }
    
    async def cluster_nodes_node(state: RaptorGraphState) -> Dict[str, Any]:
        """
        Cluster nodes at current level into groups for summarization.
        
        Uses embedding-based similarity grouping.
        """
        nodes = state.get("nodes_at_level", [])
        cluster_size = state.get("cluster_size", 5)
        embeddings = state.get("embeddings")
        
        if not nodes or not embeddings:
            return {"clusters": [], "should_continue": False}
        
        # If fewer nodes than cluster_size, put all in one cluster
        if len(nodes) <= cluster_size:
            return {"clusters": [nodes], "should_continue": True}
        
        try:
            # Generate embeddings for clustering
            texts = [node.get("content", "") for node in nodes]
            
            # Simple sequential clustering (can be enhanced with K-Means)
            clusters = []
            for i in range(0, len(nodes), cluster_size):
                cluster = nodes[i:i + cluster_size]
                clusters.append(cluster)
            
            logger.info(f"Created {len(clusters)} clusters from {len(nodes)} nodes")
            
            return {"clusters": clusters}
            
        except Exception as e:
            logger.error(f"Error clustering nodes: {e}")
            # Fallback: sequential grouping
            fallback_clusters = [
                nodes[i:i + cluster_size] for i in range(0, len(nodes), cluster_size)
            ]
            return {"clusters": fallback_clusters}
    
    async def generate_summaries_node(state: RaptorGraphState) -> Dict[str, Any]:
        """
        Generate summaries for each cluster using LLM.
        """
        clusters = state.get("clusters", [])
        current_level = state.get("current_level", 0)
        next_level = current_level + 1
        llm = state.get("llm")
        doc_id = state["doc_id"]
        namespace = state["namespace"]
        
        if not clusters or not llm:
            return {"summaries": [], "should_continue": False}
        
        summaries = []
        
        for cluster_idx, cluster in enumerate(clusters):
            # Prepare context
            context_parts = [node.get("content", "") for node in cluster]
            context = "\n\n---\n\n".join(context_parts)
            child_ids = [node.get("node_id") for node in cluster]
            
            # Choose prompt based on level
            if next_level == 1:
                prompt_text = RAPTOR_SUMMARY_PROMPT.format(context=context)
            else:
                prompt_text = RAPTOR_HIERARCHICAL_SUMMARY_PROMPT.format(context=context)
            
            try:
                # Generate summary
                response = await llm.ainvoke(prompt_text)
                summary_text = response.content.strip()
                
                # Create summary node
                node_id = f"summary_{doc_id}_L{next_level}_{uuid.uuid4().hex[:8]}"
                
                summary_node = RaptorNode(
                    node_id=node_id,
                    level=RaptorLevel(next_level),
                    content=summary_text,
                    summary=summary_text,
                    child_ids=child_ids,
                    parent_id=None,  # Will be set if this node gets a parent
                    metadata={
                        "doc_id": doc_id,
                        "namespace": namespace,
                        "num_children": len(child_ids),
                        "cluster_theme": f"Summary of {len(child_ids)} items",
                    },
                    created_at=datetime.utcnow().isoformat(),
                    model_used=getattr(llm, 'model_name', None),
                )
                
                summaries.append(summary_node.model_dump(mode='json'))
                
                logger.debug(
                    f"Generated L{next_level} summary from {len(cluster)} nodes"
                )
                
            except Exception as e:
                logger.error(f"Error generating summary for cluster {cluster_idx}: {e}")
                # Skip this cluster
        
        # Update parent references for child nodes
        tree_nodes = state.get("tree_nodes", {})
        for summary in summaries:
            for child_id in summary.get("child_ids", []):
                if child_id in tree_nodes:
                    tree_nodes[child_id]["parent_id"] = summary["node_id"]
        
        return {
            "summaries": summaries,
            "tree_nodes": tree_nodes,
        }
    
    async def index_summaries_node(state: RaptorGraphState) -> Dict[str, Any]:
        """
        Index summary embeddings in vector store.
        """
        summaries = state.get("summaries", [])
        namespace = state["namespace"]
        embeddings = state.get("embeddings")
        
        if not summaries or not embeddings:
            return {}
        
        indexed_count = 0
        
        for summary in summaries:
            try:
                await repo.index_summary_embedding(
                    node=RaptorNode(**summary),
                    namespace=namespace,
                    repo=vector_repo,
                    engine=engine,
                    llm_embeddings=embeddings,
                )
                indexed_count += 1
            except Exception as e:
                logger.error(f"Error indexing summary {summary.get('node_id')}: {e}")
        
        logger.info(f"Indexed {indexed_count} summary embeddings")
        
        return {}
    
    async def update_tree_node(state: RaptorGraphState) -> Dict[str, Any]:
        """
        Update tree structure with new summaries and prepare for next level.
        """
        summaries = state.get("summaries", [])
        current_level = state.get("current_level", 0)
        next_level = current_level + 1
        max_levels = state.get("max_levels", 3)
        cluster_size = state.get("cluster_size", 5)
        
        tree_nodes = state.get("tree_nodes", {})
        tree_levels = state.get("tree_levels", {})
        
        # Add summaries to tree
        summary_ids = []
        for summary in summaries:
            node_id = summary.get("node_id")
            tree_nodes[node_id] = summary
            summary_ids.append(node_id)
        
        tree_levels[next_level] = summary_ids
        
        # Determine if we should continue to next level
        should_continue = (
            next_level < max_levels and
            len(summaries) >= cluster_size
        )
        
        logger.info(
            f"Updated tree: Level {next_level} has {len(summaries)} nodes. "
            f"Continue: {should_continue}"
        )
        
        return {
            "tree_nodes": tree_nodes,
            "tree_levels": tree_levels,
            "nodes_at_level": summaries,
            "current_level": next_level,
            "should_continue": should_continue,
        }
    
    async def finalize_tree_node(state: RaptorGraphState) -> Dict[str, Any]:
        """
        Finalize tree structure and save to repository.
        """
        tree_nodes = state.get("tree_nodes", {})
        tree_levels = state.get("tree_levels", {})
        leaf_ids = state.get("leaf_ids", [])
        doc_id = state["doc_id"]
        namespace = state["namespace"]
        
        # Root nodes are at the highest level
        max_level = max(tree_levels.keys()) if tree_levels else 0
        root_ids = tree_levels.get(max_level, [])
        
        # Generate tree ID
        tree_id = f"raptor_{doc_id}_{int(time.time())}"
        
        # Create RaptorTree structure
        from tilellm.modules.raptor.models.models import RaptorTree, RaptorConfig
        
        # Convert nodes back to RaptorNode objects
        nodes_obj = {
            node_id: RaptorNode(**node_data)
            for node_id, node_data in tree_nodes.items()
        }
        
        tree = RaptorTree(
            tree_id=tree_id,
            namespace=namespace,
            doc_id=doc_id,
            nodes=nodes_obj,
            levels={k: v for k, v in tree_levels.items()},
            root_ids=root_ids,
            leaf_ids=leaf_ids,
            config=RaptorConfig(
                cluster_size=state.get("cluster_size", 5),
                max_levels=state.get("max_levels", 3),
            ),
            created_at=datetime.utcnow().isoformat(),
            total_nodes=len(tree_nodes),
        )
        
        # Save to repository
        await repo.save_tree(tree)
        
        logger.info(
            f"Finalized RAPTOR tree {tree_id}: {len(tree_nodes)} nodes, "
            f"{len(tree_levels)} levels"
        )
        
        return {
            "tree_id": tree_id,
            "root_ids": root_ids,
            "success": True,
            "processing_time": time.time() - state.get("start_time", time.time()),
        }
    
    # Return all nodes
    return {
        "initialize_tree": initialize_tree_node,
        "cluster_nodes": cluster_nodes_node,
        "generate_summaries": generate_summaries_node,
        "index_summaries": index_summaries_node,
        "update_tree": update_tree_node,
        "finalize_tree": finalize_tree_node,
    }
