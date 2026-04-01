"""
RAPTOR utility functions.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def format_chunks_for_clustering(chunks: List[Dict[str, Any]]) -> str:
    """
    Format chunks for clustering prompt.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Formatted string for clustering
    """
    formatted = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")[:500]  # Truncate for efficiency
        formatted.append(f"[Chunk {i+1}]: {content}")
    
    return "\n\n".join(formatted)


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text for clustering hints.
    
    Args:
        text: Text to analyze
        max_phrases: Maximum phrases to extract
        
    Returns:
        List of key phrases
    """
    # Simple implementation: extract noun phrases
    # Can be enhanced with NLP libraries
    sentences = text.split(".")
    phrases = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and len(phrases) < max_phrases:
            phrases.append(sentence)
    
    return phrases


def calculate_cluster_coherence(
    chunks: List[Dict[str, Any]],
    embeddings: Any,
) -> float:
    """
    Calculate coherence score for a cluster of chunks.
    
    Args:
        chunks: List of chunks in cluster
        embeddings: Embeddings model
        
    Returns:
        Coherence score (0-1)
    """
    try:
        if len(chunks) < 2:
            return 1.0
        
        # Generate embeddings
        texts = [chunk.get("content", "") for chunk in chunks]
        chunk_embeddings = embeddings.embed_documents(texts)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(chunk_embeddings)):
            for j in range(i + 1, len(chunk_embeddings)):
                sim = cosine_similarity(
                    chunk_embeddings[i],
                    chunk_embeddings[j]
                )
                similarities.append(sim)
        
        # Average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return avg_similarity
        
    except Exception as e:
        logger.error(f"Error calculating cluster coherence: {e}")
        return 0.0


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (0-1)
    """
    try:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


def truncate_text_for_prompt(
    text: str,
    max_tokens: int = 1000,
    overlap: int = 100,
) -> str:
    """
    Truncate text to fit within token limit while preserving context.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens (approximate)
        overlap: Overlap between chunks
        
    Returns:
        Truncated text
    """
    # Estimate: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return text
    
    # Take first and last portions with overlap
    portion_size = (max_chars - overlap) // 2
    
    first_portion = text[:portion_size]
    last_portion = text[-portion_size:]
    
    return f"{first_portion}\n\n...[truncated]...\n\n{last_portion}"


def format_tree_statistics(tree_data: Dict[str, Any]) -> str:
    """
    Format tree statistics for display/logging.
    
    Args:
        tree_data: Tree dictionary
        
    Returns:
        Formatted statistics string
    """
    levels = tree_data.get("levels", {})
    total_nodes = tree_data.get("total_nodes", 0)
    
    stats = ["RAPTOR Tree Statistics:", f"  Total nodes: {total_nodes}"]
    
    for level, nodes in sorted(levels.items()):
        stats.append(f"  Level {level}: {len(nodes)} nodes")
    
    return "\n".join(stats)


def validate_tree_structure(tree_data: Dict[str, Any]) -> bool:
    """
    Validate RAPTOR tree structure for consistency.
    
    Args:
        tree_data: Tree dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        nodes = tree_data.get("nodes", {})
        levels = tree_data.get("levels", {})
        leaf_ids = tree_data.get("leaf_ids", [])
        root_ids = tree_data.get("root_ids", [])
        
        # Check all leaf IDs exist
        for leaf_id in leaf_ids:
            if leaf_id not in nodes:
                logger.warning(f"Leaf node {leaf_id} not found in nodes")
                return False
        
        # Check all root IDs exist
        for root_id in root_ids:
            if root_id not in nodes:
                logger.warning(f"Root node {root_id} not found in nodes")
                return False
        
        # Check level consistency
        for level, node_ids in levels.items():
            for node_id in node_ids:
                if node_id not in nodes:
                    logger.warning(f"Node {node_id} at level {level} not found")
                    return False
                
                node = nodes[node_id]
                if node.get("level") != level:
                    logger.warning(f"Node {node_id} level mismatch")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating tree: {e}")
        return False


def get_timestamp() -> str:
    """
    Get current ISO timestamp.
    
    Returns:
        ISO format timestamp
    """
    return datetime.utcnow().isoformat()
