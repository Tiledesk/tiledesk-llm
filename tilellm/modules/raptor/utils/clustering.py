"""
RAPTOR Clustering Algorithm using UMAP + GMM.

Based on the official RAPTOR implementation:
https://github.com/parthsarthi03/raptor

This module provides hierarchical clustering using:
- UMAP for dimensionality reduction
- GMM (Gaussian Mixture Models) for clustering
- BIC for optimal cluster selection

Note: This module requires optional dependencies. Install with:
    poetry install --extras "raptor"
or
    pip install umap-learn scikit-learn
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import numpy as np
    import tiktoken
    import umap
    from sklearn.mixture import GaussianMixture
    RAPTOR_CLUSTERING_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"RAPTOR clustering dependencies not available: {e}. "
        f"Install with: poetry install --extras 'raptor'"
    )
    RAPTOR_CLUSTERING_AVAILABLE = False
    # Create stubs for graceful degradation
    np = None
    tiktoken = None
    umap = None
    GaussianMixture = None

from tilellm.modules.raptor.models.models import RaptorNode

# Set random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Reduce embeddings dimensionality for global clustering using UMAP.
    
    Args:
        embeddings: Input embeddings array
        dim: Target dimensionality
        n_neighbors: Number of neighbors for UMAP (default: sqrt(n-1))
        metric: Distance metric for UMAP
        
    Returns:
        Reduced embeddings
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    
    # Ensure n_neighbors doesn't exceed number of samples
    n_neighbors = min(n_neighbors, len(embeddings) - 1)
    
    # Ensure dim doesn't exceed available dimensions
    dim = min(dim, len(embeddings) - 2, embeddings.shape[1] - 1)
    
    if dim < 1:
        dim = 1
    
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        random_state=RANDOM_SEED
    ).fit_transform(embeddings)
    
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    num_neighbors: int = 10,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Reduce embeddings dimensionality for local clustering using UMAP.
    
    Args:
        embeddings: Input embeddings array
        dim: Target dimensionality
        num_neighbors: Number of neighbors for UMAP
        metric: Distance metric for UMAP
        
    Returns:
        Reduced embeddings
    """
    # Ensure constraints are met
    dim = min(dim, len(embeddings) - 2, embeddings.shape[1] - 1)
    num_neighbors = min(num_neighbors, len(embeddings) - 1)
    
    if dim < 1:
        dim = 1
    
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors,
        n_components=dim,
        metric=metric,
        random_state=RANDOM_SEED
    ).fit_transform(embeddings)
    
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = RANDOM_SEED
) -> int:
    """
    Determine optimal number of clusters using BIC (Bayesian Information Criterion).
    
    Args:
        embeddings: Input embeddings
        max_clusters: Maximum number of clusters to try
        random_state: Random seed
        
    Returns:
        Optimal number of clusters
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters + 1)
    bics = []
    
    for n in n_clusters:
        try:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        except Exception as e:
            logger.warning(f"GMM failed for n_clusters={n}: {e}")
            bics.append(float('inf'))
    
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(
    embeddings: np.ndarray,
    threshold: float,
    random_state: int = RANDOM_SEED
) -> Tuple[List[np.ndarray], int]:
    """
    Cluster embeddings using Gaussian Mixture Model.
    
    Args:
        embeddings: Input embeddings
        threshold: Probability threshold for cluster assignment
        random_state: Random seed
        
    Returns:
        Tuple of (cluster labels for each embedding, number of clusters)
    """
    n_clusters = get_optimal_clusters(embeddings)
    
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    
    # Each embedding can belong to multiple clusters if prob > threshold
    labels = [np.where(prob > threshold)[0] for prob in probs]
    
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int = 10,
    threshold: float = 0.1,
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Perform hierarchical clustering using UMAP + GMM.
    
    This implements a two-level clustering:
    1. Global clustering on all embeddings
    2. Local clustering within each global cluster
    
    Args:
        embeddings: Input embeddings array
        dim: Dimensionality for UMAP reduction
        threshold: Probability threshold for cluster assignment
        verbose: Enable verbose logging
        
    Returns:
        List of cluster labels for each embedding (can belong to multiple clusters)
        
    Raises:
        ImportError: If required dependencies are not installed
    """
    if not RAPTOR_CLUSTERING_AVAILABLE:
        raise ImportError(
            "RAPTOR clustering requires optional dependencies. "
            "Install with: poetry install --extras 'raptor'"
        )
    
    if len(embeddings) == 0:
        return []
    
    if len(embeddings) == 1:
        return [np.array([0])]
    
    # Global clustering
    reduced_embeddings_global = global_cluster_embeddings(
        embeddings,
        min(dim, len(embeddings) - 2)
    )
    
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )
    
    if verbose:
        logger.info(f"Global Clusters: {n_global_clusters}")
    
    # Local clustering within each global cluster
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0
    
    for i in range(n_global_clusters):
        # Get embeddings in this global cluster
        global_cluster_mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embeddings_ = embeddings[global_cluster_mask]
        
        if verbose:
            logger.info(f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}")
        
        if len(global_cluster_embeddings_) == 0:
            continue
        
        # If too few points, don't recluster
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local UMAP + GMM
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )
        
        if verbose:
            logger.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")
        
        # Map local clusters to global indices
        for j in range(n_local_clusters):
            local_cluster_mask = np.array([j in lc for lc in local_clusters])
            local_cluster_embeddings_ = global_cluster_embeddings_[local_cluster_mask]
            
            # Find indices in original embeddings
            indices = find_matching_indices(embeddings, local_cluster_embeddings_)
            
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )
        
        total_clusters += n_local_clusters
    
    if verbose:
        logger.info(f"Total Clusters: {total_clusters}")
    
    return all_local_clusters


def find_matching_indices(
    embeddings: np.ndarray,
    subset: np.ndarray
) -> np.ndarray:
    """
    Find indices where subset embeddings match the full embeddings array.

    Args:
        embeddings: Full embeddings array
        subset: Subset of embeddings to find

    Returns:
        Array of indices where matches were found
    """
    indices = []
    for sub_emb in subset:
        # Use allclose with tolerance instead of exact equality for floating point comparison
        # This avoids losing nodes due to numerical precision differences in embeddings
        matches = np.where(
            np.array([np.allclose(e, sub_emb, atol=1e-6) for e in embeddings])
        )[0]
        indices.extend(matches.tolist())
    return np.array(indices)


class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms."""
    
    @abstractmethod
    def perform_clustering(
        self,
        embeddings: np.ndarray,
        **kwargs
    ) -> List[List[int]]:
        """
        Perform clustering on embeddings.
        
        Args:
            embeddings: Input embeddings array
            **kwargs: Additional clustering parameters
            
        Returns:
            List of cluster assignments (each embedding can be in multiple clusters)
        """
        pass


class RAPTORClustering(ClusteringAlgorithm):
    """
    RAPTOR clustering algorithm using UMAP + GMM.
    
    Implements hierarchical clustering with:
    - UMAP for dimensionality reduction (global and local)
    - GMM for clustering with automatic cluster selection via BIC
    - Recursive reclustering if cluster exceeds token limit
    
    Note: Requires optional dependencies (umap-learn, scikit-learn).
    Falls back to SequentialClustering if dependencies are missing.
    """
    
    def perform_clustering(
        self,
        nodes: List[RaptorNode],
        embedding_model_name: str = "default",
        max_length_in_cluster: int = 3500,
        tokenizer=None,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[RaptorNode]]:
        """
        Perform RAPTOR clustering on nodes.
        
        Args:
            nodes: List of RaptorNode objects with embeddings
            embedding_model_name: Key to access embeddings in node
            max_length_in_cluster: Maximum tokens per cluster
            tokenizer: Tiktoken tokenizer for counting tokens
            reduction_dimension: UMAP reduction dimension
            threshold: GMM probability threshold
            verbose: Enable verbose logging
            
        Returns:
            List of clusters, where each cluster is a list of RaptorNode
            
        Raises:
            ImportError: If required dependencies are not installed
        """
        if not RAPTOR_CLUSTERING_AVAILABLE:
            logger.warning(
                "RAPTOR clustering not available, falling back to sequential clustering. "
                "Install with: poetry install --extras 'raptor'"
            )
            # Fallback to sequential clustering
            sequential = SequentialClustering()
            return sequential.perform_clustering(nodes, cluster_size=5)
        
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Extract embeddings from nodes
        embeddings = []
        for node in nodes:
            emb = node.metadata.get(f"embedding_{embedding_model_name}")
            if emb is None:
                # Fallback: try to get embedding from metadata
                emb = node.metadata.get("embedding")
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            logger.warning("No embeddings found in nodes, returning each node as separate cluster")
            return [[node] for node in nodes]
        
        embeddings = np.array(embeddings)
        
        # Perform hierarchical clustering
        clusters = perform_clustering(
            embeddings,
            dim=reduction_dimension,
            threshold=threshold,
            verbose=verbose
        )
        
        # Convert cluster labels to node clusters
        node_clusters = []
        
        # Get all unique cluster labels
        all_labels = np.concatenate(clusters) if clusters else np.array([])
        unique_labels = np.unique(all_labels)
        
        for label in unique_labels:
            # Get indices of nodes in this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]
            
            # Base case: single node cluster
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue
            
            # Check if cluster exceeds token limit
            total_length = sum(
                [len(tokenizer.encode(node.content)) for node in cluster_nodes]
            )
            
            if total_length > max_length_in_cluster:
                # Recursively recluster this cluster
                if verbose:
                    logger.info(
                        f"Reclustering cluster with {len(cluster_nodes)} nodes "
                        f"({total_length} tokens > {max_length_in_cluster})"
                    )
                
                sub_clusters = self.perform_clustering(
                    cluster_nodes,
                    embedding_model_name=embedding_model_name,
                    max_length_in_cluster=max_length_in_cluster,
                    tokenizer=tokenizer,
                    reduction_dimension=reduction_dimension,
                    threshold=threshold,
                    verbose=verbose,
                )
                node_clusters.extend(sub_clusters)
            else:
                node_clusters.append(cluster_nodes)
        
        if verbose:
            logger.info(f"Created {len(node_clusters)} final clusters")
        
        return node_clusters


class SequentialClustering(ClusteringAlgorithm):
    """
    Simple sequential clustering (baseline).
    
    Groups nodes sequentially without considering semantic similarity.
    Useful as a baseline or for very large datasets where speed is critical.
    """
    
    def perform_clustering(
        self,
        nodes: List[RaptorNode],
        cluster_size: int = 5,
        **kwargs
    ) -> List[List[RaptorNode]]:
        """
        Perform sequential clustering.
        
        Args:
            nodes: List of RaptorNode objects
            cluster_size: Target cluster size
            **kwargs: Unused
            
        Returns:
            List of clusters
        """
        clusters = []
        
        for i in range(0, len(nodes), cluster_size):
            cluster = nodes[i:i + cluster_size]
            if cluster:
                clusters.append(cluster)
        
        return clusters


def get_clustering_algorithm(algorithm: str) -> ClusteringAlgorithm:
    """
    Factory function to get clustering algorithm by name.
    
    Args:
        algorithm: Algorithm name ('raptor' or 'sequential')
        
    Returns:
        ClusteringAlgorithm instance
    """
    if algorithm.lower() == "raptor":
        return RAPTORClustering()
    elif algorithm.lower() == "sequential":
        return SequentialClustering()
    else:
        logger.warning(f"Unknown clustering algorithm '{algorithm}', using RAPTOR")
        return RAPTORClustering()
