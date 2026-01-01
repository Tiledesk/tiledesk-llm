"""
Reciprocal Rank Fusion (RRF) utility for combining multiple ranking results.

RRF is a simple yet effective algorithm for combining rankings from different sources
(e.g., dense vector search and sparse keyword search).

Formula: RRF_score(doc) = Σ (1 / (k + rank(doc)))
where k is a constant (typically 60) and rank is the position in the ranking.
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
    weights: List[float] = None
) -> List[Tuple[str, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.

    Args:
        rankings: List of rankings, where each ranking is a list of document IDs
                 ordered by relevance (most relevant first)
        k: Constant for RRF formula (default: 60, as in original paper)
        weights: Optional weights for each ranking (default: equal weights)

    Returns:
        List of (document_id, score) tuples ordered by RRF score (descending)

    Example:
        >>> dense_results = ["doc1", "doc2", "doc3"]
        >>> sparse_results = ["doc2", "doc1", "doc4"]
        >>> rrf_results = reciprocal_rank_fusion([dense_results, sparse_results])
        >>> # doc2 will likely rank first as it appears high in both rankings
    """
    if not rankings:
        return []

    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0] * len(rankings)

    if len(weights) != len(rankings):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of rankings ({len(rankings)})")

    # Calculate RRF scores
    scores = defaultdict(float)

    for ranking_idx, ranking in enumerate(rankings):
        weight = weights[ranking_idx]
        for rank, doc_id in enumerate(ranking):
            # RRF formula: score += weight / (k + rank)
            # rank is 0-indexed, so rank+1 gives position
            rrf_score = weight / (k + rank + 1)
            scores[doc_id] += rrf_score

            logger.debug(f"Doc {doc_id} in ranking {ranking_idx} at position {rank+1}: +{rrf_score:.4f}")

    # Sort by score descending
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"RRF combined {len(rankings)} rankings with {sum(len(r) for r in rankings)} total items "
                f"into {len(sorted_results)} unique documents")

    return sorted_results


def reciprocal_rank_fusion_with_metadata(
    rankings_with_metadata: List[List[Dict[str, Any]]],
    id_field: str = "id",
    k: int = 60,
    weights: List[float] = None
) -> List[Dict[str, Any]]:
    """
    RRF for document objects with metadata.

    Args:
        rankings_with_metadata: List of rankings, where each ranking is a list of
                                document dictionaries with at least an ID field
        id_field: Name of the field containing the document ID (default: "id")
        k: Constant for RRF formula
        weights: Optional weights for each ranking

    Returns:
        List of document dictionaries with added "rrf_score" field,
        ordered by RRF score (descending)

    Example:
        >>> dense = [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.8}]
        >>> sparse = [{"id": "doc2", "score": 15.0}, {"id": "doc3", "score": 12.0}]
        >>> results = reciprocal_rank_fusion_with_metadata([dense, sparse])
    """
    if not rankings_with_metadata:
        return []

    # Extract ID rankings
    id_rankings = []
    doc_metadata_map = {}

    for ranking in rankings_with_metadata:
        id_ranking = []
        for doc in ranking:
            if id_field not in doc:
                logger.warning(f"Document missing ID field '{id_field}': {doc}")
                continue
            doc_id = doc[id_field]
            id_ranking.append(doc_id)
            # Store metadata (will use first occurrence if duplicates across rankings)
            if doc_id not in doc_metadata_map:
                doc_metadata_map[doc_id] = doc.copy()
        id_rankings.append(id_ranking)

    # Apply RRF
    rrf_scores = reciprocal_rank_fusion(id_rankings, k=k, weights=weights)

    # Combine with metadata
    results = []
    for doc_id, rrf_score in rrf_scores:
        if doc_id in doc_metadata_map:
            doc = doc_metadata_map[doc_id].copy()
            doc["rrf_score"] = rrf_score
            results.append(doc)
        else:
            logger.warning(f"Document ID {doc_id} not found in metadata map")

    return results


def normalize_scores(documents: List[Dict[str, Any]], score_field: str = "score") -> List[Dict[str, Any]]:
    """
    Normalize scores in a list of documents to [0, 1] range using min-max normalization.

    Args:
        documents: List of document dictionaries with scores
        score_field: Name of the score field to normalize

    Returns:
        List of documents with normalized scores in "normalized_score" field
    """
    if not documents:
        return []

    scores = [doc.get(score_field, 0.0) for doc in documents]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        # All scores are the same, normalize to 1.0
        for doc in documents:
            doc["normalized_score"] = 1.0
    else:
        for doc in documents:
            original_score = doc.get(score_field, 0.0)
            normalized = (original_score - min_score) / (max_score - min_score)
            doc["normalized_score"] = normalized

    return documents
