# from collections import Counter
# from transformers import BertTokenizerFast

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

def hybrid_score_norm(dense, sparse, alpha: float):
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: float between 0 and 1 where 0 == sparse only
                       and 1 == dense only
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs


def hybrid_rrf(dense, sparse, k=60):
    """Reciprocal Rank Fusion (RRF) for combining two ranked lists.

    RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)

    Args:
        dense: List of float scores from dense retrieval (higher is better)
        sparse: List of float scores from sparse retrieval (higher is better)
        k: Smoothing constant (default 60)

    Returns:
        List of combined RRF scores in same order as input lists.
        
    Note:
        Assumes dense and sparse are aligned (same documents in same order).
        Ranks are determined by sorting scores in descending order.
    """
    if len(dense) != len(sparse):
        raise ValueError("dense and sparse must have same length")
    
    # Create list of indices
    indices = list(range(len(dense)))
    
    # Sort indices by dense scores descending
    dense_ranked = sorted(indices, key=lambda i: dense[i], reverse=True)
    # Assign ranks (0 for highest score)
    dense_ranks = {idx: rank for rank, idx in enumerate(dense_ranked)}
    
    # Sort indices by sparse scores descending
    sparse_ranked = sorted(indices, key=lambda i: sparse[i], reverse=True)
    sparse_ranks = {idx: rank for rank, idx in enumerate(sparse_ranked)}
    
    # Calculate RRF scores
    rrf_scores = []
    for i in indices:
        rrf_score = 1/(k + dense_ranks[i]) + 1/(k + sparse_ranks[i])
        rrf_scores.append(rrf_score)
    
    return rrf_scores




class HybridRetriever(BaseRetriever):
    """

    """

    documents: List[Document]
    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""


        if len(self.documents) > self.k:
            return self.documents[0:self.k]
        else:
            return self.documents


    # Optional: Provide a more efficient native implementation by overriding
    # _aget_relevant_documents
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.

    #    Args:
    #         query: String to find relevant documents for
    #         run_manager: The callbacks handler to use

    #     Returns:
    #         List of relevant documents
    #     """
        if len(self.documents) > self.k:
            return self.documents[0:self.k]
        else:
            return self.documents
