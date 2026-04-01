"""
RAPTOR Module - Recursive Abstractive Processing for Tree-Organized Retrieval

Implements hierarchical summarization for long documents using LangGraph.

Reference: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
(Sarthi et al., arXiv:2401.18059, 2024)

Level 0: Raw chunks (already in vector store)
Level 1: LLM summaries of groups of 5-10 chunks → indexed ({doc_id}_summaries)
Level 2+: LLM summaries of summaries → indexed (for very long documents)

Retrieval strategies:
- Collapsed tree: All nodes (leaves + summaries) in same vector space
- Tree traversal: Agent decides which level to search based on query complexity
"""

from .controllers import router
from tilellm.modules.raptor.models.models import (
    RaptorConfig,
    RaptorTree,
    RaptorNode,
    RaptorLevel,
    RaptorRetrievalStrategy,
    RaptorRequest,
    RaptorResponse,
    RaptorSummaryRequest,
    RaptorSummaryResponse,
)
from .services.raptor_service import RaptorService
from .agents.summary_agent import create_summary_agent_workflow

__all__ = [
    "router",
    "RaptorConfig",
    "RaptorTree",
    "RaptorNode",
    "RaptorLevel",
    "RaptorRetrievalStrategy",
    "RaptorRequest",
    "RaptorResponse",
    "RaptorSummaryRequest",
    "RaptorSummaryResponse",
    "RaptorService",
    "create_summary_agent_workflow",
]
