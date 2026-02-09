"""
Utils module for Knowledge Graph operations.
"""

from .prompts import (
    GRAPH_QA_SYSTEM_PROMPT,
    GRAPH_QA_PROMPT_TEMPLATE,
    ADVANCED_GRAPH_QA_SYSTEM_PROMPT,
    ADVANCED_GRAPH_QA_PROMPT_TEMPLATE,
    format_community_reports,
    format_document_excerpts,
    format_graph_context,
    format_chat_history
)

from .rrf import (
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_with_metadata,
    normalize_scores
)

from .query_analysis import (
    QueryType,
    detect_query_type_with_llm,
    detect_query_type_heuristic,
    get_weight_adjustments,
    apply_weight_adjustments
)

from .synthetic_qa import (
    generate_synthetic_questions,
    enrich_reports_with_synthetic_qa,
    format_report_with_questions_for_indexing,
    parse_questions_from_text
)

from .graph_expansion import (
    GraphExpander
)

__all__ = [
    # Prompts
    "GRAPH_QA_SYSTEM_PROMPT",
    "GRAPH_QA_PROMPT_TEMPLATE",
    "ADVANCED_GRAPH_QA_SYSTEM_PROMPT",
    "ADVANCED_GRAPH_QA_PROMPT_TEMPLATE",
    "format_community_reports",
    "format_document_excerpts",
    "format_graph_context",
    "format_chat_history",
    # RRF
    "reciprocal_rank_fusion",
    "reciprocal_rank_fusion_with_metadata",
    "normalize_scores",
    # Query Analysis
    "QueryType",
    "detect_query_type_with_llm",
    "detect_query_type_heuristic",
    "get_weight_adjustments",
    "apply_weight_adjustments",
    # Synthetic QA
    "generate_synthetic_questions",
    "enrich_reports_with_synthetic_qa",
    "format_report_with_questions_for_indexing",
    "parse_questions_from_text",
    # Graph Expansion
    "GraphExpander"
]