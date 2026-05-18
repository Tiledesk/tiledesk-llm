from langgraph.graph import StateGraph, END
from tilellm.agents.nodes import (
    input_guard_node,
    intent_router_node,
    cache_lookup_node,
    hyde_node,
    compliance_node,
    rag_node,
    raptor_node,
    hallucination_grader_node,
    cache_store_node,
    fail_safe_node,
)
from tilellm.models.graph_state import GraphState


def _router_validazione(state: GraphState):
    if state.get("is_grounded") == "yes":
        return "concludi"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        return "riprova"

    return "esaurito"


# ---------------------------------------------------------------------------
# Full agentic workflow  →  /api/v2/query
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node("guardia", input_guard_node)
workflow.add_node("intent_router", intent_router_node)
workflow.add_node("cache_lookup", cache_lookup_node)
workflow.add_node("hyde", hyde_node)
workflow.add_node("compliance_node", compliance_node)
workflow.add_node("rag_core", rag_node)
workflow.add_node("raptor", raptor_node)
workflow.add_node("validatore", hallucination_grader_node)
workflow.add_node("cache_store", cache_store_node)
workflow.add_node("fail_safe_node", fail_safe_node)

workflow.set_entry_point("guardia")

workflow.add_conditional_edges(
    "guardia",
    lambda x: "procedi" if x.get("is_on_topic") != "no" else "rifiuta",
    {"procedi": "intent_router", "rifiuta": END},
)

workflow.add_conditional_edges(
    "intent_router",
    lambda x: "compliance" if x.get("intent") == "compliance" else "qa",
    {"compliance": "compliance_node", "qa": "cache_lookup"},
)

workflow.add_edge("compliance_node", END)


def _router_cache(state):
    return "hit" if state.get("cache_hit") else "miss"


workflow.add_conditional_edges(
    "cache_lookup",
    _router_cache,
    {"hit": END, "miss": "hyde"},
)


def _router_retrieval_method(state):
    qa = state.get("question_answer")
    if qa and qa.use_raptor:
        return "raptor"
    return "rag"


workflow.add_conditional_edges(
    "hyde",
    _router_retrieval_method,
    {"raptor": "raptor", "rag": "rag_core"},
)

workflow.add_edge("rag_core", "validatore")
workflow.add_edge("raptor", "validatore")

workflow.add_conditional_edges(
    "validatore",
    _router_validazione,
    {"concludi": "cache_store", "riprova": "rag_core", "esaurito": "fail_safe_node"},
)

workflow.add_edge("cache_store", END)
workflow.add_edge("fail_safe_node", END)

app = workflow.compile()


# ---------------------------------------------------------------------------
# Simple workflow  →  /api/v2/qa
# Nodi: guard (opzionale) → intent_router → compliance | rag → grader (opzionale)
# ---------------------------------------------------------------------------

def _simple_guard_router(state: GraphState):
    """Skip guardia if use_guard=False on the payload."""
    qa = state.get("question_answer")
    return "guard" if getattr(qa, "use_guard", True) else "skip_guard"


def _simple_after_rag_router(state: GraphState):
    """Skip hallucination grader if use_hallucination_grader=False on the payload."""
    qa = state.get("question_answer")
    return "grade" if getattr(qa, "use_hallucination_grader", True) else "skip_grade"


simple_workflow = StateGraph(GraphState)

simple_workflow.add_node("_start", lambda s: {})
simple_workflow.add_node("guardia", input_guard_node)
simple_workflow.add_node("intent_router", intent_router_node)
simple_workflow.add_node("compliance_node", compliance_node)
simple_workflow.add_node("rag_core", rag_node)
simple_workflow.add_node("validatore", hallucination_grader_node)
simple_workflow.add_node("fail_safe_node", fail_safe_node)

simple_workflow.set_entry_point("_start")

simple_workflow.add_conditional_edges(
    "_start",
    _simple_guard_router,
    {"guard": "guardia", "skip_guard": "intent_router"},
)

simple_workflow.add_conditional_edges(
    "guardia",
    lambda x: "procedi" if x.get("is_on_topic") != "no" else "rifiuta",
    {"procedi": "intent_router", "rifiuta": END},
)

simple_workflow.add_conditional_edges(
    "intent_router",
    lambda x: "compliance" if x.get("intent") == "compliance" else "qa",
    {"compliance": "compliance_node", "qa": "rag_core"},
)

simple_workflow.add_edge("compliance_node", END)

simple_workflow.add_conditional_edges(
    "rag_core",
    _simple_after_rag_router,
    {"grade": "validatore", "skip_grade": END},
)

simple_workflow.add_conditional_edges(
    "validatore",
    _router_validazione,
    {"concludi": END, "riprova": "rag_core", "esaurito": "fail_safe_node"},
)

simple_workflow.add_edge("fail_safe_node", END)

simple_app = simple_workflow.compile()