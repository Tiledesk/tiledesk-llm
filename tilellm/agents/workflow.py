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


# Configurazione del Grafo
workflow = StateGraph(GraphState)

# Nodi
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

# Entry Point
workflow.set_entry_point("guardia")

# --- LOGICA 1: Guardrail ingresso ---
workflow.add_conditional_edges(
    "guardia",
    lambda x: "procedi" if x.get("is_on_topic") != "no" else "rifiuta",
    {
        "procedi": "intent_router",
        "rifiuta": END,
    }
)

# --- LOGICA 2: Intent routing ---
workflow.add_conditional_edges(
    "intent_router",
    lambda x: "compliance" if x.get("intent") == "compliance" else "qa",
    {
        "compliance": "compliance_node",
        "qa": "cache_lookup",   # cache check before HyDE / retrieval
    }
)

# Compliance chiude direttamente (already grounded=yes)
workflow.add_edge("compliance_node", END)

# --- LOGICA 3: Cache hit/miss routing ---
def _router_cache(state):
    """Skip retrieval entirely on cache hit."""
    return "hit" if state.get("cache_hit") else "miss"

workflow.add_conditional_edges(
    "cache_lookup",
    _router_cache,
    {
        "hit": END,
        "miss": "hyde",
    }
)

# HyDE -> Router (choose between RAPTOR or standard RAG)
def _router_retrieval_method(state):
    """Route to RAPTOR or standard RAG based on use_raptor flag."""
    qa = state.get("question_answer")
    if qa and qa.use_raptor:
        return "raptor"
    return "rag"

workflow.add_conditional_edges(
    "hyde",
    _router_retrieval_method,
    {
        "raptor": "raptor",
        "rag": "rag_core",
    }
)

# --- LOGICA 4: Ciclo RAG/RAPTOR <-> Validatore ---
workflow.add_edge("rag_core", "validatore")
workflow.add_edge("raptor", "validatore")

workflow.add_conditional_edges(
    "validatore",
    _router_validazione,
    {
        "concludi": "cache_store",   # store result before returning
        "riprova": "rag_core",
        "esaurito": "fail_safe_node",
    }
)

workflow.add_edge("cache_store", END)
workflow.add_edge("fail_safe_node", END)

app = workflow.compile()