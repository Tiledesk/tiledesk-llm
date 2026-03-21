from langgraph.graph import StateGraph, END
from tilellm.agents.nodes import (
    input_guard_node,
    intent_router_node,
    compliance_node,
    rag_node,
    hallucination_grader_node,
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
workflow.add_node("compliance_node", compliance_node)
workflow.add_node("rag_core", rag_node)
workflow.add_node("validatore", hallucination_grader_node)
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
        "qa": "rag_core",
    }
)

# Compliance chiude direttamente (already grounded=yes)
workflow.add_edge("compliance_node", END)

# --- LOGICA 3: Ciclo RAG <-> Validatore ---
workflow.add_edge("rag_core", "validatore")

workflow.add_conditional_edges(
    "validatore",
    _router_validazione,
    {
        "concludi": END,
        "riprova": "rag_core",
        "esaurito": "fail_safe_node",
    }
)

workflow.add_edge("fail_safe_node", END)

app = workflow.compile()