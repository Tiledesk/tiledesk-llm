from langgraph.graph import StateGraph, END
from tilellm.agents.nodes import input_guard_node, rag_node, hallucination_grader_node, fail_safe_node
from tilellm.models.graph_state import GraphState


# 1. Definiamo un'unica funzione di routing chiara
def _router_validazione(state: GraphState):
    """
    Gestisce tutto il traffico in uscita dal validatore.
    """
    # Se è tutto ok, finiamo
    if state.get("is_grounded") == "yes":
        return "concludi"

    # Se c'è un'allucinazione, controlliamo i tentativi
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        return "riprova"

    # Se abbiamo esaurito i tentativi, andiamo al nodo di sicurezza
    return "esaurito"


# 2. Configurazione del Grafo
workflow = StateGraph(GraphState)

# Aggiunta Nodi
workflow.add_node("guardia", input_guard_node)
workflow.add_node("rag_core", rag_node)
workflow.add_node("validatore", hallucination_grader_node)
workflow.add_node("fail_safe_node", fail_safe_node)

# Entry Point
workflow.set_entry_point("guardia")

# --- LOGICA 1: Ingresso ---
workflow.add_conditional_edges(
    "guardia",
    lambda x: "procedi" if x["is_on_topic"] == "yes" else "rifiuta",
    {
        "procedi": "rag_core",
        "rifiuta": END  # Qui potresti anche mettere un nodo 'rifiuto_node' se vuoi un messaggio custom
    }
)

# --- LOGICA 2: Ciclo RAG <-> Validatore ---
workflow.add_edge("rag_core", "validatore")

workflow.add_conditional_edges(
    "validatore",
    _router_validazione,
    {
        "concludi": END,
        "riprova": "rag_core",  # Loop di Self-Correction
        "esaurito": "fail_safe_node"  # Uscita di sicurezza
    }
)

# Il nodo fail_safe chiude sempre il grafo
workflow.add_edge("fail_safe_node", END)

app = workflow.compile()