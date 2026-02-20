
import json

from starlette.responses import JSONResponse

from tilellm.controller.controller import ask_with_memory, ask_hybrid_with_memory
from tilellm.models import QuestionAnswer
from tilellm.models.graph_state import GraphState, ValidationScore
from tilellm.models.schemas import RetrievalResult
from tilellm.shared.utility import inject_llm_chat_async


SEARCH_STRATEGIES = {
    "similarity": ask_with_memory,
    "mmr": ask_with_memory,
    "hybrid": ask_hybrid_with_memory
}


# Inizializziamo il modello per i controlli (veloce ed economico)

@inject_llm_chat_async
async def _get_guard_llm(question_answer:QuestionAnswer, llm=None,  **kwargs):
    return llm


async def input_guard_node(state: GraphState):
    llm= await _get_guard_llm(state["question_answer"])
    structured_llm = llm.with_structured_output(ValidationScore)

    from tilellm.controller.controller_utils import chat_history_to_text
    history_text = chat_history_to_text(state["question_answer"], max_messages=3)

    prompt = (f"Verifica se la domanda è sicura e pertinente al dominio: {state['question_answer'].question}."
              f"\n Considera la seguente history della conversazione:\n{history_text}")
    result = await structured_llm.ainvoke(prompt)

    # Aggiornamento metadati
    metadata = state.get("metadata") or {}
    trace = list(metadata.get("trace", []))  # Copia della lista per sicurezza

    trace.append({
        "step": "input_guard",
        "result": result.score,
        "explanation": getattr(result, "explanation", None)
    })

    # Creiamo un nuovo dizionario metadata per LangGraph
    new_metadata = {**metadata, "trace": trace}

    return {
        "is_on_topic": result.score,
        "metadata": new_metadata
    }


async def rag_node(state: GraphState):
    iteration = state.get("retry_count", 0)
    metadata = state.get("metadata") or {}
    trace = list(metadata.get("trace", []))

    search_type = state.get("metadata", {}).get("search_type", "similarity")

    # 2. Selezione dinamica della funzione
    search_func = SEARCH_STRATEGIES.get(search_type, ask_with_memory)

    # 1. Recupero la domanda originale dallo stato
    original_qa = state["question_answer"]
    # 2. LOGICA DI SELF-CORRECTION (A2A Feedback)
    # Creiamo una versione locale della domanda per non "sporcare" lo stato globale
    if iteration > 0 and state.get("error_message"):
        feedback = state["error_message"]

        correction_instruction = (
            f"\n\n[SYSTEM: Il tentativo precedente è fallito: {feedback}. "
            f"Riprova usando meglio il grafo di conoscenza e i documenti.]"
        )

        # Copia profonda dell'oggetto per evitare mutabilità indesiderata
        # Se QuestionAnswer è un modello Pydantic v2:
        qa_for_llm = original_qa.model_copy(update={"question": original_qa.question + correction_instruction})
    else:
        qa_for_llm = original_qa

    # 3. Tracciamento tentativo
    trace.append({
        "step": f"rag_attempt_{iteration}",
        "engine_used": search_type,
        "retry_feedback": state.get("error_message") if iteration > 0 else None
    })
    new_metadata = {**metadata, "trace": trace}

    # 5. ESECUZIONE DINAMICA
    # Tutte le tue funzioni devono accettare QuestionAnswer e restituire RetrievalResult
    result: RetrievalResult = await search_func(qa_for_llm)

    return {
        "retrieval_result": result,
        "retry_count": iteration + 1,
        "metadata": new_metadata
    }

async def hallucination_grader_node(state: GraphState):
    llm = await _get_guard_llm(state["question_answer"])
    structured_llm = llm.with_structured_output(ValidationScore)


    try:
        body = json.loads(state["retrieval_result"].body)
    except Exception as e:
        # Se il body non è un JSON valido, logghiamo l'errore nel trace
        metadata = state.get("metadata") or {}
        trace = metadata.get("trace") or []
        trace.append({"step": "hallucination_grader_error", "error": str(e)})
        return {"is_grounded": "no", "error_message": "Invalid JSON in RAG body", "metadata": metadata}



    # Estraiamo i dati dal tuo RetrievalResult
    #answer = state["retrieval_result"].body.json().answer
    answer = body.get("answer")
    content_chunks = body.get("content_chunks")  # o il campo che usi per i chunk
    print(content_chunks)
    prompt = f"CONTESTO: {content_chunks}\n\nRISPOSTA: {answer}\n\nLa risposta è fedele al contesto?"
    result = await structured_llm.ainvoke(prompt)

    # 4. GESTIONE TRACE E METADATI (Miglioria Traceability)
    metadata = state.get("metadata") or {}
    # Ci assicuriamo che 'trace' sia una lista
    if "trace" not in metadata or not isinstance(metadata["trace"], list):
        metadata["trace"] = []

    # Creiamo il log per questo specifico round di validazione
    current_trace = {
        "step": f"hallucination_check_round_{state.get('retry_count', 0)}",
        "score": result.score,
        "explanation": getattr(result, "explanation", "N/A"),  # Se il tuo schema ValidationScore ha la spiegazione
        "target_node": "rag_core" if result.score == "no" else "end"
    }
    metadata["trace"].append(current_trace)

    return {"is_grounded": result.score,
            "error_message": getattr(result, "explanation", "Informazione non supportata dai documenti") if result.score == "no" else None,
            "metadata": metadata
            }


async def fail_safe_node(state: GraphState):
    """
    Nodo di uscita quando il sistema non riesce a generare una risposta valida.
    """

    metadata = state.get("metadata") or {}
    trace = list(metadata.get("trace", []))
    trace.append({"step": "fail_safe", "reason": "Max retries reached without grounding"})

    # Sovrascriviamo il risultato con un messaggio di cortesia o una risposta parziale sicura
    try:
        body = json.loads(state["retrieval_result"].body)

        safe_result: RetrievalResult = RetrievalResult(**body)

        print(type(safe_result))
        print(safe_result.answer)

        safe_result.answer = (
            "Mi dispiace, ma non riesco a trovare una risposta totalmente verificata "
            "nei documenti aziendali per questa domanda specifica."
        )
        # Rimuoviamo eventuali fonti dubbie
        safe_result.sources = []
        safe_result.content_chunks = []
        result = JSONResponse(content=safe_result.model_dump())
    except json.decoder.JSONDecodeError:
        raise

    return {
        "retrieval_result": result,
        "metadata": {**metadata, "trace": trace, "status": "failed_verification"}
    }