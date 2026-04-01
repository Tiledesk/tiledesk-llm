
import json
import logging
from typing import Optional, Literal

from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from tilellm.controller.controller import ask_with_memory, ask_hybrid_with_memory
from tilellm.models import QuestionAnswer
from tilellm.models.graph_state import GraphState, ValidationScore
from tilellm.models.schemas import RetrievalResult
from tilellm.shared.utility import inject_llm_chat_async

logger = logging.getLogger(__name__)

SEARCH_STRATEGIES = {
    "similarity": ask_with_memory,
    "mmr": ask_with_memory,
    "hybrid": ask_hybrid_with_memory
}


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM outputs
# ---------------------------------------------------------------------------

class HyDEDocument(BaseModel):
    hypothetical_document: str = Field(
        description="A hypothetical passage that would directly answer the query"
    )


# ---------------------------------------------------------------------------
# Shared LLM getter (cached via TimedCache inside the decorator)
# ---------------------------------------------------------------------------

@inject_llm_chat_async
async def _get_guard_llm(question_answer: QuestionAnswer, llm=None, **kwargs):
    return llm


# ---------------------------------------------------------------------------
# Node: hyde (Hypothetical Document Embeddings)
# ---------------------------------------------------------------------------

async def hyde_node(state: GraphState) -> dict:
    """
    Generate a hypothetical document based on the query for improved retrieval.

    If use_hyde=False, this node passes through without modifications.
    If use_hyde=True, generates a hypothetical passage and sets it as retrieval_query.
    The original question is preserved for answer generation.
    """
    qa = state["question_answer"]

    # Pass-through if HyDE is disabled
    if not qa.use_hyde:
        return {}

    logger.info(f"HyDE enabled - generating hypothetical document for query: {qa.question}")

    # Get LLM via DI
    llm = await _get_guard_llm(qa)
    structured_llm = llm.with_structured_output(HyDEDocument)

    # Prompt for generating hypothetical document
    prompt = f"""Generate a hypothetical document passage that would directly answer the following question.
Write it as if it were an excerpt from a relevant document, without citations or sources.
Be concise (2-3 paragraphs max).

Question: {qa.question}"""

    result = await structured_llm.ainvoke(prompt)

    # Update question_answer: set retrieval_query to the hypothetical document
    # The original question is preserved for answer generation
    updated_qa = qa.model_copy(update={"retrieval_query": result.hypothetical_document})

    logger.info(f"HyDE document generated (length={len(result.hypothetical_document)})")

    return {
        "question_answer": updated_qa,
        "hyde_document": result.hypothetical_document,
    }


# ---------------------------------------------------------------------------
# Node: input_guard  (guardrail sull'input)
# ---------------------------------------------------------------------------

async def input_guard_node(state: GraphState):
    llm = await _get_guard_llm(state["question_answer"])
    structured_llm = llm.with_structured_output(ValidationScore)

    from tilellm.controller.controller_utils import chat_history_to_text
    history_text = chat_history_to_text(state["question_answer"], max_messages=3)

    prompt = (
        f"Valuta se la seguente domanda è sicura e ragionevolmente pertinente al contesto aziendale/documentale.\n"
        f"Rispondi 'yes' in caso di dubbio: blocca solo domande chiaramente offensive, dannose o del tutto prive di senso.\n"
        f"Domanda: {state['question_answer'].question}\n"
        f"History della conversazione:\n{history_text}"
    )
    result = await structured_llm.ainvoke(prompt)

    metadata = state.get("metadata") or {}
    trace = list(metadata.get("trace", []))
    trace.append({
        "step": "input_guard",
        "result": result.score,
        "explanation": getattr(result, "explanation", None)
    })

    return {
        "is_on_topic": result.score,
        "metadata": {**metadata, "trace": trace}
    }


# ---------------------------------------------------------------------------
# Node: intent_router  (classifica intent + estrae CSV se compliance)
# ---------------------------------------------------------------------------

class IntentExtractionResult(BaseModel):
    intent: Literal["compliance", "qa"] = Field(
        description=(
            "Usa 'compliance' se la richiesta chiede di verificare, valutare o controllare "
            "se dei requisiti/criteri sono soddisfatti da un documento/offerta. "
            "Usa 'qa' per qualsiasi altra domanda, ricerca o approfondimento."
        )
    )
    csv_requirements: Optional[str] = Field(
        default=None,
        description=(
            "Solo se intent='compliance': estrai la tabella dei requisiti dal messaggio "
            "preservando il formato CSV originale (prima colonna = testo del requisito). "
            "Se non c'è una tabella esplicita ma ci sono requisiti elencati, convertili in CSV. "
            "Null se intent='qa'."
        )
    )
    domain: str = Field(
        default="e_procurement",
        description=(
            "Dominio di compliance più appropriato: "
            "medical_devices | e_procurement | hr_assessment | legal_audit. "
            "Usa 'medical_devices' per gare ospedaliere/dispositivi medici (MDR, ISO, UNI EN). "
            "Usa 'e_procurement' per altre gare d'appalto pubbliche. "
            "Usa 'hr_assessment' per valutazione CV/competenze. "
            "Usa 'legal_audit' per audit normativi/legali."
        )
    )


async def intent_router_node(state: GraphState):
    llm = await _get_guard_llm(state["question_answer"])
    structured_llm = llm.with_structured_output(IntentExtractionResult)

    prompt = (
        f"Analizza questa richiesta e determina l'intento:\n\n"
        f"Richiesta: {state['question_answer'].question}\n\n"
        f"Rispondi con un JSON strutturato secondo lo schema richiesto."
    )

    try:
        result = await structured_llm.ainvoke(prompt)
    except Exception as e:
        logger.warning(f"intent_router LLM call failed, defaulting to 'qa': {e}")
        result = IntentExtractionResult(intent="qa")

    metadata = state.get("metadata") or {}
    trace = list(metadata.get("trace", []))
    trace.append({
        "step": "intent_router",
        "intent": result.intent,
        "domain": result.domain,
        "csv_extracted": result.csv_requirements is not None,
    })

    return {
        "intent": result.intent,
        "parsed_csv": result.csv_requirements,
        "compliance_domain": result.domain,
        "metadata": {**metadata, "trace": trace},
    }


# ---------------------------------------------------------------------------
# Node: compliance_node  (esegue il compliance check)
# ---------------------------------------------------------------------------

async def compliance_node(state: GraphState):
    from tilellm.modules.compliance_checker.models import ComplianceRequest
    from tilellm.modules.compliance_checker.logic import check_compliance
    from tilellm.modules.compliance_checker.prompts import get_builtin_config
    from tilellm.modules.compliance_checker.prompts.e_procurement import E_PROCUREMENT_CONFIG

    qa = state["question_answer"]
    domain = state.get("compliance_domain") or "e_procurement"
    csv_text = state.get("parsed_csv") or qa.question

    config = get_builtin_config(domain) or E_PROCUREMENT_CONFIG

    metadata = state.get("metadata") or {}
    trace = list(metadata.get("trace", []))

    try:
        request = ComplianceRequest(
            config=config,
            csv_requirements=csv_text,
            namespace=qa.namespace,
            engine=qa.engine,
            embedding=qa.embedding,
            sparse_encoder=qa.sparse_encoder,
            gptkey=qa.gptkey,
            model=qa.model if isinstance(qa.model, str) else qa.model.name,
            llm=qa.llm or "openai",
            temperature=qa.temperature,
            top_p=getattr(qa, "top_p", 1.0),
            max_tokens=qa.max_tokens,
            debug=qa.debug,
            top_k=qa.top_k,
            search_type=qa.search_type,
        )

        report = await check_compliance(request)
        s = report.summary

        answer = (
            f"**Verifica di conformità completata** (dominio: {domain})\n\n"
            f"| Totale | Conformi | Non conformi | Parziali | Non verificabili | Tasso |\n"
            f"|--------|----------|--------------|----------|------------------|-------|\n"
            f"| {s.total} | {s.compliant} | {s.non_compliant} | {s.partial} | {s.not_verifiable} | {s.compliance_rate:.0%} |\n\n"
        )
        for r in report.results:
            jmap = {"compliant": "✅ SI", "non_compliant": "❌ NO",
                    "partial": "⚠️ PARZIALE", "not_verifiable": "❓ N/V"}
            label = jmap.get(r.judgment, r.judgment)
            answer += f"- **{r.requirement_id}** {label} (conf. {r.confidence:.0%}): {r.justification}\n"

        answer += f"\n\n{report.to_markdown(judgment_map=config.judgment_map)}"
        # RTM CSV
        rtm_csv = report.to_rtm_csv(judgment_map=config.judgment_map)

        trace.append({
            "step": "compliance_node",
            "domain": domain,
            "total": s.total,
            "compliance_rate": s.compliance_rate,
        })

        retrieval_result = RetrievalResult(
            answer=answer,
            success=True,
            namespace=qa.namespace,
            id=qa.namespace,
        )

        return {
            "retrieval_result": retrieval_result,
            "compliance_report": {**report.model_dump(), "rtm_csv": rtm_csv},
            "is_grounded": "yes",
            "metadata": {**metadata, "trace": trace},
        }

    except Exception as e:
        logger.error(f"compliance_node failed: {e}", exc_info=True)
        trace.append({"step": "compliance_node", "error": str(e)})
        retrieval_result = RetrievalResult(
            answer=f"Errore durante il compliance check: {str(e)}",
            success=False,
            namespace=qa.namespace,
            id=qa.namespace,
        )
        return {
            "retrieval_result": retrieval_result,
            "is_grounded": "yes",  # non ritentare — errore strutturale
            "metadata": {**metadata, "trace": trace},
        }


# ---------------------------------------------------------------------------
# Node: rag_node  (RAG standard con self-correction)
# ---------------------------------------------------------------------------

async def rag_node(state: GraphState):
    iteration = state.get("retry_count", 0)
    metadata = state.get("metadata") or {}
    trace = list(metadata.get("trace", []))

    search_type = state.get("metadata", {}).get("search_type", "similarity")
    search_func = SEARCH_STRATEGIES.get(search_type, ask_with_memory)

    original_qa = state["question_answer"]
    if iteration > 0 and state.get("error_message"):
        feedback = state["error_message"]
        correction_instruction = (
            f"\n\n[SYSTEM: Il tentativo precedente è fallito: {feedback}. "
            f"Riprova usando meglio il grafo di conoscenza e i documenti.]"
        )
        qa_for_llm = original_qa.model_copy(update={"question": original_qa.question + correction_instruction})
    else:
        qa_for_llm = original_qa

    trace.append({
        "step": f"rag_attempt_{iteration}",
        "engine_used": search_type,
        "retry_feedback": state.get("error_message") if iteration > 0 else None
    })

    result: RetrievalResult = await search_func(qa_for_llm)

    return {
        "retrieval_result": result,
        "retry_count": iteration + 1,
        "metadata": {**metadata, "trace": trace},
    }


# ---------------------------------------------------------------------------
# Node: hallucination_grader_node
# ---------------------------------------------------------------------------

async def hallucination_grader_node(state: GraphState):
    llm = await _get_guard_llm(state["question_answer"])
    structured_llm = llm.with_structured_output(ValidationScore)

    try:
        body = json.loads(state["retrieval_result"].body)
    except Exception as e:
        metadata = state.get("metadata") or {}
        trace = metadata.get("trace") or []
        trace.append({"step": "hallucination_grader_error", "error": str(e)})
        return {"is_grounded": "no", "error_message": "Invalid JSON in RAG body", "metadata": metadata}

    answer = body.get("answer")
    content_chunks = body.get("content_chunks")

    # If no chunks available (debug=False or empty context), skip grounding check
    if not content_chunks:
        metadata = state.get("metadata") or {}
        if "trace" not in metadata or not isinstance(metadata["trace"], list):
            metadata["trace"] = []
        metadata["trace"].append({
            "step": f"hallucination_check_round_{state.get('retry_count', 0)}",
            "score": "yes",
            "explanation": "No content_chunks available — grounding check skipped",
            "target_node": "end"
        })
        return {"is_grounded": "yes", "error_message": None, "metadata": metadata}

    prompt = (
        f"Sei un validatore che verifica se una risposta è supportata dal contesto recuperato.\n"
        f"Rispondi 'yes' se la risposta è almeno parzialmente fondata sul contesto, anche se incompleta.\n"
        f"Rispondi 'no' solo se la risposta contiene affermazioni CHIARAMENTE inventate o contraddice il contesto.\n"
        f"In caso di dubbio, rispondi 'yes'.\n\n"
        f"CONTESTO:\n{content_chunks}\n\n"
        f"RISPOSTA:\n{answer}\n\n"
        f"La risposta è supportata dal contesto?"
    )
    result = await structured_llm.ainvoke(prompt)

    metadata = state.get("metadata") or {}
    if "trace" not in metadata or not isinstance(metadata["trace"], list):
        metadata["trace"] = []

    metadata["trace"].append({
        "step": f"hallucination_check_round_{state.get('retry_count', 0)}",
        "score": result.score,
        "explanation": getattr(result, "explanation", "N/A"),
        "target_node": "rag_core" if result.score == "no" else "end"
    })

    return {
        "is_grounded": result.score,
        "error_message": getattr(result, "explanation", "Informazione non supportata dai documenti") if result.score == "no" else None,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Node: fail_safe_node
# ---------------------------------------------------------------------------

async def fail_safe_node(state: GraphState):
    metadata = state.get("metadata") or {}
    trace = list(metadata.get("trace", []))
    trace.append({"step": "fail_safe", "reason": "Max retries reached without grounding"})

    try:
        raw = state["retrieval_result"]
        body = json.loads(raw.body) if hasattr(raw, "body") else raw.model_dump()
        safe_result: RetrievalResult = RetrievalResult(**body)
        safe_result.answer = (
            "Mi dispiace, ma non riesco a trovare una risposta totalmente verificata "
            "nei documenti aziendali per questa domanda specifica."
        )
        safe_result.sources = []
        safe_result.content_chunks = []
    except json.decoder.JSONDecodeError:
        raise

    return {
        "retrieval_result": safe_result,
        "metadata": {**metadata, "trace": trace, "status": "failed_verification"}
    }


# ---------------------------------------------------------------------------
# Node: raptor_node (RAPTOR-based retrieval)
# ---------------------------------------------------------------------------

async def raptor_node(state: GraphState):
    """
    RAPTOR-based retrieval with hierarchical tree traversal.

    Replaces standard RAG when use_raptor=True in QuestionAnswer.
    Retrieves from RAPTOR tree and generates answer with LLM.
    """
    iteration = state.get("retry_count", 0)
    metadata = state.get("metadata") or {}
    trace = list(metadata.get("trace", []))

    original_qa = state["question_answer"]

    # Build RAPTOR request from QuestionAnswer
    from tilellm.modules.raptor.models.models import RaptorQARequest, RaptorRetrievalStrategy
    from tilellm.modules.raptor.controllers import _raptor_qa_logic

    raptor_request = RaptorQARequest(
        question=original_qa.question,
        namespace=original_qa.namespace,
        doc_id=None,  # Search across all documents in namespace
        strategy=RaptorRetrievalStrategy.COLLAPSED_TREE,
        top_k=original_qa.top_k,
        top_k_per_level=3,
        engine=original_qa.engine,
        gptkey=original_qa.gptkey,
        model=original_qa.model,
        llm=original_qa.llm,
        embedding=original_qa.embedding,
        temperature=original_qa.temperature,
        max_tokens=original_qa.max_tokens,
        debug=original_qa.debug,
        sparse_encoder=original_qa.sparse_encoder,
        use_hybrid=True if original_qa.sparse_encoder else False,
        alpha=original_qa.alpha if hasattr(original_qa, 'alpha') and original_qa.alpha else 0.7,
    )

    trace.append({
        "step": f"raptor_attempt_{iteration}",
        "retrieval_method": "raptor_tree",
        "strategy": "collapsed_tree",
    })

    try:
        raptor_result = await _raptor_qa_logic(raptor_request)

        # Convert RAPTOR response to RetrievalResult format
        retrieval_result = RetrievalResult(
            answer=raptor_result.answer,
            sources=[{"title": f"Level {r.get('level')}", "url": ""} for r in raptor_result.retrieved_chunks[:3]],
            content_chunks=[r.get("content", "") for r in raptor_result.retrieved_chunks],
            status="success" if raptor_result.success else "failed",
        )

        return {
            "retrieval_result": retrieval_result,
            "retry_count": iteration + 1,
            "metadata": {**metadata, "trace": trace},
        }
    except Exception as e:
        logger.error(f"RAPTOR retrieval failed: {e}")
        return {
            "error_message": f"RAPTOR retrieval failed: {str(e)}",
            "retry_count": iteration + 1,
            "metadata": {**metadata, "trace": trace},
        }
