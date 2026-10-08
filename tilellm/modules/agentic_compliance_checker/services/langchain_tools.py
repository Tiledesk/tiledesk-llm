"""
LangChain tool adapters over tools_core's _core coroutines.

No business logic and no tracing here — both live in tools_core.py, shared
with the MCP server adapter (services/mcp_server.py, P6). This file only
translates argument/return shapes for langchain.agents.create_agent, the
agent builder tilellm/controller/controller.py already uses for /api/ask.

The session is bound server side, never passed by the model: the functions
below take session_id, but the schemas the LLM sees don't have it. build_tool()
closes over the session (from QuestionToLLM.compliance_session_id on /api/ask,
from the X-Compliance-Session header on MCP), so a session_id can't leak from
the prompt nor be swapped by the agent.
"""
import json
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Literal, NamedTuple, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tilellm.modules.agentic_compliance_checker.models import (
    EvidenceNotFound,
    SessionNotFound,
    TraceIncompleteError,
)
from tilellm.modules.agentic_compliance_checker.services.tools_core import (
    build_report_core,
    check_l01_core,
    check_tabular_core,
    evaluate_criteria_core,
    list_requirements_core,
    resolve_proportional_core,
    retrieve_evidence_core,
)


async def _safe(coro: Awaitable[str]) -> str:
    """Every tool call flows through this: an unknown session/evidence_ref or a
    bad argument combination (e.g. a criterion id that doesn't exist) becomes a
    JSON error string the agent can read and act on, not a raised exception
    that would end the agent's turn."""
    try:
        return await coro
    except (SessionNotFound, EvidenceNotFound, TraceIncompleteError, ValueError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


class ListRequirementsArgs(BaseModel):
    kind: Literal["all", "tabular", "discretionary"] = "all"
    status: Literal["all", "pending", "done", "human_review"] = "all"
    operator: Optional[str] = Field(
        default=None,
        description="Etichetta o namespace dell'operatore economico. Omettere se la gara ha un solo operatore.",
    )


async def compliance_list_requirements(
    session_id: str, kind: str = "all", status: str = "all", operator: Optional[str] = None,
) -> str:
    """Elenca i requisiti e i criteri di una gara pubblica aperta in una sessione di
    verifica, con il loro stato di valutazione. Da chiamare SEMPRE per prima cosa:
    restituisce gli id dei criteri necessari a tutti gli altri tool di compliance.
    Non recupera evidenze e non valuta nulla."""
    return await _safe(list_requirements_core(
        session_id=session_id, kind=kind, status=status, operator=operator,
    ))


class RetrieveEvidenceArgs(BaseModel):
    criterion_id: Optional[str] = Field(
        default=None, description="Id del criterio da cercare (usa il suo testo come query di ricerca)."
    )
    query: Optional[str] = Field(
        default=None, description="Query di ricerca alternativa. Usa questa oppure criterion_id, non entrambe."
    )
    operator: Optional[str] = Field(
        default=None, description="Etichetta o namespace dell'operatore economico. Omettere se la gara ha un solo operatore."
    )
    top_k: Optional[int] = Field(default=None, description="Numero di passaggi da recuperare (default: quello configurato).")
    include_chiarimenti: bool = Field(
        default=False, description="Includere i documenti di chiarimento gara (normalmente esclusi)."
    )


async def compliance_retrieve_evidence(
    session_id: str, criterion_id: Optional[str] = None, query: Optional[str] = None,
    operator: Optional[str] = None, top_k: Optional[int] = None, include_chiarimenti: bool = False,
) -> str:
    """Recupera dai documenti di offerta di un operatore economico i passaggi rilevanti
    per un criterio (o per una query libera), e li memorizza nella sessione restituendo
    un riferimento opaco (evidence_ref) più una breve anteprima. Utile per ispezionare
    le evidenze PRIMA di valutare, o per ritentare con una formulazione diversa.
    NON assegna punteggi né esprime giudizi."""
    return await _safe(retrieve_evidence_core(
        session_id=session_id, criterion_id=criterion_id, query=query,
        operator=operator, top_k=top_k, include_chiarimenti=include_chiarimenti,
    ))


class EvaluateCriteriaArgs(BaseModel):
    criterion_ids: List[str] = Field(description="Uno o più id di criteri discrezionali da valutare.")
    operators: Optional[List[str]] = Field(
        default=None, description="Operatori economici da valutare. Default: tutti quelli della sessione."
    )
    evidence_ref: Optional[str] = Field(
        default=None,
        description="Valuta usando queste evidenze già recuperate invece di recuperarne di nuove. "
                    "Ammesso un solo criterio e un solo operatore.",
    )
    retrieval_query: Optional[str] = Field(
        default=None, description="Formulazione alternativa per il recupero delle evidenze."
    )
    reason: Optional[str] = Field(
        default=None, description="Perché stai (ri)valutando questi criteri. Registrato nella traccia di audit."
    )


async def compliance_evaluate_criteria(
    session_id: str, criterion_ids: List[str], operators: Optional[List[str]] = None,
    evidence_ref: Optional[str] = None, retrieval_query: Optional[str] = None,
    reason: Optional[str] = None,
) -> str:
    """Valuta uno o più criteri discrezionali di una gara sui documenti di uno o più
    operatori economici. Recupera le evidenze, interroga il giudice LLM e applica le
    regole deterministiche di punteggio e di controllo (soglia di confidenza, coerenza
    della citazione, fallback HyDE, modalità proporzionale). Il punteggio è calcolato
    dal sistema: non può essere proposto, suggerito o modificato da chi chiama il tool."""
    return await _safe(evaluate_criteria_core(
        session_id=session_id, criterion_ids=criterion_ids, operators=operators,
        evidence_ref=evidence_ref, retrieval_query=retrieval_query, reason=reason,
    ))


class CheckTabularArgs(BaseModel):
    requirement_ids: Optional[List[str]] = Field(
        default=None,
        description="Id dei requisiti tabellari da verificare. Default: tutti quelli del lotto.",
    )
    operators: Optional[List[str]] = Field(
        default=None, description="Operatori economici da verificare. Default: tutti quelli della sessione."
    )


async def compliance_check_tabular(
    session_id: str, requirement_ids: Optional[List[str]] = None, operators: Optional[List[str]] = None,
) -> str:
    """Verifica i requisiti tabellari (presenza/assenza, es. certificazioni, requisiti
    obbligatori) di uno o più operatori economici. Usa lo stesso motore di giudizio
    dell'endpoint /v2/check: recupera evidenze e produce un giudizio SI/NO/PARZIALE/N.V.
    per ciascun requisito. Il giudizio è calcolato dal sistema, non dall'agente."""
    return await _safe(check_tabular_core(
        session_id=session_id, requirement_ids=requirement_ids, operators=operators,
    ))


class CheckL01Args(BaseModel):
    operators: Optional[List[str]] = Field(
        default=None, description="Operatori economici da verificare. Default: tutti quelli della sessione."
    )


async def compliance_check_l01(session_id: str, operators: Optional[List[str]] = None) -> str:
    """Verifica la coerenza tra il listino L01 di uno o più operatori economici e le
    relative schede tecniche PDF già indicizzate: per ogni prodotto a listino controlla
    che esista una scheda che lo supporti. Nessuna chiamata LLM: puro riscontro
    codice/nome. Attivo solo se la sessione ha un l01_xlsx_url configurato per
    l'operatore; altrimenti riporta used=False, non è un errore."""
    return await _safe(check_l01_core(session_id=session_id, operators=operators))


class ResolveProportionalArgs(BaseModel):
    criterion_ids: Optional[List[str]] = Field(
        default=None,
        description="Id dei criteri proporzionali da risolvere. Default: tutti quelli del lotto.",
    )
    allow_partial: bool = Field(
        default=False,
        description="Procedi anche se non tutti gli operatori hanno una valutazione per i criteri "
                    "target (normalmente rifiutato: falserebbe il confronto in modo silenzioso).",
    )


async def compliance_resolve_proportional(
    session_id: str, criterion_ids: Optional[List[str]] = None, allow_partial: bool = False,
) -> str:
    """Risolve i criteri proporzionali (es. 'ampiezza di gamma') confrontando la
    quantità misurata di TUTTI gli operatori economici della sessione: chi offre di
    più (o di meno, se il criterio è a direzione inversa) riceve il punteggio pieno,
    gli altri una quota proporzionale. Richiede che ogni operatore sia già stato
    valutato su questi criteri con compliance_evaluate_criteria — altrimenti rifiuta,
    a meno di allow_partial=True. Per i criteri la cui quantità viene dal listino L01
    (es. ampiezza di gamma) chiama PRIMA compliance_check_l01 per ogni operatore: la
    risposta elenca in 'l01_not_checked' gli operatori per cui non è stato fatto.
    Il punteggio resta 'proposta da confermare' (human_review_required) per design,
    non è mai definitivo automaticamente."""
    return await _safe(resolve_proportional_core(
        session_id=session_id, criterion_ids=criterion_ids, allow_partial=allow_partial,
    ))


class BuildReportArgs(BaseModel):
    operator: Optional[str] = Field(
        default=None, description="Etichetta o namespace dell'operatore economico. Omettere se la gara ha un solo operatore."
    )


async def compliance_build_report(session_id: str, operator: Optional[str] = None) -> str:
    """Produce il riepilogo di conformità per un operatore economico a partire
    ESCLUSIVAMENTE dalle valutazioni già registrate nella sessione. Non valuta nulla
    e non accetta punteggi: il riepilogo è sempre ricalcolato dallo stato salvato."""
    return await _safe(build_report_core(session_id=session_id, operator=operator))


class ToolSpec(NamedTuple):
    function: Callable[..., Awaitable[str]]  # takes session_id + the schema's arguments
    args_schema: Type[BaseModel]              # what the model sees: never session_id


AGENTIC_COMPLIANCE_TOOLS: Dict[str, ToolSpec] = {
    "compliance_list_requirements": ToolSpec(compliance_list_requirements, ListRequirementsArgs),
    "compliance_retrieve_evidence": ToolSpec(compliance_retrieve_evidence, RetrieveEvidenceArgs),
    "compliance_evaluate_criteria": ToolSpec(compliance_evaluate_criteria, EvaluateCriteriaArgs),
    "compliance_check_tabular": ToolSpec(compliance_check_tabular, CheckTabularArgs),
    "compliance_check_l01": ToolSpec(compliance_check_l01, CheckL01Args),
    "compliance_resolve_proportional": ToolSpec(compliance_resolve_proportional, ResolveProportionalArgs),
    "compliance_build_report": ToolSpec(compliance_build_report, BuildReportArgs),
}


def tool_description(name: str) -> str:
    return (AGENTIC_COMPLIANCE_TOOLS[name].function.__doc__ or "").strip()


def build_tool(name: str, session_id: str) -> StructuredTool:
    """A LangChain tool bound to one session. Arguments outside the schema —
    a session_id the model invents included — are dropped by args_schema."""
    spec = AGENTIC_COMPLIANCE_TOOLS[name]

    async def run(**kwargs: Any) -> str:
        return await spec.function(session_id=session_id, **kwargs)

    return StructuredTool.from_function(
        coroutine=run, name=name, description=tool_description(name), args_schema=spec.args_schema,
    )


def _registry_factory(name: str, compliance_session_id: Optional[str] = None, **_: Any) -> StructuredTool:
    if not compliance_session_id:
        raise ValueError(f"{name}: compliance_session_id is required to use the compliance tools")
    return build_tool(name, compliance_session_id)


def registry_factories() -> Dict[str, Callable[..., StructuredTool]]:
    """name -> factory for tools_registry (is_factory=True): resolve_tools()
    calls it per request with that request's compliance_session_id."""
    return {name: partial(_registry_factory, name) for name in AGENTIC_COMPLIANCE_TOOLS}
