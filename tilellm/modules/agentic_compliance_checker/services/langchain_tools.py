"""
LangChain @tool adapters over tools_core's _core coroutines.

No business logic and no tracing here — both live in tools_core.py, shared
with the MCP server adapter (services/mcp_server.py, P6). This file only
translates argument/return shapes for langchain.agents.create_agent, the
agent builder tilellm/controller/controller.py already uses for /api/ask.
"""
import json
from typing import Awaitable, List, Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tilellm.modules.agentic_compliance_checker.models import EvidenceNotFound, SessionNotFound
from tilellm.modules.agentic_compliance_checker.services.tools_core import (
    build_report_core,
    evaluate_criteria_core,
    list_requirements_core,
    retrieve_evidence_core,
)


async def _safe(coro: Awaitable[str]) -> str:
    """Every tool call flows through this: an unknown session/evidence_ref or a
    bad argument combination (e.g. a criterion id that doesn't exist) becomes a
    JSON error string the agent can read and act on, not a raised exception
    that would end the agent's turn."""
    try:
        return await coro
    except (SessionNotFound, EvidenceNotFound, ValueError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


class ListRequirementsArgs(BaseModel):
    session_id: str = Field(
        description="Identificativo della sessione di verifica, fornito dall'utente."
    )
    kind: Literal["all", "tabular", "discretionary"] = "all"
    status: Literal["all", "pending", "done", "human_review"] = "all"
    operator: Optional[str] = Field(
        default=None,
        description="Etichetta o namespace dell'operatore economico. Omettere se la gara ha un solo operatore.",
    )


@tool(args_schema=ListRequirementsArgs)
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
    session_id: str = Field(description="Identificativo della sessione di verifica.")
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


@tool(args_schema=RetrieveEvidenceArgs)
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
    session_id: str = Field(description="Identificativo della sessione di verifica.")
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


@tool(args_schema=EvaluateCriteriaArgs)
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


class BuildReportArgs(BaseModel):
    session_id: str = Field(description="Identificativo della sessione di verifica.")
    operator: Optional[str] = Field(
        default=None, description="Etichetta o namespace dell'operatore economico. Omettere se la gara ha un solo operatore."
    )


@tool(args_schema=BuildReportArgs)
async def compliance_build_report(session_id: str, operator: Optional[str] = None) -> str:
    """Produce il riepilogo di conformità per un operatore economico a partire
    ESCLUSIVAMENTE dalle valutazioni già registrate nella sessione. Non valuta nulla
    e non accetta punteggi: il riepilogo è sempre ricalcolato dallo stato salvato."""
    return await _safe(build_report_core(session_id=session_id, operator=operator))


# name -> LangChain tool object, mirrors tools_registry.TOOL_REGISTRY's shape
# for the entries controllers.py registers there.
AGENTIC_COMPLIANCE_TOOLS = {
    "compliance_list_requirements": compliance_list_requirements,
    "compliance_retrieve_evidence": compliance_retrieve_evidence,
    "compliance_evaluate_criteria": compliance_evaluate_criteria,
    "compliance_build_report": compliance_build_report,
}
