"""
Bridges an agentic session's stored config to compliance_checker's real
services. The ONLY file in this module that imports DiscretionaryCheckService
directly — if compliance_checker's constructor or per-operator request shape
ever changes, this is the one place to touch.
"""
from typing import Optional

from tilellm.modules.agentic_compliance_checker.services.deps import _resolve_deps
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.compliance_checker.models_v2 import (
    DiscretionaryCriterion,
    OperatorRef,
    TabularRequirementV2,
)
from tilellm.modules.compliance_checker.services.discretionary_check_service import (
    DiscretionaryCheckService,
)


async def resolve_operator(session_id: str, operator: Optional[str]) -> OperatorRef:
    """Resolve an 'operator' tool argument (label or namespace, or omitted) to
    its OperatorRef. When the session has exactly one operator and the caller
    didn't specify one, that operator is used — the common single-operator
    case needs no argument at all."""
    request = await SessionStore.get_request(session_id)
    if operator is None:
        if len(request.operators) == 1:
            return request.operators[0]
        raise ValueError(
            f"'operator' è obbligatorio: la sessione ha {len(request.operators)} operatori "
            f"({[op.operator_label or op.namespace for op in request.operators]})."
        )
    for op in request.operators:
        if operator in (op.operator_label, op.namespace):
            return op
    raise ValueError(
        f"Operatore '{operator}' non trovato in sessione. Disponibili: "
        f"{[op.operator_label or op.namespace for op in request.operators]}."
    )


async def resolve_criterion(session_id: str, criterion_id: str) -> DiscretionaryCriterion:
    lot = await SessionStore.get_lot(session_id)
    for c in lot.requirements.discretionary:
        if c.id == criterion_id:
            return c
    raise ValueError(f"Criterio discrezionale '{criterion_id}' non trovato nel lotto.")


async def resolve_tabular_requirement(session_id: str, requirement_id: str) -> TabularRequirementV2:
    lot = await SessionStore.get_lot(session_id)
    for r in lot.requirements.tabular:
        if r.id == requirement_id:
            return r
    raise ValueError(f"Requisito tabellare '{requirement_id}' non trovato nel lotto.")


async def build_service_for_operator(session_id: str, operator: OperatorRef) -> DiscretionaryCheckService:
    """One fresh DiscretionaryCheckService per call — repo/llm are TimedCache-cached
    per-process on the underlying config, so this is cheap after the first call;
    service.tokens starts empty each time, which is exactly what a per-tool-call
    trace record wants (this call's token cost, not a running session total)."""
    bulk_request = await SessionStore.get_request(session_id)
    request = bulk_request.to_operator_request(operator)
    repo, llm = await _resolve_deps(request)
    return DiscretionaryCheckService(repo=repo, llm=llm, request=request)
