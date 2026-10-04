"""
The _core coroutines here are the ONE implementation of each tool; both
adapters (services/langchain_tools.py, services/mcp_server.py from P6) call
them and add nothing of their own beyond translating argument/return shapes.

@traced_tool is what makes "a report implies a trace" hold: every write to
session state happens inside a traced call, so build_report (P7) can verify
every stored result has a matching trace entry. No tool-specific
instrumentation — one decorator, reused by all seven.
"""
import contextvars
import functools
import json
import logging
import secrets
import time
from typing import Any, Callable, Dict, List, Optional

from tilellm.models import QuestionAnswer
from tilellm.modules.agentic_compliance_checker.models import EvidenceEntry, TraceRecord
from tilellm.modules.agentic_compliance_checker.services.audit_archive import (
    compute_result_digest,
    verify_trace_completeness,
)
from tilellm.modules.agentic_compliance_checker.services.deps import _resolve_deps
from tilellm.modules.agentic_compliance_checker.services import runner
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.compliance_checker.logic import EXCLUDE_CHIARIMENTI_FILTER, _retrieve_evidence

logger = logging.getLogger(__name__)

# Populated by a tool core's own retrieval/judge/guardrail logic (from P3
# onward) via record_trace_detail(), drained by @traced_tool when the call
# finishes. Keeps the decorator generic instead of threading a recorder
# object through every tool signature.
_trace_extra: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "_trace_extra", default=None
)


def record_trace_detail(**fields: Any) -> None:
    """Call from inside a @traced_tool-decorated core to attach llm/retrieval/
    judge/guardrails detail to the TraceRecord it will append. No-op outside
    a traced call (e.g. a core invoked directly in a unit test)."""
    current = _trace_extra.get()
    if current is not None:
        current.update(fields)


def traced_tool(name: str) -> Callable:
    """Every tool core takes session_id as a required keyword argument —
    enforced here, not inferred positionally, so tracing can never silently
    attach to the wrong session."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if "session_id" not in kwargs:
                raise TypeError(f"{name}: session_id must be passed as a keyword argument")
            session_id = kwargs["session_id"]
            started = time.monotonic()
            token = _trace_extra.set({})
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                # A trace-append failure while handling a real business error
                # (e.g. the session expired mid-call) must not mask that
                # error — log it and let the original exception propagate.
                try:
                    await _append(name, session_id, kwargs, "error", started, error=str(e))
                except Exception:
                    logger.warning(
                        "traced_tool '%s': could not append error trace for session '%s'",
                        name, session_id,
                    )
                raise
            else:
                # On the success path a failed trace-append IS a real
                # failure: a tool call that "succeeds" for the agent without
                # a matching trace record is exactly the audit gap this
                # decorator exists to prevent, so this is NOT swallowed —
                # build_report's (P7) trace-completeness check is a second
                # line of defense, not a substitute for this one.
                extra = _trace_extra.get() or {}
                await _append(name, session_id, kwargs, "ok", started, **extra)
                return result
            finally:
                _trace_extra.reset(token)

        return wrapper

    return decorator


async def _append(
    name: str, session_id: str, kwargs: Dict[str, Any], outcome: str, started: float,
    error: Optional[str] = None, **extra: Any,
) -> None:
    duration_ms = int((time.monotonic() - started) * 1000)
    redacted_args = {k: v for k, v in kwargs.items() if k != "session_id"}
    record = TraceRecord(
        seq=0,  # overwritten by SessionStore.append_trace via Redis INCR (race-free)
        tool=name,
        session_id=session_id,
        args=redacted_args,
        outcome=outcome,
        error=error,
        duration_ms=duration_ms,
        **extra,
    )
    await SessionStore.append_trace(session_id, record)


# ---------------------------------------------------------------------------
# compliance_list_requirements (P2)
# ---------------------------------------------------------------------------

@traced_tool("compliance_list_requirements")
async def list_requirements_core(
    *, session_id: str, kind: str = "all", status: str = "all", operator: Optional[str] = None,
) -> str:
    """Lists a session's tabular/discretionary requirements with their status.

    Deliberately the only "read" tool that never touches results/evidence
    directly — a caller reaches for this first to learn the criterion ids
    every other tool needs, and again later to see what's still pending.

    Status reflects stored discretionary results for ONE resolvable operator
    (given explicitly, or the session's only one) — "pending" (never
    evaluated), "human_review" (evaluated, flagged), "done" (evaluated,
    scored). With multiple operators and none specified, every criterion
    shows "pending" (there is no single operator's status to show) — tabular
    requirements always show "pending" until compliance_check_tabular (P4)
    exists.
    """
    lot = await SessionStore.get_lot(session_id)
    request = await SessionStore.get_request(session_id)

    stored_by_criterion: Dict[str, Any] = {}
    try:
        resolved_operator = await runner.resolve_operator(session_id, operator)
    except ValueError:
        resolved_operator = None  # ambiguous (multi-operator, none given) — every row stays "pending"
    if resolved_operator is not None:
        for result in await SessionStore.get_results(session_id, namespace=resolved_operator.namespace):
            stored_by_criterion[result.criterion_id] = result

    rows = []
    if kind in ("all", "tabular"):
        for r in lot.requirements.tabular:
            rows.append({
                "id": r.id,
                "kind": "tabular",
                "text": r.text[:200],
                "mandatory": r.mandatory,
                "status": "pending",
            })
    if kind in ("all", "discretionary"):
        for c in lot.requirements.discretionary:
            result = stored_by_criterion.get(c.id)
            if result is None:
                crit_status = "pending"
            elif result.human_review_required:
                crit_status = "human_review"
            else:
                crit_status = "done"
            rows.append({
                "id": c.id,
                "kind": "discretionary",
                "text": c.text[:200],
                "mode": c.mode.value,
                "max_points": c.max_points,
                "human_only": c.human_only,
                "direction": c.direction.value,
                "status": crit_status,
            })
    if status != "all":
        rows = [r for r in rows if r["status"] == status]

    return json.dumps({
        "tender": {"lot_id": lot.tender.lot_id, "lot_name": lot.tender.lot_name},
        "operators": [op.operator_label or op.namespace for op in request.operators],
        "totals": {
            "tabular": len(lot.requirements.tabular),
            "discretionary": len(lot.requirements.discretionary),
        },
        "requirements": rows,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# compliance_retrieve_evidence (P3)
# ---------------------------------------------------------------------------

@traced_tool("compliance_retrieve_evidence")
async def retrieve_evidence_core(
    *, session_id: str, criterion_id: Optional[str] = None, query: Optional[str] = None,
    operator: Optional[str] = None, top_k: Optional[int] = None, include_chiarimenti: bool = False,
) -> str:
    """Retrieves evidence for a criterion (or a free-form query) from an
    operator's namespace, caches it under an evidence_ref, and returns a
    compact preview + that ref — never the full chunk text, which stays in
    the session (and, verbatim, in the trace record for this call).

    Same evidence as DiscretionaryCheckService._evaluate_criterion_once would
    judge: identical query building (oversample, exclude_chiarimenti filter)
    and the shared compliance_checker.logic._retrieve_evidence pipeline
    (retrieve -> rerank -> re-attach split neighbours).
    """
    if not criterion_id and not query:
        raise ValueError("Fornire 'criterion_id' oppure 'query'.")
    if criterion_id and query:
        raise ValueError("'criterion_id' e 'query' sono mutuamente esclusivi.")

    operator_ref = await runner.resolve_operator(session_id, operator)
    bulk_request = await SessionStore.get_request(session_id)
    request = bulk_request.to_operator_request(operator_ref)

    if criterion_id:
        criterion = await runner.resolve_criterion(session_id, criterion_id)
        search_text = criterion.search_query or criterion.text
        query_kind = "criterion"
    else:
        search_text = query
        query_kind = "custom"

    repo, _llm = await _resolve_deps(request)

    reranker_config = request.reranker_config
    effective_top_k = top_k or request.top_k
    search_top_k = effective_top_k * request.reranking_multiplier if reranker_config else effective_top_k

    qa = QuestionAnswer(
        question=search_text,
        namespace=request.namespace,
        engine=request.engine,
        embedding=request.embedding,
        sparse_encoder=request.sparse_encoder,
        gptkey=request.gptkey,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_k=search_top_k,
        search_type=request.search_type,
    )
    if request.exclude_chiarimenti and not include_chiarimenti:
        qa._metadata_filter = EXCLUDE_CHIARIMENTI_FILTER

    chunks, metadata = await _retrieve_evidence(
        repo, qa, search_text, reranker_config, effective_top_k, f"'{search_text}'"
    )

    evidence_ref = f"ev-{secrets.token_hex(8)}"
    entry = EvidenceEntry(
        evidence_ref=evidence_ref, namespace=request.namespace, criterion_id=criterion_id,
        query=search_text, query_kind=query_kind, chunks=chunks, metadata=metadata,
    )
    await SessionStore.store_evidence(session_id, entry)

    record_trace_detail(retrieval={
        "namespace": request.namespace, "query_used": search_text, "query_kind": query_kind,
        "top_k": effective_top_k, "reranking": bool(reranker_config), "evidence_ref": evidence_ref,
        "chunk_count": len(chunks),
    })

    preview = [
        {
            "i": i,
            "document": meta.get("file_name", meta.get("source", "unknown")),
            "page": meta.get("page", "?"),
            "excerpt": chunk[:200],
        }
        for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1)
    ]
    return json.dumps({
        "evidence_ref": evidence_ref,
        "chunk_count": len(chunks),
        "namespace": request.namespace,
        "query_kind": query_kind,
        "preview": preview,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# compliance_evaluate_criteria (P3) — the core tool
# ---------------------------------------------------------------------------

@traced_tool("compliance_evaluate_criteria")
async def evaluate_criteria_core(
    *, session_id: str, criterion_ids: List[str], operators: Optional[List[str]] = None,
    evidence_ref: Optional[str] = None, retrieval_query: Optional[str] = None,
    reason: Optional[str] = None,
) -> str:
    """Evaluates one or more discretionary criteria for one or more operators.

    Delegates every guardrail/scoring/HyDE decision to
    DiscretionaryCheckService — this function's only job is choosing WHICH of
    its three entry points to call:
      - evidence_ref given: _evaluate_criterion_once(pre_fetched=...) — judge
        specific already-retrieved evidence, no fresh retrieval, no HyDE.
        Only valid for exactly one criterion and one operator (the ref itself
        is namespace- and (usually) criterion-scoped).
      - retrieval_query given: _evaluate_criterion_once(retrieval_query=...) —
        one retrieval with a caller-supplied reformulation, no automatic HyDE
        (the caller already reformulated).
      - neither given: _evaluate_criterion(...) — the full default path,
        including the automatic HyDE fallback.
    No tool argument here can carry a score, coefficient or judgment — see
    tests/.../test_tools_core.py::test_no_tool_exposes_a_verdict_field.
    """
    if evidence_ref is not None and (len(criterion_ids) != 1 or (operators and len(operators) != 1)):
        raise ValueError("'evidence_ref' richiede esattamente un criterio e un operatore.")

    bulk_request = await SessionStore.get_request(session_id)
    target_operators = (
        [await runner.resolve_operator(session_id, o) for o in operators]
        if operators else bulk_request.operators
    )

    results_out = []
    guardrails_seen = set()
    token_total = {"prompt": 0, "completion": 0, "total": 0}
    digests: List[str] = []

    for operator_ref in target_operators:
        service = await runner.build_service_for_operator(session_id, operator_ref)
        for criterion_id in criterion_ids:
            criterion = await runner.resolve_criterion(session_id, criterion_id)

            if evidence_ref is not None:
                entry = await SessionStore.get_evidence(session_id, evidence_ref)
                result = await service._evaluate_criterion_once(
                    criterion, pre_fetched=(entry.chunks, entry.metadata),
                )
                if entry.criterion_id and entry.criterion_id != criterion_id:
                    note = (
                        "Evidenza fornita manualmente dall'agente (evidence_ref di un "
                        "altro criterio), non recuperata per questo criterio."
                    )
                    result.human_review_required = True
                    result.human_review_reason = (
                        f"{result.human_review_reason} {note}" if result.human_review_reason else note
                    )
                    guardrails_seen.add("evidence_override")
            elif retrieval_query is not None:
                result = await service._evaluate_criterion_once(criterion, retrieval_query=retrieval_query)
            else:
                result = await service._evaluate_criterion(criterion)

            await SessionStore.store_result(session_id, operator_ref.namespace, criterion_id, result)
            digests.append(compute_result_digest(result))
            attempt = await SessionStore.increment_attempts(session_id, operator_ref.namespace, criterion_id)

            if result.human_review_required:
                guardrails_seen.add("human_review")
            if getattr(result, "hyde_used", False):
                guardrails_seen.add("hyde")
            if criterion.human_only:
                guardrails_seen.add("human_only")

            results_out.append({
                "criterion_id": criterion_id,
                "operator": operator_ref.operator_label or operator_ref.namespace,
                "coefficient": result.coefficient,
                "score": result.score,
                "measured_value": result.measured_value,
                "measured_quantity": result.measured_quantity,
                "confidence": result.confidence,
                "human_review_required": result.human_review_required,
                "human_review_reason": result.human_review_reason,
                "citation_attributed": result.citation_attributed,
                "evidence_document": result.evidence_document,
                "evidence_page": result.evidence_page,
                "attempt": attempt,
            })

        totals = service.tokens.total()
        for k in token_total:
            token_total[k] += totals.get(k, 0)

    record_trace_detail(
        llm=token_total,
        judge={"criterion_count": len(criterion_ids), "operator_count": len(target_operators), "reason": reason},
        guardrails=sorted(guardrails_seen),
        result_digests=digests,
    )
    return json.dumps({"results": results_out}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# compliance_check_tabular (P4)
# ---------------------------------------------------------------------------

@traced_tool("compliance_check_tabular")
async def check_tabular_core(
    *, session_id: str, requirement_ids: Optional[List[str]] = None,
    operators: Optional[List[str]] = None,
) -> str:
    """Evaluates tabular (presence/absence) requirements for one or more
    operators via DiscretionaryCheckService._check_tabular — the same path
    /v2/check uses, which itself delegates to v1 check_compliance. No
    separate judgment logic here; this function only selects WHICH
    requirements/operators, same division of responsibility as
    evaluate_criteria_core.
    """
    bulk_request = await SessionStore.get_request(session_id)
    target_operators = (
        [await runner.resolve_operator(session_id, o) for o in operators]
        if operators else bulk_request.operators
    )
    lot = await SessionStore.get_lot(session_id)

    if requirement_ids is not None:
        wanted = set(requirement_ids)
        for rid in wanted:
            await runner.resolve_tabular_requirement(session_id, rid)  # raises if unknown
        filtered_tabular = [r for r in lot.requirements.tabular if r.id in wanted]
    else:
        filtered_tabular = lot.requirements.tabular
    filtered_lot = lot.model_copy(update={
        "requirements": lot.requirements.model_copy(update={
            "tabular": filtered_tabular, "discretionary": [],
        }),
    })

    results_out = []
    token_total = {"prompt": 0, "completion": 0, "total": 0}
    digests: List[str] = []
    for operator_ref in target_operators:
        service = await runner.build_service_for_operator(session_id, operator_ref)
        results = await service._check_tabular(filtered_lot)
        for result in results:
            await SessionStore.store_tabular_result(
                session_id, operator_ref.namespace, result.requirement_id, result,
            )
            digests.append(compute_result_digest(result))
            results_out.append({
                "requirement_id": result.requirement_id,
                "operator": operator_ref.operator_label or operator_ref.namespace,
                "judgment": result.judgment,
                "confidence": result.confidence,
                "mandatory": result.mandatory,
                "evidence_document": result.evidence_document,
                "evidence_page": result.evidence_page,
            })
        totals = service.tokens.total()
        for k in token_total:
            token_total[k] += totals.get(k, 0)

    record_trace_detail(
        llm=token_total,
        judge={"requirement_count": len(filtered_tabular), "operator_count": len(target_operators)},
        result_digests=digests,
    )
    return json.dumps({"results": results_out}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# compliance_check_l01 (P4) — pure listino<->PDF reconciliation, zero LLM
# ---------------------------------------------------------------------------

@traced_tool("compliance_check_l01")
async def check_l01_core(*, session_id: str, operators: Optional[List[str]] = None) -> str:
    """Reconciles each operator's structured L01 price list against the PDF
    technical sheets already indexed for them, via
    DiscretionaryCheckService._check_l01 — the same path /v2/check uses. Pure
    code/name matching, no LLM call; opt-in per operator (l01_xlsx_url on
    that operator's request — used=False, not an error, when absent).
    """
    bulk_request = await SessionStore.get_request(session_id)
    target_operators = (
        [await runner.resolve_operator(session_id, o) for o in operators]
        if operators else bulk_request.operators
    )

    results_out = []
    digests: List[str] = []
    for operator_ref in target_operators:
        service = await runner.build_service_for_operator(session_id, operator_ref)
        result = await service._check_l01()
        await SessionStore.store_l01_result(session_id, operator_ref.namespace, result)
        if result.used:
            digests.append(compute_result_digest(result))
        results_out.append({
            "operator": operator_ref.operator_label or operator_ref.namespace,
            "used": result.used,
            "l01_products_total": result.l01_products_total,
            "matched": result.matched,
            "missing_count": result.missing_count,
            "missing": result.missing,
        })

    record_trace_detail(judge={"operator_count": len(target_operators)}, result_digests=digests)
    return json.dumps({"results": results_out}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# compliance_resolve_proportional (P5)
# ---------------------------------------------------------------------------

@traced_tool("compliance_resolve_proportional")
async def resolve_proportional_core(
    *, session_id: str, criterion_ids: Optional[List[str]] = None, allow_partial: bool = False,
) -> str:
    """Resolves 'proporzionale' criteria across ALL operators of the session,
    via bulk_check_service.resolve_proportional — the same cross-operator
    formula /v2/check/bulk uses (qmax/qmin over measured_quantity). This
    function's only job is assembling the per-operator result set it runs on
    and re-storing the mutated results; it invents no scoring of its own.

    Precondition: every operator must already have a stored
    compliance_evaluate_criteria result for every targeted proportional
    criterion — qmax/qmin computed on a subset silently skews every score in
    the lot. Missing an operator/criterion combination raises ValueError
    unless allow_partial=True, which proceeds anyway and marks every result
    it touches with a note (and the trace with guardrail "partial").
    """
    from tilellm.modules.compliance_checker.models_v2 import (
        ComplianceReportV2,
        ComplianceSummaryV2,
        DiscretionaryMode,
    )
    from tilellm.modules.compliance_checker.services.bulk_check_service import (
        resolve_proportional,
    )

    bulk_request = await SessionStore.get_request(session_id)
    lot = await SessionStore.get_lot(session_id)
    all_operators = bulk_request.operators

    known_ids = {c.id for c in lot.requirements.discretionary}
    if criterion_ids is not None:
        for cid in criterion_ids:
            if cid not in known_ids:
                raise ValueError(f"Criterio discrezionale '{cid}' non trovato nel lotto.")

    proportional_ids = {
        c.id for c in lot.requirements.discretionary
        if c.mode == DiscretionaryMode.PROPORZIONALE and (criterion_ids is None or c.id in criterion_ids)
    }

    missing: Dict[str, List[str]] = {}
    per_operator_results = []
    for operator_ref in all_operators:
        stored = await SessionStore.get_results(session_id, namespace=operator_ref.namespace)
        by_id = {r.criterion_id: r for r in stored}
        op_missing = sorted(cid for cid in proportional_ids if cid not in by_id)
        if op_missing:
            missing[operator_ref.operator_label or operator_ref.namespace] = op_missing
        targeted = [by_id[cid] for cid in proportional_ids if cid in by_id]
        per_operator_results.append((operator_ref, targeted))

    if missing and not allow_partial:
        raise ValueError(
            "Impossibile risolvere il proporzionale: valutazioni mancanti per "
            f"{missing}. Valutarle con compliance_evaluate_criteria, oppure "
            "passare allow_partial=True per procedere comunque."
        )

    # quantity_from_l01 criteria (e.g. range breadth): the comparable quantity is the
    # operator's L01 product count, stored by compliance_check_l01 — same fallback
    # /v2/check applies (_apply_l01_quantity: a quantity the judge measured wins).
    l01_not_checked: List[str] = []
    if any(c.quantity_from_l01 for c in lot.requirements.discretionary if c.id in proportional_ids):
        from tilellm.modules.compliance_checker.services.discretionary_check_service import (
            _apply_l01_quantity,
        )
        for operator_ref, targeted in per_operator_results:
            l01_result = await SessionStore.get_l01_result(session_id, operator_ref.namespace)
            if l01_result is None:
                l01_not_checked.append(operator_ref.operator_label or operator_ref.namespace)
            _apply_l01_quantity(lot, targeted, l01_result)

    compliance_reports = [
        ComplianceReportV2(
            tender=lot.tender, namespace=operator_ref.namespace, summary=ComplianceSummaryV2(),
            tabular_results=[], discretionary_results=targeted,
        )
        for operator_ref, targeted in per_operator_results if targeted
    ]
    resolve_proportional(compliance_reports)  # mutates in place

    partial = bool(missing)
    results_out = []
    digests: List[str] = []
    for operator_ref, targeted in per_operator_results:
        for result in targeted:
            if partial:
                note = "Risoluzione proporzionale eseguita con operatori/criteri mancanti (allow_partial=True)."
                result.human_review_reason = (
                    f"{result.human_review_reason} {note}" if result.human_review_reason else note
                )
            await SessionStore.store_result(session_id, operator_ref.namespace, result.criterion_id, result)
            digests.append(compute_result_digest(result))
            results_out.append({
                "criterion_id": result.criterion_id,
                "operator": operator_ref.operator_label or operator_ref.namespace,
                "score": result.score,
                "proportional_auto": result.proportional_auto,
                "measured_quantity": result.measured_quantity,
                "direction": result.direction.value,
            })

    record_trace_detail(
        judge={"criterion_count": len(proportional_ids), "operator_count": len(all_operators)},
        guardrails=(["partial"] if partial else []),
        result_digests=digests,
    )
    return json.dumps({
        "results": results_out,
        "partial": partial,
        "missing": missing or None,
        # compliance_check_l01 never called for these operators: their L01 count
        # couldn't be used, so they may be unscored on quantity_from_l01 criteria.
        "l01_not_checked": l01_not_checked or None,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# compliance_build_report (P3 minimal, extended in P4 with tabular/L01;
# proportional resolution itself is a separate tool, compliance_resolve_
# proportional (P5) — this function still reports whatever score ended up
# stored, proportional or not, single operator at a time)
# ---------------------------------------------------------------------------

@traced_tool("compliance_build_report")
async def build_report_core(*, session_id: str, operator: Optional[str] = None) -> str:
    """Builds a report from ONLY the results already stored in the session —
    this function accepts no score/judgment of any kind, by design (see
    evaluate_criteria_core's docstring). The summary is recomputed from
    stored state every time, never cached, so it can never drift from what
    the trace actually shows happened.

    P7: before building anything, verifies every stored result has a
    matching digest somewhere in this session's trace (see
    services/audit_archive.py::verify_trace_completeness) — a result written
    by anything other than a @traced_tool call (a manual Redis edit, a bug)
    fails this and raises TraceIncompleteError instead of silently being
    reported as if it were legitimate.
    """
    from tilellm.modules.compliance_checker.models_v2 import ComplianceSummaryV2

    operator_ref = await runner.resolve_operator(session_id, operator)
    lot = await SessionStore.get_lot(session_id)
    disc_results = await SessionStore.get_results(session_id, namespace=operator_ref.namespace)
    tab_results = await SessionStore.get_tabular_results(session_id, namespace=operator_ref.namespace)
    l01_result = await SessionStore.get_l01_result(session_id, operator_ref.namespace)
    trace = await SessionStore.get_trace(session_id)
    verify_trace_completeness(session_id, trace, disc_results, tab_results, l01_result)

    evaluated_ids = {r.criterion_id for r in disc_results}
    unevaluated = [c.id for c in lot.requirements.discretionary if c.id not in evaluated_ids]
    evaluated_tab_ids = {r.requirement_id for r in tab_results}
    unevaluated_tabular = [r.id for r in lot.requirements.tabular if r.id not in evaluated_tab_ids]

    summary = ComplianceSummaryV2.from_results(tabular_results=tab_results, disc_results=disc_results)

    return json.dumps({
        "operator": operator_ref.operator_label or operator_ref.namespace,
        "tender": {"lot_id": lot.tender.lot_id, "lot_name": lot.tender.lot_name},
        "summary": json.loads(summary.model_dump_json()),
        "unevaluated_criteria": unevaluated,
        "unevaluated_tabular": unevaluated_tabular,
        "l01_check": json.loads(l01_result.model_dump_json()) if l01_result else None,
        "human_review_count": summary.human_review_count,
    }, ensure_ascii=False)
