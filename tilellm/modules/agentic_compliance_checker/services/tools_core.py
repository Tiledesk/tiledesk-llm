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
from tilellm.modules.agentic_compliance_checker.services.deps import _resolve_deps
from tilellm.modules.agentic_compliance_checker.services import runner
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.compliance_checker.logic import _rerank_chunks

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

    Mirrors DiscretionaryCheckService._evaluate_criterion_once's retrieval
    exactly (oversample-then-rerank, exclude_chiarimenti filter) rather than
    calling it, because that method always retrieves internally — there is
    no seam to reuse for a standalone "just fetch, don't judge" call. Same
    small retrieval-building pattern this codebase already repeats in three
    other places (v1 check_compliance, v2 _evaluate_criterion_once, v2
    _fetch_capitolato_evidence); not new duplication, the existing idiom.
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
        search_text = criterion.text
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
        qa._metadata_filter = {"doc_type": {"$ne": "chiarimento"}}

    try:
        retrieval = await repo.get_chunks_from_repo(qa)
        chunks = retrieval.chunks or []
        metadata = retrieval.metadata or []
    except Exception as e:
        logger.warning("Retrieval failed for '%s': %s", search_text, e)
        chunks, metadata = [], []

    reranked = False
    if reranker_config and chunks:
        try:
            chunks, metadata = await _rerank_chunks(search_text, chunks, metadata, reranker_config, effective_top_k)
            reranked = True
        except Exception as e:
            logger.warning("Reranking failed for '%s': %s — proceeding without", search_text, e)

    evidence_ref = f"ev-{secrets.token_hex(8)}"
    entry = EvidenceEntry(
        evidence_ref=evidence_ref, namespace=request.namespace, criterion_id=criterion_id,
        query=search_text, query_kind=query_kind, chunks=chunks, metadata=metadata,
    )
    await SessionStore.store_evidence(session_id, entry)

    record_trace_detail(retrieval={
        "namespace": request.namespace, "query_used": search_text, "query_kind": query_kind,
        "top_k": effective_top_k, "reranked": reranked, "evidence_ref": evidence_ref,
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
    )
    return json.dumps({"results": results_out}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# compliance_build_report (P3, minimal — single operator; multi-operator
# aggregation across proportional criteria is P5)
# ---------------------------------------------------------------------------

@traced_tool("compliance_build_report")
async def build_report_core(*, session_id: str, operator: Optional[str] = None) -> str:
    """Builds a report from ONLY the results already stored in the session —
    this function accepts no score/judgment of any kind, by design (see
    evaluate_criteria_core's docstring). The summary is recomputed from
    stored state every time, never cached, so it can never drift from what
    the trace actually shows happened.
    """
    from tilellm.modules.compliance_checker.models_v2 import ComplianceSummaryV2

    operator_ref = await runner.resolve_operator(session_id, operator)
    lot = await SessionStore.get_lot(session_id)
    disc_results = await SessionStore.get_results(session_id, namespace=operator_ref.namespace)

    evaluated_ids = {r.criterion_id for r in disc_results}
    unevaluated = [c.id for c in lot.requirements.discretionary if c.id not in evaluated_ids]

    summary = ComplianceSummaryV2.from_results(tabular_results=[], disc_results=disc_results)

    return json.dumps({
        "operator": operator_ref.operator_label or operator_ref.namespace,
        "tender": {"lot_id": lot.tender.lot_id, "lot_name": lot.tender.lot_name},
        "summary": json.loads(summary.model_dump_json()),
        "unevaluated_criteria": unevaluated,
        "human_review_count": summary.human_review_count,
    }, ensure_ascii=False)
