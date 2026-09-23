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
import time
from typing import Any, Callable, Dict, Optional

from tilellm.modules.agentic_compliance_checker.models import TraceRecord
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore

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

    Deliberately the only "read" tool that never touches results/evidence —
    a caller reaches for this first to learn the criterion ids every other
    tool needs, and again later to see what's still pending. `status`
    filtering only has "all" to filter against until P3 starts writing
    results: every criterion is "pending" until compliance_evaluate_criteria
    exists to move it to "done"/"human_review".
    """
    lot = await SessionStore.get_lot(session_id)
    request = await SessionStore.get_request(session_id)

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
            rows.append({
                "id": c.id,
                "kind": "discretionary",
                "text": c.text[:200],
                "mode": c.mode.value,
                "max_points": c.max_points,
                "human_only": c.human_only,
                "direction": c.direction.value,
                "status": "pending",
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
