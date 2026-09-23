"""
Session lifecycle: open/status/close. The DI seam (_resolve_deps) moved to
services/deps.py to avoid a circular import — see that file's docstring.

P7 additions: build_full_report and render_trace_markdown back the two new
HTTP-only report views (GET .../report, GET .../report/xlsx via controllers.py,
GET .../trace?format=md) — the human/programmatic "give me everything" export,
distinct from compliance_build_report (tools_core.py), which stays a compact,
agent-context-sized summary. Both call the SAME
services/audit_archive.py::verify_trace_completeness, so the
report-implies-trace guarantee holds identically on either surface.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from tilellm.modules.agentic_compliance_checker.models import (
    ComplianceReportV2,
    SessionNotFound,
    SessionOpenResponse,
    SessionStatusResponse,
    TraceRecord,
)
from tilellm.modules.agentic_compliance_checker.services.audit_archive import (
    verify_trace_completeness,
)
from tilellm.modules.agentic_compliance_checker.services.session_store import (
    SESSION_TTL_SECONDS,
    SessionStore,
)
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    ComplianceSummaryV2,
)
from tilellm.modules.compliance_checker.services.yaml_requirements_loader import (
    YamlRequirementsLoader,
)
from tilellm.modules.agentic_compliance_checker.services.langchain_tools import (
    AGENTIC_COMPLIANCE_TOOLS,
)

logger = logging.getLogger(__name__)


async def open_session(request: BulkComplianceRequestV2) -> SessionOpenResponse:
    lot = await YamlRequirementsLoader().load(
        yaml_inline=request.requirements_yaml,
        yaml_url=request.requirements_yaml_url,
        xlsx_url=request.requirements_xlsx_url,
        lot_id=request.requirements_lot_id,
    )
    session_id = await SessionStore.create(request, lot)
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)).isoformat()
    return SessionOpenResponse(
        session_id=session_id,
        expires_at=expires_at,
        tender=lot.tender,
        tabular_count=len(lot.requirements.tabular),
        discretionary_count=len(lot.requirements.discretionary),
        operators=[op.operator_label or op.namespace for op in request.operators],
        tools=sorted(AGENTIC_COMPLIANCE_TOOLS.keys()),
    )


async def get_session_status(session_id: str) -> SessionStatusResponse:
    meta = await SessionStore.get_meta(session_id)
    lot = await SessionStore.get_lot(session_id)
    request = await SessionStore.get_request(session_id)
    return SessionStatusResponse(
        session_id=session_id,
        tender=lot.tender,
        operators=[op.operator_label or op.namespace for op in request.operators],
        created_at=meta.get("created_at", ""),
    )


async def close_session(session_id: str) -> List[TraceRecord]:
    if not await SessionStore.exists(session_id):
        raise SessionNotFound(session_id)
    return await SessionStore.delete(session_id)


async def build_full_report(session_id: str, operator: Optional[str]) -> ComplianceReportV2:
    """The full ComplianceReportV2 for one operator — same shape /v2/check
    returns — built from stored session state (never re-evaluates anything).
    Raises TraceIncompleteError (via verify_trace_completeness) if any
    stored result's digest is missing from this session's trace history.
    """
    from tilellm.modules.agentic_compliance_checker.services import runner

    operator_ref = await runner.resolve_operator(session_id, operator)
    lot = await SessionStore.get_lot(session_id)
    disc_results = await SessionStore.get_results(session_id, namespace=operator_ref.namespace)
    tab_results = await SessionStore.get_tabular_results(session_id, namespace=operator_ref.namespace)
    l01_result = await SessionStore.get_l01_result(session_id, operator_ref.namespace)
    trace = await SessionStore.get_trace(session_id)
    verify_trace_completeness(session_id, trace, disc_results, tab_results, l01_result)

    summary = ComplianceSummaryV2.from_results(tabular_results=tab_results, disc_results=disc_results)
    return ComplianceReportV2(
        tender=lot.tender,
        namespace=operator_ref.namespace,
        summary=summary,
        tabular_results=tab_results,
        discretionary_results=disc_results,
        l01_check=l01_result,
    )


def render_trace_markdown(trace: List[TraceRecord]) -> str:
    """Human-readable rendering for GET .../trace?format=md — one row per
    tool call, in order, so an auditor can read the whole session's history
    without parsing JSON."""
    lines = [
        "# Traccia di audit",
        "",
        "| seq | timestamp | tool | esito | durata (ms) | errore |",
        "|---|---|---|---|---|---|",
    ]
    for r in trace:
        error_cell = (r.error or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r.seq} | {r.ts} | {r.tool} | {r.outcome} | {r.duration_ms} | {error_cell} |")
    return "\n".join(lines)
