"""
Audit primitives (P7): the "report implies trace" guarantee, and durable
archival of a finalized report past Redis's TTL.

Two independent things live here on purpose:

1. Trace-completeness verification (compute_result_digest /
   verify_trace_completeness) — pure, no I/O, used by BOTH
   tools_core.build_report_core (the agent-facing tool) and logic.py's
   build_full_report (the HTTP /report endpoint), so the guarantee holds no
   matter which surface asks for a report.
2. MinIO archival (archive_report) — I/O, best-effort, used only by the HTTP
   endpoint. The agent-facing tool can be called many times in a session (an
   agent checking progress) and must stay cheap; a human calling
   GET /sessions/{id}/report is the actual "finalize this" moment that's
   worth writing an immutable copy for.
"""
import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel

from tilellm.modules.agentic_compliance_checker.models import TraceIncompleteError, TraceRecord
from tilellm.modules.compliance_checker.models_v2 import (
    ComplianceResult,
    DiscretionaryResult,
    L01CheckResult,
)

logger = logging.getLogger(__name__)

_AUDIT_BUCKET = os.environ.get("AGENTIC_COMPLIANCE_AUDIT_BUCKET", "agentic-compliance-audit")


def compute_result_digest(result: BaseModel) -> str:
    """sha256 of the result's own canonical JSON — recomputed identically at
    write time (tools_core, to record in the trace) and at read time
    (verify_trace_completeness, to check a currently-stored result against
    trace history). Any manual edit to a stored result in Redis changes this
    digest, which is exactly the tamper this function exists to catch."""
    return hashlib.sha256(result.model_dump_json().encode("utf-8")).hexdigest()


def verify_trace_completeness(
    session_id: str,
    trace: List[TraceRecord],
    disc_results: List[DiscretionaryResult],
    tab_results: List[ComplianceResult],
    l01_result: Optional[L01CheckResult],
) -> None:
    """Every currently-stored result must have been produced by a traced
    tool call at some point in this session's history — its digest must
    appear in SOME record's `result_digests`, not necessarily the most
    recent one (a result can survive unchanged across an unrelated later
    call). A result stored by anything else (a manual Redis edit, a bug that
    bypasses @traced_tool) has no such record and fails this check.
    Raises TraceIncompleteError listing every orphan found, not just the
    first — a caller fixing this wants the whole list at once.
    """
    known_digests = {d for record in trace for d in record.result_digests}

    orphans: List[str] = []
    for r in disc_results:
        if compute_result_digest(r) not in known_digests:
            orphans.append(f"disc:{r.criterion_id}")
    for r in tab_results:
        if compute_result_digest(r) not in known_digests:
            orphans.append(f"tab:{r.requirement_id}")
    if l01_result is not None and l01_result.used and compute_result_digest(l01_result) not in known_digests:
        orphans.append("l01")

    if orphans:
        raise TraceIncompleteError(session_id, orphans)


async def archive_report(session_id: str, namespace: str, report_json: str, trace: List[TraceRecord]) -> Optional[str]:
    """Writes {report, trace} as one immutable JSON object to MinIO, keyed by
    session/operator/timestamp (never overwritten — a later archive of the
    same session+operator is a NEW object). Returns its s3://bucket/key URI,
    or None if MinIO is unavailable/unconfigured — archival failing must
    never fail the report itself (Redis is still the authoritative working
    state; this is best-effort durability past its TTL), so every failure
    here is caught and logged, not raised.
    """
    try:
        from tilellm.shared.minio_storage import get_minio_storage_service

        service = get_minio_storage_service()
    except (ImportError, ValueError) as e:
        logger.warning("Agentic compliance audit archive skipped (MinIO not available): %s", e)
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    object_name = f"agentic-compliance/{session_id}/{namespace}/{timestamp}.json"
    payload = json.dumps({
        "session_id": session_id,
        "namespace": namespace,
        "archived_at": timestamp,
        "report": json.loads(report_json),
        "trace": [r.model_dump(mode="json") for r in trace],
    }).encode("utf-8")

    try:
        # MinIOStorageService's client is the synchronous `minio` SDK — off
        # the event loop, same boundary-crossing pattern pdf_ocr uses for
        # its own sync I/O calls.
        await asyncio.to_thread(
            service.upload_data, _AUDIT_BUCKET, object_name, payload, "application/json",
        )
    except Exception as e:
        logger.warning("Agentic compliance audit archive failed for session '%s': %s", session_id, e)
        return None

    return f"s3://{_AUDIT_BUCKET}/{object_name}"
