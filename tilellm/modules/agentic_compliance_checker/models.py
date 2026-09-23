"""
Pydantic models for the agentic compliance session: config, trace record, exceptions.

The HTTP request body to open a session IS compliance_checker's own
BulkComplianceRequestV2 (re-exported here) — a single-operator agentic run is
just `operators=[OperatorRef(namespace=...)]`. No parallel request model: the
one compliance_checker already validates is the right shape.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Re-exported so callers/tests can import everything they need from this module
# without reaching into compliance_checker directly.
from tilellm.modules.compliance_checker.models_v2 import (  # noqa: F401
    BulkComplianceRequestV2,
    OperatorRef,
    TenderInfo,
)


class SessionNotFound(Exception):
    """Raised when a session_id is unknown or its TTL has expired."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Sessione '{session_id}' non trovata (scaduta o mai creata).")


class TraceIncompleteError(Exception):
    """Raised by build_report when a stored result has no matching, digest-verified
    trace entry — the report-implies-trace guarantee this module exists to provide."""

    def __init__(self, session_id: str, orphans: List[str]):
        self.session_id = session_id
        self.orphans = orphans
        super().__init__(
            f"Sessione '{session_id}': {len(orphans)} risultati senza traccia di audit "
            f"corrispondente: {orphans}"
        )


class TraceRecord(BaseModel):
    """One record per tool call, appended (RPUSH) to acc:sess:{sid}:trace.

    Only `seq`/`ts`/`tool`/`session_id`/`args`/`outcome`/`duration_ms`/`error`
    are populated by @traced_tool in every phase; `llm`/`retrieval`/`judge`/
    `guardrails`/`result_digest` are filled in by the tools introduced from P3
    onward via the contextvar the decorator drains (see services/tools_core.py).
    """
    seq: int
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool: str
    session_id: str
    operator_namespace: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)
    outcome: str = "ok"  # "ok" | "error"
    error: Optional[str] = None
    duration_ms: int = 0
    llm: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None
    judge: Optional[Dict[str, Any]] = None
    guardrails: List[str] = Field(default_factory=list)
    result_digest: Optional[str] = None
    attempt: Optional[int] = None


class SessionOpenResponse(BaseModel):
    session_id: str
    expires_at: str
    tender: TenderInfo
    tabular_count: int
    discretionary_count: int
    operators: List[str]
    tools: List[str]


class SessionStatusResponse(BaseModel):
    session_id: str
    tender: TenderInfo
    operators: List[str]
    created_at: str
