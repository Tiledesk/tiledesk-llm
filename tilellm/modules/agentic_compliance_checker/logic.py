"""
Session lifecycle (open/status/close) + the DI seam used by every tool from P3
onward to obtain a repo/llm pair for a given operator's ComplianceRequestV2.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List

from tilellm.modules.agentic_compliance_checker.models import (
    SessionNotFound,
    SessionOpenResponse,
    SessionStatusResponse,
    TraceRecord,
)
from tilellm.modules.agentic_compliance_checker.services.session_store import (
    SESSION_TTL_SECONDS,
    SessionStore,
)
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    ComplianceRequestV2,
)
from tilellm.modules.compliance_checker.services.yaml_requirements_loader import (
    YamlRequirementsLoader,
)
from tilellm.modules.agentic_compliance_checker.services.langchain_tools import (
    AGENTIC_COMPLIANCE_TOOLS,
)
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async

logger = logging.getLogger(__name__)


@inject_llm_chat_async
@inject_repo_async
async def _resolve_deps(request: ComplianceRequestV2, repo=None, llm=None, **kwargs):
    """The DI seam every tool uses to get a (repo, llm) pair for one operator.

    repo/llm are never stored in the session (unpicklable, and already
    TimedCache-cached per-process on the same config fields) — they are
    re-derived from the stored config on every tool call; after the first call
    per worker process this is a cache hit, not a fresh construction.
    """
    return repo, llm


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
