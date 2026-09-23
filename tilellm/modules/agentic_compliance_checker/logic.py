"""
Session lifecycle: open/status/close. The DI seam (_resolve_deps) moved to
services/deps.py to avoid a circular import — see that file's docstring.
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
from tilellm.modules.compliance_checker.models_v2 import BulkComplianceRequestV2
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
