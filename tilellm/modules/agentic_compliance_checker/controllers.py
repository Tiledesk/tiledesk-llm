"""
Agentic Compliance Checker — FastAPI router.

Session lifecycle only in this phase (P1): opening a session is how a caller
hands the agent a session_id without ever putting gptkey/YAML into the LLM's
context. Tool endpoints are not exposed over HTTP directly — they are
registered into tools_registry.TOOL_REGISTRY (P2 onward) and reached via
POST /api/ask's `tools=[...]` selection, or the MCP server (P6).
"""
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from tilellm.modules.agentic_compliance_checker.logic import (
    close_session,
    get_session_status,
    open_session,
)
from tilellm.modules.agentic_compliance_checker.models import (
    BulkComplianceRequestV2,
    SessionNotFound,
    SessionOpenResponse,
    SessionStatusResponse,
    TraceRecord,
)
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agentic-compliance", tags=["Agentic Compliance Checker"])


@router.post("/sessions", response_model=SessionOpenResponse)
async def post_open_session(request: BulkComplianceRequestV2):
    try:
        return await open_session(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session(session_id: str):
    try:
        return await get_session_status(session_id)
    except SessionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=List[TraceRecord])
async def delete_session(session_id: str):
    try:
        return await close_session(session_id)
    except SessionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sessions/{session_id}/trace", response_model=List[TraceRecord])
async def get_trace(session_id: str):
    if not await SessionStore.exists(session_id):
        raise HTTPException(status_code=404, detail=f"Sessione '{session_id}' non trovata.")
    return await SessionStore.get_trace(session_id)
