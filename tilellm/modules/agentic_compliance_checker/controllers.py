"""
Agentic Compliance Checker — FastAPI router.

Session lifecycle (P1): opening a session is how a caller hands the agent a
session_id without ever putting gptkey/YAML into the LLM's context. The seven
tools themselves are not exposed as regular HTTP endpoints here — they reach
callers on two adapter surfaces built over the same tools_core.py functions:
the internal LangChain registry (registered into tools_registry.TOOL_REGISTRY
below, reached via POST /api/ask's `tools=[...]` selection) and the MCP
server mounted at the bottom of this file (P6, for an external MCP client).
"""
import contextlib
import logging
from typing import AsyncIterator, List

from fastapi import APIRouter, HTTPException
from starlette.routing import Route

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
from tilellm.modules.agentic_compliance_checker.services.langchain_tools import (
    AGENTIC_COMPLIANCE_TOOLS,
)
from tilellm.modules.agentic_compliance_checker.services.mcp_server import transport
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.tools_registry.services.tool_registry import TOOL_REGISTRY

logger = logging.getLogger(__name__)

_MCP_PATH = "/api/agentic-compliance/mcp"


@contextlib.asynccontextmanager
async def _mcp_lifespan(_router: APIRouter) -> AsyncIterator[None]:
    """Starts/stops the MCP transport together with the app. FastAPI merges
    an included APIRouter's `lifespan` into the app's own lifespan on
    `include_router()` (see `_merge_lifespan_context` in fastapi.routing —
    confirmed by reading it, not assumed), so this runs without adding a
    fourth edit to tilellm/__main__.py: the module stays self-contained the
    same way P1-P5's TOOL_REGISTRY.update() below does. Without this, any
    request to _MCP_PATH raises "MCP transport is not active" (see
    services/mcp_server.py._MCPTransport) — verified by omitting it once.
    """
    async with transport.activate():
        yield


router = APIRouter(
    prefix="/api/agentic-compliance", tags=["Agentic Compliance Checker"], lifespan=_mcp_lifespan,
)

# Registered here (not in tools_registry) so the module stays self-contained:
# TOOL_REGISTRY is a plain module-level dict read at call time by both
# resolve_tools() and get_available_tools_list(), so this needs no edit to
# tools_registry itself — the tools just appear on GET /api/tools and become
# selectable via QuestionToLLM.tools on /api/ask and /api/thinking.
TOOL_REGISTRY.update({
    name: {"description": tool_obj.description, "implementation": tool_obj, "is_factory": False}
    for name, tool_obj in AGENTIC_COMPLIANCE_TOOLS.items()
})

# MCP server (P6): a plain Starlette Route wrapping `transport` (a stable
# dispatcher, see services/mcp_server.py — NOT the FastMCP-produced ASGI app
# directly), added straight to the router's route list — NOT
# `router.mount(_MCP_PATH, ...)`. That would look right and silently do
# nothing: FastAPI's APIRouter.include_router() only forwards APIRoute/
# Route/WebSocketRoute entries (see its source), so a Mount added via
# `router.mount()` never reaches the app's route table — no error, no route,
# no 404-with-explanation, which is exactly why this is spelled out instead
# of just `router.mount(...)`. Explicit `methods=` is required too: FastMCP
# builds its own internal route with `methods=None` (meaning "any method",
# Starlette's raw-ASGI-endpoint convention), and include_router's
# `methods = list(route.methods or [])` turns that into an empty list — a
# route nobody can ever call. The path must be the FULL final path here, not
# just "/mcp": routes appended directly to `router.routes` (bypassing
# router.get()/.post()/etc.) do not get `router.prefix` applied by
# include_router — only its own `prefix=` KEYWORD ARGUMENT does, which
# nothing passes when tilellm/__main__.py calls `_app.include_router(module.router)`.
router.routes.append(Route(_MCP_PATH, endpoint=transport, methods=["GET", "POST", "DELETE"]))


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
