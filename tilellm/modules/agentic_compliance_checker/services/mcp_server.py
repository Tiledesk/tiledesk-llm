"""
MCP server adapter (P6) — the same seven tools as services/langchain_tools.py,
exposed over the Model Context Protocol (streamable HTTP) for an external
agent (Claude Desktop, another orchestrator) that does not go through
POST /api/ask.

This file registers `AGENTIC_COMPLIANCE_TOOLS[name].coroutine` directly — the
exact plain async function LangChain's @tool decorator wraps, `_safe()` error
handling included — not a second implementation of the seven tools. Both
adapters therefore expose identical behaviour (arguments, error shape) by
construction, matching tools_core.py's one-implementation-per-tool principle.
Reusing `.coroutine` does mean per-argument descriptions come only from the
plain function's own docstring, not from the richer `Field(description=...)`
metadata on langchain_tools.py's `*Args` Pydantic classes (those only govern
the LangChain-side schema) — a real but minor documentation gap on the MCP
side, not a functional one; the top-level tool description (full docstring)
is identical either way.

Config choices, both deployment-driven:

- `stateless_http=True`: entrypoint.sh runs gunicorn with WORKERS=3 by
  default (the same constraint documented in session_store.py's module
  docstring). Stateful streamable HTTP pins an MCP *protocol* session to
  whichever worker process handled `initialize`; a follow-up `tools/call`
  landing on a different worker would get "session not found" at the MCP
  transport layer, even though our own `session_id` (stored in Redis) is
  still perfectly valid on any worker. Every tool call here is already
  self-contained via that `session_id` argument, so there is nothing for a
  stateful MCP transport session to usefully hold onto — statelessness
  avoids the multi-worker failure mode instead of working around it.
- DNS-rebinding protection is opt-in via env var
  `AGENTIC_COMPLIANCE_MCP_ALLOWED_HOSTS` (comma-separated `Host` header
  values to accept). Left unset, protection is OFF: the plan's explicit
  decision for this module was "open like the rest of the API, no
  additional protection" (nothing under /api/agentic-compliance requires
  auth). Set the env var if this endpoint ends up reachable from a network
  path a browser could reach and DNS-rebind through.

See docs/AGENTIC_COMPLIANCE_MCP.md for the full writeup, including why this
mounts as a plain Starlette Route (built in controllers.py) instead of
`app.mount()`-ing the Starlette app `streamable_http_app()` returns, and why
the transport below is a small stable dispatcher instead of a bare ASGI app.
"""
import contextlib
import logging
import os
from typing import AsyncIterator, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import Receive, Scope, Send

from tilellm.modules.agentic_compliance_checker.services.langchain_tools import (
    AGENTIC_COMPLIANCE_TOOLS,
)

logger = logging.getLogger(__name__)

_allowed_hosts = [
    h.strip()
    for h in os.environ.get("AGENTIC_COMPLIANCE_MCP_ALLOWED_HOSTS", "").split(",")
    if h.strip()
]


def _build_mcp() -> FastMCP:
    mcp = FastMCP(
        "tiledesk-agentic-compliance",
        instructions=(
            "Tool per la verifica di conformita' di gare pubbliche (D.Lgs. 36/2023). "
            "Apri prima una sessione con POST /api/agentic-compliance/sessions per "
            "ottenere un session_id, poi chiama compliance_list_requirements per "
            "scoprire gli id dei requisiti/criteri della gara."
        ),
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=bool(_allowed_hosts),
            allowed_hosts=_allowed_hosts,
            allowed_origins=_allowed_hosts,
        ),
    )
    for name, tool_obj in AGENTIC_COMPLIANCE_TOOLS.items():
        # .coroutine is the plain async function `@tool(args_schema=...)` wraps —
        # reusing it, not re-wrapping tools_core's _core coroutines a second time.
        mcp.add_tool(tool_obj.coroutine, name=name, description=tool_obj.description)
    return mcp


class _MCPTransport:
    """Stable ASGI-callable dispatcher, registered as a Route exactly once at
    import time (see controllers.py), that delegates each request to
    whichever FastMCP instance is currently active.

    Why the indirection: mcp.server.streamable_http_manager.
    StreamableHTTPSessionManager.run() may be entered exactly ONCE per
    instance — "Create a new instance if you need to run again" is its own
    error message. A single module-level `FastMCP`/session_manager built
    once at import time works in production (one worker process, one app
    lifespan, one .run() call for that process's entire life) but breaks the
    moment a lifespan cycle happens more than once in the same process — the
    case a pytest run with fakeredis (via tests/conftest.py's `client`
    fixture) or any other repeated TestClient(app) usage hits directly. This
    class rebuilds a fresh FastMCP (and re-enters .run() on its fresh
    session_manager) on every `activate()`, so each app lifespan cycle gets
    an instance that has only ever run once — true in production and in
    a test suite alike.
    """

    def __init__(self) -> None:
        self._active: Optional[FastMCP] = None
        self._endpoint = None

    @contextlib.asynccontextmanager
    async def activate(self) -> AsyncIterator[None]:
        mcp = _build_mcp()
        # streamable_http_app() lazily creates mcp's session_manager and
        # returns a throwaway Starlette app whose only job is holding the one
        # ASGI-callable Route we actually want — see controllers.py for why
        # we take just the callable instead of mounting that whole app.
        endpoint = mcp.streamable_http_app().routes[0].endpoint
        async with mcp.session_manager.run():
            self._active, self._endpoint = mcp, endpoint
            try:
                yield
            finally:
                self._active, self._endpoint = None, None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._endpoint is None:
            # Only reachable if a request lands outside an active app
            # lifespan (should not happen under uvicorn/gunicorn) or in a
            # test that hits this route without going through the app's
            # lifespan context — fail loudly rather than silently 404.
            logger.error("MCP transport hit while inactive — app lifespan not running?")
            raise RuntimeError(
                "Agentic compliance MCP transport is not active "
                "(the FastAPI app lifespan is not running)."
            )
        await self._endpoint(scope, receive, send)


# Built once, at import time — this object's identity is what
# controllers.py's Route holds onto; its behavior changes across lifespan
# cycles, but the Route itself never needs re-registering.
transport = _MCPTransport()

# Exposed for tests / introspection that want the tool list without spinning
# up a full app lifespan (e.g. asserting all seven tools are registered).
# Safe to build extra instances of this — see _build_mcp's docstring context
# above: only the *running* transport (via `transport`) is constrained to
# one .run() per instance, not FastMCP construction itself.
mcp = _build_mcp()
