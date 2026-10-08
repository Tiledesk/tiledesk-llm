"""
MCP server adapter (P6) — the same seven tools as services/langchain_tools.py,
exposed over the Model Context Protocol (streamable HTTP) for an external
agent (Claude Desktop, another orchestrator) that does not go through
POST /api/ask.

This file registers the same functions langchain_tools.py binds
(`AGENTIC_COMPLIANCE_TOOLS[name].function`, `_safe()` error handling
included) — not a second implementation of the seven tools — with the same
argument schema (`*Args`, field descriptions included). Both adapters expose
identical behaviour by construction.

The session is NOT a tool argument: the MCP client sends it in the
X-Compliance-Session header of the HTTP connection, read here from the
request. It never enters the model's context, so it can't leak from it nor be
swapped by the agent.

Config choices, both deployment-driven:

- `stateless_http=True`: entrypoint.sh runs gunicorn with WORKERS=3 by
  default (the same constraint documented in session_store.py's module
  docstring). Stateful streamable HTTP pins an MCP *protocol* session to
  whichever worker process handled `initialize`; a follow-up `tools/call`
  landing on a different worker would get "session not found" at the MCP
  transport layer, even though our own `session_id` (stored in Redis) is
  still perfectly valid on any worker. Every tool call here is already
  self-contained via the X-Compliance-Session header, so there is nothing for a
  stateful MCP transport session to usefully hold onto — statelessness
  avoids the multi-worker failure mode instead of working around it.
- DNS-rebinding protection is opt-in via env var
  `AGENTIC_COMPLIANCE_MCP_ALLOWED_HOSTS` (comma-separated `Host` header
  values to accept). Left unset, protection is OFF: the plan's explicit
  decision for this module was "open like the rest of the API, no
  additional protection" (nothing under /api/agentic-compliance requires
  auth). Set the env var if this endpoint ends up reachable from a network
  path a browser could reach and DNS-rebind through.

It mounts as a plain Starlette Route (built in controllers.py) instead of
`app.mount()`-ing the Starlette app `streamable_http_app()` returns, and the
transport below is a small stable dispatcher instead of a bare ASGI app (see
the comments in controllers.py and _MCPTransport).
"""
import contextlib
import inspect
import json
import logging
import os
from typing import Any, AsyncIterator, Callable, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import Receive, Scope, Send

from tilellm.modules.agentic_compliance_checker.services.langchain_tools import (
    AGENTIC_COMPLIANCE_TOOLS,
    tool_description,
)

logger = logging.getLogger(__name__)

_allowed_hosts = [
    h.strip()
    for h in os.environ.get("AGENTIC_COMPLIANCE_MCP_ALLOWED_HOSTS", "").split(",")
    if h.strip()
]


SESSION_HEADER = "X-Compliance-Session"


def _session_from(ctx: Context) -> Optional[str]:
    request = ctx.request_context.request
    return request.headers.get(SESSION_HEADER) if request is not None else None


def _mcp_tool(name: str) -> Callable[..., Any]:
    """The tool function with the session taken from the request header and
    the argument signature of its `*Args` schema (FastMCP builds the input
    schema from the signature: no session_id in it)."""
    spec = AGENTIC_COMPLIANCE_TOOLS[name]

    async def run(ctx: Context, **kwargs: Any) -> str:
        session_id = _session_from(ctx)
        if not session_id:
            return json.dumps({"error": f"Missing {SESSION_HEADER} header (session from "
                                        "POST /api/agentic-compliance/sessions)."})
        return await spec.function(session_id=session_id, **kwargs)

    params = [inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY, annotation=Context)]
    params += [
        inspect.Parameter(field_name, inspect.Parameter.KEYWORD_ONLY, default=field, annotation=field.annotation)
        for field_name, field in spec.args_schema.model_fields.items()
    ]
    run.__signature__ = inspect.Signature(params, return_annotation=str)
    run.__name__ = name
    return run


def _build_mcp() -> FastMCP:
    mcp = FastMCP(
        "tiledesk-agentic-compliance",
        instructions=(
            "Tool per la verifica di conformita' di gare pubbliche (D.Lgs. 36/2023). "
            "Apri prima una sessione con POST /api/agentic-compliance/sessions e invia "
            "il session_id ottenuto nell'header HTTP X-Compliance-Session della "
            "connessione MCP (mai come argomento dei tool), poi chiama "
            "compliance_list_requirements per scoprire gli id dei requisiti/criteri."
        ),
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=bool(_allowed_hosts),
            allowed_hosts=_allowed_hosts,
            allowed_origins=_allowed_hosts,
        ),
    )
    for name in AGENTIC_COMPLIANCE_TOOLS:
        mcp.add_tool(_mcp_tool(name), name=name, description=tool_description(name))
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
