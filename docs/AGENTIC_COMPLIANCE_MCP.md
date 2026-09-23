# Agentic Compliance Checker — MCP server (P6)

## Overview

`tilellm/modules/agentic_compliance_checker/` exposes `compliance_checker`'s
scoring/guardrail logic as seven tools an agent decides when and how to call,
instead of the single all-or-nothing `POST /api/compliance/v2/check`
endpoint. The full architecture (session model, Redis keys, guardrails) is
documented in memory/the original plan; this document covers only **P6: the
MCP server** — how the same seven tools are reached by an external
[Model Context Protocol](https://modelcontextprotocol.io) client (Claude
Desktop, another orchestrator) instead of through `POST /api/ask`.

Both surfaces are thin adapters over the exact same implementation:

```
tools_core.py            ← ONE implementation per tool (_core coroutines)
        │
        ├── services/langchain_tools.py  ← @tool wrappers, registered into
        │                                   tools_registry.TOOL_REGISTRY,
        │                                   reached via POST /api/ask
        │                                   (tools=[...] selection)
        │
        └── services/mcp_server.py       ← same tools, registered on a
                                             FastMCP server, reached via
                                             POST /api/agentic-compliance/mcp
```

`services/mcp_server.py` does not call `tools_core.py` a second time — it
registers `AGENTIC_COMPLIANCE_TOOLS[name].coroutine`, the exact plain async
function LangChain's `@tool(args_schema=...)` decorator wraps (including its
`_safe()` error handling). Both adapters therefore return byte-identical
JSON for the same call, by construction, not by convention.

## Endpoint

```
POST /api/agentic-compliance/mcp     (streamable HTTP, JSON-RPC 2.0)
```

Usage from an external client is the two-step flow already documented for
the LangChain surface:

1. `POST /api/agentic-compliance/sessions` (regular REST call — full
   `ComplianceRequestV2`/`operators`, including `gptkey`) → `session_id`.
2. Hand the MCP client `session_id` (and the server's tool descriptions,
   which the `initialize` handshake already provides) — it calls
   `compliance_list_requirements`, `compliance_retrieve_evidence`,
   `compliance_evaluate_criteria`, `compliance_check_tabular`,
   `compliance_check_l01`, `compliance_resolve_proportional`,
   `compliance_build_report`, exactly as the internal agent would.

No authentication on this endpoint — an explicit, already-made decision:
"open like the rest of the API" (nothing under `/api/agentic-compliance`
requires auth today). The only barrier is the `session_id` itself
(`secrets.token_urlsafe(32)`, 6h TTL by default); whoever has it can drive
that one compliance run. See "DNS-rebinding protection" below for the one
security knob this endpoint does have.

## Why this needed three non-obvious decisions

Getting a `FastMCP` server correctly mounted under an existing FastAPI app,
composed through the same dynamic module-loading `tilellm/__main__.py`
already uses for every other module (`register_feature_routers` →
`_app.include_router(module.router)`, no direct access to `app`), required
working around three separate limitations that all fail **silently** — no
exception, no error log, just a route that mysteriously 404s or a lifespan
that never starts. Each is verified empirically in this codebase (not
assumed from documentation), and each has a comment at its exact line in
`controllers.py` / `services/mcp_server.py`.

### 1. `router.mount()` doesn't survive `include_router()`

The obvious code is:

```python
router.mount("/mcp", mcp.streamable_http_app())
app.include_router(router)
```

This *looks* correct, imports cleanly, and the app starts up with no error —
but the route is never reachable (404). Reading
`fastapi.routing.APIRouter.include_router`'s source shows why: it iterates
`router.routes` and only forwards `APIRoute`, plain Starlette `Route`,
`APIWebSocketRoute`, and `WebSocketRoute` — there is no branch for
`starlette.routing.Mount`. A `Mount` added via `router.mount()` sits in
`router.routes` but is silently skipped, forever, every time.

**Fix:** register a plain `starlette.routing.Route` directly instead,
wrapping the ASGI callable itself:

```python
from starlette.routing import Route
router.routes.append(Route(full_path, endpoint=asgi_callable, methods=[...]))
```

Starlette's `Route.__init__` treats a class instance (not a function/method)
as an ASGI app it calls directly — the same mechanism FastMCP's own
`streamable_http_app()` uses internally to wire up its
`StreamableHTTPASGIApp`. `include_router`'s `Route` branch **does** forward
this correctly.

### 2. `methods=None` becomes `methods=[]` through `include_router()`

FastMCP builds its own internal route with `methods=None`, Starlette's
convention for "accept any HTTP method" (appropriate for a raw ASGI
passthrough). `include_router`'s `Route` branch does
`methods = list(route.methods or [])` — for `None`, that's `[]`, and the
route gets re-created with an empty method set, i.e. one nothing can ever
call. **Fix:** the `Route` built for step 1 above must pass explicit
`methods=["GET", "POST", "DELETE"]` (the three streamable-HTTP verbs a
spec-compliant MCP client uses).

### 3. The path must be absolute; `router.prefix` is not applied here

`router.get("/sessions")` and friends get `router.prefix` prepended
automatically — but that happens inside those convenience methods, not for
routes appended directly to `router.routes`. `include_router`'s own
`prefix` parameter (a *separate*, optional keyword argument on the
`include_router()` *call*, not the router's own constructor `prefix`)
defaults to `""`, and `tilellm/__main__.py` calls
`_app.include_router(module.router)` with no `prefix=` argument — so a
manually appended `Route("/mcp", ...)` ends up registered at `/mcp`, not
`/api/agentic-compliance/mcp`. **Fix:** bake the full final path into the
`Route` itself (`_MCP_PATH = "/api/agentic-compliance/mcp"` in
`controllers.py`), rather than relying on any prefix being applied for it.

## Why the transport is a stable dispatcher object, not the ASGI app directly

`mcp.server.streamable_http_manager.StreamableHTTPSessionManager.run()` may
be entered **exactly once per instance** — "Create a new instance if you
need to run again" is the SDK's own error message. A single
`FastMCP`/`session_manager` built once at import time (the natural first
attempt) works perfectly in production: one gunicorn worker process, one
FastAPI app lifespan, one `.run()` call for that process's entire life. It
breaks the instant a lifespan cycle happens more than once in the same
Python process — which a test suite using `TestClient(app)` (entering and
exiting the app's lifespan once per test function) hits on the very second
test that touches the endpoint, with no relation to which test file it's in
(this actually broke `test_endpoints.py`, an unrelated pre-existing test
file, the first time this was tried).

**Fix:** `services/mcp_server.py::_MCPTransport` is a small stable
ASGI-callable object, registered as the `Route`'s endpoint exactly once at
import time. Its `activate()` async context manager — entered by the
router's `lifespan`, see below — builds a **fresh** `FastMCP` instance (with
all seven tools freshly registered on it) and enters `.run()` on *that*
instance's session manager. Every app lifespan cycle therefore gets an
instance that has only ever run once, whether that cycle is a real worker
process starting up or the Nth `TestClient(app)` in a test session:

```python
class _MCPTransport:
    async def activate(self):        # entered once per app lifespan cycle
        mcp = _build_mcp()           # fresh FastMCP + fresh session_manager
        endpoint = mcp.streamable_http_app().routes[0].endpoint
        async with mcp.session_manager.run():
            self._endpoint = endpoint
            yield
        self._endpoint = None

    async def __call__(self, scope, receive, send):
        if self._endpoint is None:
            raise RuntimeError("MCP transport is not active")
        await self._endpoint(scope, receive, send)
```

A request arriving outside an active lifespan (should not happen under
uvicorn/gunicorn; can happen in a misconfigured test) fails loudly with a
clear `RuntimeError` instead of silently 404ing or reusing stale state.

## Wiring the lifespan without a fourth edit to `tilellm/__main__.py`

The plan's constraint for this whole module has been at most two lines
changed outside `tilellm/modules/agentic_compliance_checker/` (the
`module_config_mapping` entry and the `get_service_config()` flag — both in
place since P1). Starting/stopping `transport.activate()` together with the
app therefore could not mean touching `__main__.py`'s own lifespan function.

`fastapi.routing.APIRouter` accepts a `lifespan=` context manager at
construction, and `APIRouter.include_router()` (confirmed by reading its
source, not assumed) does `self.lifespan_context =
_merge_lifespan_context(self.lifespan_context, router.lifespan_context)` —
i.e. **an included router's lifespan is automatically composed into the
app's own lifespan**, the same mechanism that already makes
`app.include_router(router)` work for every other module. So:

```python
router = APIRouter(
    prefix="/api/agentic-compliance",
    lifespan=_mcp_lifespan,   # async with transport.activate(): yield
)
```

is enough — no change to `tilellm/__main__.py` at all. Omitting this
`lifespan=` is exactly what produces the "MCP transport is not active"
`RuntimeError` on every request; there is a test pinning this
(`test_mcp_route_has_explicit_non_empty_methods` plus the full protocol
round trip in `test_mcp_protocol_round_trip_over_http`).

## Deployment-driven config

Two settings in `services/mcp_server.py::_build_mcp()` exist specifically
because of how this app is deployed, not as generic defaults:

- **`stateless_http=True`.** `entrypoint.sh` runs gunicorn with `WORKERS=3`
  by default (2-3 across every `docker-compose*.yml`) — the same constraint
  `session_store.py` was built around from P1. *Stateful* streamable HTTP
  pins an MCP *protocol* session (tracked via an `mcp-session-id` header) to
  whichever worker process handled `initialize`; a follow-up `tools/call`
  landing on a different worker fails with "session not found" at the MCP
  transport layer, even though our own `session_id` argument (stored in
  Redis, readable from any worker) is still perfectly valid. Every tool call
  here is already fully self-contained via that argument — there is nothing
  useful for a stateful MCP transport session to hold onto — so statelessness
  sidesteps the multi-worker failure mode instead of working around it.
- **DNS-rebinding protection, opt-in via `AGENTIC_COMPLIANCE_MCP_ALLOWED_HOSTS`**
  (comma-separated `Host` header values to accept). Unset (the default),
  protection is off, consistent with the "open like the rest of the API"
  decision. Set it — a real hostname, not `*` — if this endpoint becomes
  reachable from a network path a browser could reach and DNS-rebind
  through; `mcp.server.transport_security.TransportSecuritySettings` is what
  enforces it underneath.

## Known limitation: per-argument descriptions

`mcp.add_tool()` derives each tool's JSON Schema from the registered
function's own signature (type hints + docstring), not from a separate
schema object. `langchain_tools.py`'s `*Args` Pydantic classes (e.g.
`EvaluateCriteriaArgs`) carry per-field `Field(description=...)` text that
only governs the **LangChain**-side schema — reusing `.coroutine` for MCP
means the MCP-side schema has correct types/required-vs-optional/defaults,
but plainer per-argument descriptions (the tool-level description, the full
docstring, is identical on both surfaces). This is a real but minor
documentation gap, not a functional one — flagged here rather than silently
accepted, and worth revisiting only if an external MCP client's UX actually
suffers for it.

## Testing

`tests/unit/modules/agentic_compliance_checker/test_mcp_server.py`:

- `test_all_seven_tools_registered_on_mcp` — in-process, no lifespan needed.
- `test_mcp_route_has_explicit_non_empty_methods` — pins fix #2 above by
  inspecting the real route on `tilellm.__main__.app`.
- `test_mcp_protocol_round_trip_over_http` — the real regression coverage:
  a full ASGI request through `tilellm.__main__.app` (`initialize`,
  `tools/list`, a `tools/call` against an unknown session, and a
  same-session parity check against the LangChain adapter), **all in one
  test function** deliberately — `StreamableHTTPSessionManager.run()`'s
  once-per-instance constraint means a second test using the shared
  `client` fixture to hit this endpoint would fail with "already run once"
  if split apart, for the exact reason `_MCPTransport` exists (see above).
- `test_no_mcp_tool_input_schema_exposes_a_verdict_field` — the anti-bypass
  guardrail (no tool ever accepts `score`/`coefficient`/`confidence`/
  `human_review_required`/`judgment`/`gptkey`), verified on the MCP schema
  too, mirroring the equivalent LangChain-side test in
  `test_evaluate_and_report.py`.
