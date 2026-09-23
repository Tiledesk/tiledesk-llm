"""
MCP server adapter (P6): the same seven tools as langchain_tools.py, reached
over the Model Context Protocol instead of POST /api/ask.

Two of these tests exist specifically because the "obvious" implementation
(`router.mount(path, mcp.streamable_http_app())`) fails completely silently:
FastAPI's APIRouter.include_router() drops raw Starlette Mount routes, and
FastMCP's own route uses methods=None which include_router() turns into an
empty (nobody-can-call-it) list. Both failure modes produce a working import
and a working app — no exception anywhere — so only an actual HTTP round
trip catches a regression. See controllers.py's comment on the same lines.
"""
import json

import fakeredis.aioredis
import pytest

from tilellm.modules.agentic_compliance_checker.services.langchain_tools import (
    AGENTIC_COMPLIANCE_TOOLS,
)
from tilellm.modules.agentic_compliance_checker.services.mcp_server import mcp
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore

_MCP_PATH = "/api/agentic-compliance/mcp"
_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

_MINIMAL_YAML = """\
tender:
  title: Gara test
  lot_id: L1
  lot_name: Lotto 1
requirements:
  discretionary:
    - id: P1
      text: plasticità
      mode: variabile
      max_points: 8
"""


@pytest.fixture
def fake_redis():
    SessionStore._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield SessionStore._client
    SessionStore._client = None


def _rpc(client, method, params, msg_id=1):
    resp = client.post(_MCP_PATH, json={"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params},
                        headers=_HEADERS)
    assert resp.status_code == 200, resp.text
    # streamable_http_path responses are SSE-framed ("event: message\ndata: {...}")
    raw = resp.text.split("data: ", 1)[1] if "data: " in resp.text else resp.text
    return json.loads(raw)


def _open_session(client):
    payload = {
        "requirements_yaml": _MINIMAL_YAML,
        "operators": [{"namespace": "ns-oe1", "operator_label": "OE 1"}],
        "engine": {"name": "qdrant"},
        "llm": "openai",
        "gptkey": "sk-test-secret",
        "model": "gpt-4o-mini",
    }
    return client.post("/api/agentic-compliance/sessions", json=payload).json()["session_id"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_seven_tools_registered_on_mcp():
    registered = {t.name for t in await mcp.list_tools()}
    assert registered == set(AGENTIC_COMPLIANCE_TOOLS.keys())


def test_mcp_route_has_explicit_non_empty_methods():
    """Pins the fix for FastMCP's own route (methods=None, "any method"),
    which include_router()'s `list(route.methods or [])` silently turns into
    an empty list — a route matched by GET/POST/DELETE, but registered to
    accept none of them. If this ever regresses to methods=None or [], every
    other test in this file would also fail, but this one names the cause."""
    from tilellm.__main__ import app

    route = next(r for r in app.routes if getattr(r, "path", None) == _MCP_PATH)
    assert route.methods and route.methods >= {"GET", "POST", "DELETE"}


# ---------------------------------------------------------------------------
# Protocol round trip (the real regression coverage: full ASGI requests
# through tilellm.__main__.app, not an in-process mcp.call_tool()).
#
# All in ONE test / ONE `client` fixture instance deliberately:
# StreamableHTTPSessionManager.run() may be called exactly ONCE per instance
# ("Create a new instance if you need to run again" — its own error message).
# `mcp.session_manager` is a module-level singleton (created once, when
# services/mcp_server.py is first imported), while `client` is a
# function-scoped fixture that opens a fresh TestClient — and therefore a
# fresh app lifespan, and therefore a fresh `.run()` call on that same
# singleton — for every test that uses it. A second test using `client`
# would hit that RuntimeError, not a bug in this module: one lifespan cycle
# genuinely is what this manager expects (matches production exactly — one
# worker process, one lifespan, one `.run()` for the process's lifetime).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_protocol_round_trip_over_http(client, fake_redis):
    init_body = _rpc(client, "initialize", {
        "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"},
    }, msg_id=1)
    assert init_body["result"]["serverInfo"]["name"] == "tiledesk-agentic-compliance"

    list_body = _rpc(client, "tools/list", {}, msg_id=2)
    names = {t["name"] for t in list_body["result"]["tools"]}
    assert names == set(AGENTIC_COMPLIANCE_TOOLS.keys())

    error_body = _rpc(client, "tools/call", {
        "name": "compliance_list_requirements", "arguments": {"session_id": "does-not-exist"},
    }, msg_id=3)
    # _safe() turned SessionNotFound into a JSON error string, not a raised
    # exception — isError stays False, same shape the LangChain adapter returns.
    assert error_body["result"]["isError"] is False
    assert "error" in json.loads(error_body["result"]["content"][0]["text"])

    # Acceptance check: MCP and LangChain adapters, same tool, same session, same result.
    session_id = _open_session(client)
    langchain_result = await AGENTIC_COMPLIANCE_TOOLS["compliance_list_requirements"].ainvoke({
        "session_id": session_id,
    })
    call_body = _rpc(client, "tools/call", {
        "name": "compliance_list_requirements", "arguments": {"session_id": session_id},
    }, msg_id=4)
    mcp_result = call_body["result"]["content"][0]["text"]
    assert json.loads(mcp_result) == json.loads(langchain_result)


# ---------------------------------------------------------------------------
# Anti-bypass regression test, MCP side (mirrors test_evaluate_and_report.py's
# LangChain-side test_no_tool_args_schema_exposes_a_verdict_field — the
# guardrail must hold on BOTH surfaces, not just the one that happened to be
# tested first)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_mcp_tool_input_schema_exposes_a_verdict_field():
    forbidden = {"score", "coefficient", "confidence", "human_review_required", "judgment", "gptkey"}
    for tool in await mcp.list_tools():
        fields = set(tool.inputSchema.get("properties", {}).keys())
        assert not (fields & forbidden), f"{tool.name} exposes forbidden field(s): {fields & forbidden}"
