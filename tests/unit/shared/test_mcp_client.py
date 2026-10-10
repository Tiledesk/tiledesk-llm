"""
MCP client on langchain.mcp / fastmcp (protocol 2026-07-28 with fallback to the
initialize handshake). Against REAL HTTP servers (uvicorn in a thread, keep-alive):
mocks don't see event-loop or negotiation bugs.

- protocol is negotiated per server: ServerConfig.protocol "auto" (default) /
  "legacy" / "2026-07-28";
- tool names stay exactly as the server declares them (no "{server}_" prefix),
  also with several servers in one request;
- caller headers reach the server; header values never reach the logs;
- the server URL passes the SSRF guard before any connection;
- SSE still accepted (deprecated, logged), stdio still rejected.
"""
import logging
import socket
import threading
import time

from unittest.mock import patch

import pytest
import uvicorn
from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import get_http_headers
from mcp_types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult
from pydantic import ValidationError

from tilellm.models.base import ServerConfig
from tilellm.shared import mcp_client


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(server: FastMCP, transport: str = "http"):
    port = _free_port()
    app = server.http_app(path="/mcp", transport=transport)
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=uv.run, daemon=True)
    t.start()
    for _ in range(100):
        if uv.started:
            break
        time.sleep(0.05)
    return uv, f"http://127.0.0.1:{port}/mcp"


def _make_server(name: str) -> FastMCP:
    srv = FastMCP(name)

    @srv.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @srv.tool
    def tenant_header() -> str:
        """Return the x-tenant header the client sent."""
        return get_http_headers().get("x-tenant", "")

    @srv.tool
    async def ask_user(ctx: Context) -> str:
        """Ask the user something (elicitation)."""
        result = await ctx.elicit("Come ti chiami?", response_type=str)
        return f"elicitation:{result.action}"

    @srv.tool
    async def ask_user_mrtr(ctx: Context) -> str | InputRequiredResult:
        """Ask the user something (2026-07-28 multi round-trip)."""
        if ctx.input_responses is None:
            return InputRequiredResult(input_requests={"name": ElicitRequest(params=ElicitRequestFormParams(
                message="Come ti chiami?",
                requested_schema={"type": "object", "properties": {"name": {"type": "string"}}}))})
        return f"elicitation:{ctx.input_responses['name'].action}"

    return srv


@pytest.fixture(scope="module")
def servers():
    started = [_serve(_make_server("alpha")), _serve(_make_server("beta"))]
    yield [url for _, url in started]
    for uv, _ in started:
        uv.should_exit = True


@pytest.fixture(autouse=True)
def _allow_loopback(monkeypatch):
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "127.0.0.1")


@pytest.fixture
def mcp_log(caplog):
    # the app's logging config stops "tilellm.*" from propagating to the root handler
    log = logging.getLogger("tilellm.shared.mcp_client")
    log.addHandler(caplog.handler)
    log.setLevel(logging.DEBUG)
    yield caplog
    log.removeHandler(caplog.handler)


def _cfg(url, **kw):
    return ServerConfig(transport="streamable_http", url=url, **kw)


def test_protocol_defaults_to_auto_and_rejects_unknown_values():
    assert ServerConfig(transport="streamable_http", url="https://mcp.example/mcp").protocol == "auto"
    with pytest.raises(ValidationError):
        ServerConfig(transport="streamable_http", url="https://mcp.example/mcp", protocol="2024-01-01")


def test_stdio_is_still_rejected():
    with pytest.raises(ValidationError):
        ServerConfig(transport="stdio", url="https://mcp.example/mcp")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol, expected", [("auto", "2026-07-28"), ("2026-07-28", "2026-07-28"),
                                                ("legacy", "2025-11-25")])
async def test_protocol_is_negotiated_per_server(servers, protocol, expected):
    client = mcp_client.build_client(_cfg(servers[0], protocol=protocol))
    async with client:
        assert client.protocol_version == expected


@pytest.mark.asyncio
async def test_two_servers_of_different_eras_in_one_request_keep_plain_tool_names(servers):
    tools = await mcp_client.load_tools({
        "alpha": _cfg(servers[0], protocol="legacy", enabled_tools=["add"]),
        "beta": _cfg(servers[1], headers={"x-tenant": "azienda-esempio"}),
    })
    names = sorted(t.name for t in tools)
    assert names == ["add", "add", "ask_user", "ask_user_mrtr", "tenant_header"]

    add = next(t for t in tools if t.name == "add")
    assert "3" in str(await add.ainvoke({"a": 1, "b": 2}))
    header_tool = next(t for t in tools if t.name == "tenant_header")
    assert "azienda-esempio" in str(await header_tool.ainvoke({}))


@pytest.mark.asyncio
async def test_sse_transport_still_works_and_logs_deprecation(mcp_log):
    uv, url = _serve(_make_server("gamma"), transport="sse")
    try:
        tools = await mcp_client.load_tools({"gamma": ServerConfig(transport="sse", url=url)})
        assert "add" in [t.name for t in tools]
        assert any(r.levelno >= logging.WARNING and "deprecat" in r.getMessage().lower()
                   for r in mcp_log.records)
    finally:
        uv.should_exit = True


@pytest.mark.asyncio
async def test_metadata_address_is_refused_before_connecting(monkeypatch, mcp_log):
    monkeypatch.delenv("OUTBOUND_URL_ALLOWED_HOSTS", raising=False)
    with patch.object(mcp_client, "StreamableHttpTransport") as transport:
        tools = await mcp_client.load_tools({"evil": _cfg("http://169.254.169.254/mcp")})
    assert tools == []
    transport.assert_not_called()
    assert "rifiutato" in mcp_log.text


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol, tool", [("legacy", "ask_user"), ("auto", "ask_user_mrtr")])
async def test_elicitation_is_declined_instead_of_suspending_the_agent(servers, protocol, tool):
    """legacy: server-initiated elicitation/create; 2026-07-28: InputRequiredResult (MRTR)."""
    tools = await mcp_client.load_tools({"alpha": _cfg(servers[0], protocol=protocol, enabled_tools=[tool])})
    assert "elicitation:decline" in str(await tools[0].ainvoke({}))


@pytest.mark.asyncio
async def test_header_values_never_reach_the_logs(servers, mcp_log):
    secret = "Bearer sk-segreto-di-prova-123"
    await mcp_client.load_tools({"beta": _cfg(servers[1], headers={"Authorization": secret})})
    assert "sk-segreto-di-prova-123" not in mcp_log.text
    assert "Authorization" in mcp_log.text  # the key is still logged, for diagnosis
