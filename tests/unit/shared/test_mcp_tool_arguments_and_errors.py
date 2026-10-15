"""
MCP tools: fixed per-server arguments (ServerConfig.parameters.tool_arguments)
and tool-call errors returned to the model instead of killing the agent.

Shape of the real case: a commerce MCP server wants, in every tools/call, an
argument `meta: {"ucp-agent": {"profile": "<public url>"}}`. The model cannot
know the URL; without it the server answers HTTP 422 with a JSON-RPC error body.
mcp 2.3 turns that body into an MCPError, which langchain.mcp does not handle
(only isError results are): the whole agent failed.

The server below is a REAL MCP server (fastmcp + uvicorn on a free port), so the
arguments are checked on the wire, with both the legacy and the 2026-07-28 protocol.
"""
import asyncio
import json
import logging
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastmcp import FastMCP
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from mcp.shared.exceptions import MCPError

from tilellm.controller.controller import handle_agent_exception
from tilellm.models.base import ServerConfig
from tilellm.shared import mcp_client

PROFILE_URL = "https://shop-esempio.example/profiles/agent.json"
SECRET_HEADER_VALUE = "tok-segreto-0123456789"
UCP_ERROR = "UCP discovery failed: Missing profile uri"
UCP_PARAMS = {"tool_arguments": {"meta": {"ucp-agent": {"profile": PROFILE_URL}}}}

received: list = []
wire_arguments: list = []  # raw tools/call arguments, as sent on the wire


def _build_app():
    srv = FastMCP("shop-esempio")

    @srv.tool
    def search_catalog(meta: dict, catalog: dict) -> str:
        """Search the shop catalog."""
        received.append(("search_catalog", {"meta": meta, "catalog": catalog}))
        return "found 3 products"

    @srv.tool
    def get_store_hours(day: str) -> str:
        """Opening hours."""
        received.append(("get_store_hours", {"day": day}))
        return "9-18"

    inner = srv.http_app(path="/mcp")

    async def app(scope, receive, send):
        """422 + JSON-RPC error on a search_catalog call without the expected profile."""
        if scope["type"] != "http" or scope["method"] != "POST":
            return await inner(scope, receive, send)
        body, more = b"", True
        while more:
            msg = await receive()
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
        payload = json.loads(body or b"{}")
        if payload.get("method") == "tools/call":
            args = payload["params"].get("arguments") or {}
            wire_arguments.append((payload["params"]["name"], args))
            profile = ((args.get("meta") or {}).get("ucp-agent") or {}).get("profile")
            if payload["params"]["name"] == "search_catalog" and profile != PROFILE_URL:
                err = json.dumps({
                    "jsonrpc": "2.0", "id": payload.get("id"),
                    "error": {"code": -32602, "message": UCP_ERROR},
                }).encode()
                await send({"type": "http.response.start", "status": 422,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": err})
                return

        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await inner(scope, replay, send)

    return app


@pytest.fixture(scope="module")
def mcp_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(_build_app(), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}/mcp?client=abc"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "127.0.0.1")
    received.clear()
    wire_arguments.clear()


@pytest.fixture
def mcp_log(caplog):
    # the app's logging config stops "tilellm.*" from propagating to the root handler
    log = logging.getLogger("tilellm.shared.mcp_client")
    log.addHandler(caplog.handler)
    log.setLevel(logging.DEBUG)
    yield caplog
    log.removeHandler(caplog.handler)


async def _tools(url, parameters=None, protocol="auto"):
    cfg = {"transport": "streamable_http", "url": url, "protocol": protocol,
           "headers": {"x-shop-token": SECRET_HEADER_VALUE}}
    if parameters is not None:
        cfg["parameters"] = parameters
    tools = await mcp_client.load_tools({"shop": ServerConfig(**cfg)})
    return {t.name: t for t in tools}


def _call(name, args):
    """A tool call as the agent makes it: the result is a ToolMessage."""
    return {"type": "tool_call", "name": name, "args": args, "id": "call-1"}


PROTOCOLS = pytest.mark.parametrize("protocol", ["legacy", "auto"])


@pytest.mark.asyncio
@PROTOCOLS
async def test_422_from_server_becomes_a_tool_error_message(mcp_url, protocol):
    tools = await _tools(mcp_url, protocol=protocol)

    msg = await tools["search_catalog"].ainvoke(_call("search_catalog", {"meta": {}, "catalog": {"q": "scarpe"}}))

    assert isinstance(msg, ToolMessage) and msg.status == "error"
    assert "MCP tool search_catalog failed" in msg.text
    assert UCP_ERROR in msg.text
    assert "client=abc" not in msg.text and "127.0.0.1" not in msg.text


@pytest.mark.asyncio
@PROTOCOLS
async def test_tool_arguments_are_merged_into_the_call(mcp_url, protocol):
    tools = await _tools(mcp_url, UCP_PARAMS, protocol)

    msg = await tools["search_catalog"].ainvoke(_call("search_catalog", {"catalog": {"q": "scarpe"}}))

    assert msg.status == "success" and "found 3 products" in msg.text
    assert wire_arguments == [("search_catalog", {
        "meta": {"ucp-agent": {"profile": PROFILE_URL}}, "catalog": {"q": "scarpe"},
    })]


@pytest.mark.asyncio
async def test_configured_key_replaces_the_model_value_entirely(mcp_url):
    """The model must not be able to add sub-keys under a configured key."""
    tools = await _tools(mcp_url, UCP_PARAMS)

    await tools["search_catalog"].ainvoke({
        "meta": {"ucp-agent": {"profile": "https://inventato.example/x", "extra": 1}, "lang": "it"},
        "catalog": {"q": "scarpe"},
    })

    assert wire_arguments[0][1]["meta"] == {"ucp-agent": {"profile": PROFILE_URL}}
    assert wire_arguments[0][1]["catalog"] == {"q": "scarpe"}  # keys not configured are untouched


@pytest.mark.asyncio
async def test_configured_keys_are_hidden_from_the_model_schema(mcp_url):
    tools = await _tools(mcp_url, UCP_PARAMS)

    schema = tools["search_catalog"].args_schema
    assert "meta" not in schema["properties"] and "meta" not in schema.get("required", [])
    assert "catalog" in schema["properties"] and "catalog" in schema["required"]

    # without config the schema is untouched
    plain = await _tools(mcp_url)
    assert "meta" in plain["search_catalog"].args_schema["required"]


@pytest.mark.asyncio
async def test_tool_without_the_key_in_schema_gets_no_extra_arguments(mcp_url):
    tools = await _tools(mcp_url, UCP_PARAMS)

    await tools["get_store_hours"].ainvoke({"day": "monday"})

    assert wire_arguments == [("get_store_hours", {"day": "monday"})]
    assert received == [("get_store_hours", {"day": "monday"})]


@pytest.mark.asyncio
async def test_concurrent_calls_keep_their_own_arguments(mcp_url):
    tools = await _tools(mcp_url, UCP_PARAMS)

    await asyncio.gather(*(tools["search_catalog"].ainvoke({"catalog": {"q": f"p{i}"}}) for i in range(5)))

    assert sorted(a["catalog"]["q"] for _, a in wire_arguments) == [f"p{i}" for i in range(5)]
    assert all(a["meta"] == {"ucp-agent": {"profile": PROFILE_URL}} for _, a in wire_arguments)


@pytest.mark.asyncio
async def test_header_and_argument_values_never_reach_the_logs(mcp_url, mcp_log):
    tools = await _tools(mcp_url, UCP_PARAMS)
    await tools["search_catalog"].ainvoke({"catalog": {}})
    plain = await _tools(mcp_url)
    await plain["search_catalog"].ainvoke(_call("search_catalog", {"meta": {}, "catalog": {}}))

    # ours only: httpx2 logs each request URL itself (INFO, "httpx2" logger)
    text = "\n".join(r.getMessage() for r in mcp_log.records if r.name.startswith("tilellm"))
    assert "x-shop-token" in text  # header names are fine
    assert SECRET_HEADER_VALUE not in text
    assert PROFILE_URL not in text
    assert "client=abc" not in text


@pytest.mark.asyncio
async def test_tool_arguments_no_tool_declares_are_warned(mcp_url, mcp_log):
    tools = await _tools(mcp_url, {"tool_arguments": {"sconosciuta": "VALORE-NON-LOGGARE"}})

    assert "meta" in tools["search_catalog"].args_schema["properties"]
    warnings = [r.getMessage() for r in mcp_log.records if r.levelno == logging.WARNING]
    assert any("shop" in m and "sconosciuta" in m for m in warnings), warnings
    assert "VALORE-NON-LOGGARE" not in mcp_log.text


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [["meta"], "meta", 42])
async def test_tool_arguments_not_a_dict_is_ignored_without_losing_tools(mcp_url, mcp_log, bad):
    tools = await _tools(mcp_url, {"tool_arguments": bad})

    assert set(tools) == {"search_catalog", "get_store_hours"}
    assert "meta" in tools["search_catalog"].args_schema["required"]
    errors = [r.getMessage() for r in mcp_log.records if r.levelno == logging.ERROR]
    assert any("tool_arguments" in m and "shop" in m for m in errors), errors


# --- errors raised inside the call (transport, timeout, task groups) ---------------

def _failing_tool(exc: BaseException) -> StructuredTool:
    async def call(**arguments):
        raise exc

    def mcp_handler(err):  # what langchain.mcp installs: re-raises what it doesn't own
        raise err

    return StructuredTool(name="search_catalog", description="", coroutine=call,
                          args_schema={"type": "object", "properties": {"q": {"type": "string"}}},
                          response_format="content_and_artifact", handle_tool_error=mcp_handler)


def _request():
    return httpx.Request("POST", "https://shop-esempio.example/mcp?key=SEGRETO-Q")


@pytest.mark.asyncio
@pytest.mark.parametrize("exc, expected", [
    (TimeoutError(), "TimeoutError"),
    (MCPError(-32602, UCP_ERROR), UCP_ERROR),
    (BaseExceptionGroup("unhandled errors in a TaskGroup", [MCPError(-32602, UCP_ERROR)]), UCP_ERROR),
    (httpx.ConnectError("connessione a https://interno.example/x?key=SEGRETO-Q fallita"), "ConnectError"),
    (httpx.HTTPStatusError("Client error for url https://shop-esempio.example/mcp?key=SEGRETO-Q",
                           request=_request(),
                           response=httpx.Response(422, request=_request(), json={
                               "jsonrpc": "2.0", "error": {"code": -32602, "message": UCP_ERROR}})),
     f"HTTP 422 — {UCP_ERROR}"),
])
async def test_call_errors_become_tool_messages(exc, expected):
    tool = _failing_tool(exc)
    mcp_client._prepare_tool(tool, "shop", {})

    msg = await tool.ainvoke(_call("search_catalog", {"q": "x"}))

    assert msg.status == "error"
    assert msg.content.startswith("MCP tool search_catalog failed: ")
    assert expected in msg.content
    assert "SEGRETO-Q" not in msg.content and "interno.example" not in msg.content
    assert len(msg.content) <= 500 + len("MCP tool search_catalog failed: ")


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [
    asyncio.CancelledError(),
    BaseExceptionGroup("unhandled errors in a TaskGroup", [asyncio.CancelledError()]),
])
async def test_cancellation_is_not_turned_into_a_tool_message(exc):
    tool = _failing_tool(exc)
    mcp_client._prepare_tool(tool, "shop", {})

    with pytest.raises(BaseException) as raised:
        await tool.ainvoke(_call("search_catalog", {"q": "x"}))
    assert raised.value is exc


# --- agent error reply --------------------------------------------------------------

def test_agent_exception_group_shows_root_cause():
    response = httpx.Response(422, request=_request(), json={"jsonrpc": "2.0", "error": {"code": 1, "message": UCP_ERROR}})
    root = httpx.HTTPStatusError("Client error '422' for url https://shop-esempio.example/mcp?key=SEGRETO-Q",
                                 request=_request(), response=response)
    group = BaseExceptionGroup("unhandled errors in a TaskGroup", [root])

    body = json.loads(handle_agent_exception(group, "test").body)

    assert "TaskGroup" not in body["answer"]
    assert "422" in body["answer"] and UCP_ERROR in body["answer"]
    assert "SEGRETO-Q" not in body["answer"]


def test_agent_error_detail_stays_in_the_log(caplog):
    exc = RuntimeError("tool error: connessione a https://interno.example/x?key=SEGRETO-ERR fallita")

    response = handle_agent_exception(exc, "test")
    body = json.loads(response.body)

    assert response.status_code == 400  # classification unchanged
    assert "SEGRETO-ERR" not in body["answer"] and "interno.example" not in body["answer"]


def test_agent_error_with_status_shows_only_the_status():
    class ProviderError(Exception):
        status_code = 429

    body = json.loads(handle_agent_exception(ProviderError("rate limit for key sk-FINTA-123 on model x"), "test").body)

    assert "HTTP 429" in body["answer"]
    assert "sk-FINTA-123" not in body["answer"]


# --- SSE: a 4xx on the POST must not hang the request -------------------------------
# mcp 2.3's SSE post_writer logs and drops a failed POST: the response never comes on
# the stream (pings keep it alive). Only the client's request timeout ends the wait.

def _build_sse_app():
    srv = FastMCP("shop-sse")

    @srv.tool
    def get_store_hours(day: str) -> str:
        """Opening hours."""
        return "9-18"

    inner = srv.http_app(path="/sse", transport="sse")

    async def app(scope, receive, send):
        """403 on every POST with header x-deny: all, else on tools/call only."""
        if scope["type"] != "http" or scope["method"] != "POST":
            return await inner(scope, receive, send)
        body, more = b"", True
        while more:
            msg = await receive()
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
        deny_all = (b"x-deny", b"all") in scope["headers"]
        if deny_all or json.loads(body or b"{}").get("method") == "tools/call":
            await send({"type": "http.response.start", "status": 403, "headers": []})
            await send({"type": "http.response.body", "body": b"forbidden"})
            return
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await inner(scope, replay, send)

    return app


@pytest.fixture(scope="module")
def sse_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(_build_sse_app(), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}/sse"
    server.should_exit = True
    thread.join(timeout=5)


def _sse(url, **headers):
    return {"shop": ServerConfig(transport="sse", url=url, headers=headers or None)}


@pytest.mark.asyncio
async def test_sse_403_on_initialize_ends_within_the_request_timeout(sse_url, monkeypatch, mcp_log):
    monkeypatch.setenv("MCP_REQUEST_TIMEOUT", "1")
    started = time.monotonic()

    async with asyncio.timeout(15):  # guard: without the timeout the call never returns
        tools = await mcp_client.load_tools(_sse(sse_url, **{"x-deny": "all"}))

    assert tools == []
    assert time.monotonic() - started < 10
    assert any("shop" in r.getMessage() for r in mcp_log.records if r.levelno == logging.ERROR)


@pytest.mark.asyncio
async def test_sse_403_on_tool_call_becomes_a_tool_error_message(sse_url, monkeypatch):
    monkeypatch.setenv("MCP_REQUEST_TIMEOUT", "1")
    tools = {t.name: t for t in await mcp_client.load_tools(_sse(sse_url))}

    async with asyncio.timeout(15):
        msg = await tools["get_store_hours"].ainvoke(_call("get_store_hours", {"day": "monday"}))

    assert isinstance(msg, ToolMessage) and msg.status == "error"
    assert "MCP tool get_store_hours failed" in msg.text


def test_request_timeout_defaults_to_120(monkeypatch):
    monkeypatch.delenv("MCP_REQUEST_TIMEOUT", raising=False)
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "mcp-esempio.example")
    client = mcp_client.build_client(ServerConfig(transport="streamable_http", url="https://mcp-esempio.example/mcp"))
    assert client._session_kwargs["read_timeout_seconds"] == 120
