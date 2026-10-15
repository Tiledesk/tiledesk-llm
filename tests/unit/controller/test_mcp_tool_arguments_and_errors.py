"""
MCP tools: fixed per-server arguments (ServerConfig.parameters.tool_arguments)
and tool-call errors returned to the model instead of killing the agent.

Shape of the real case: a commerce MCP server ("UCP" protocol) wants, in every
tools/call, an argument `meta: {"ucp-agent": {"profile": "<public url>"}}`.
The model cannot know the URL; without it the server answers HTTP 422 with a
JSON-RPC error body. mcp 1.26 raises inside a TaskGroup, the error surfaced as
"unhandled errors in a TaskGroup (1 sub-exception)" and the whole agent failed.

The server below is a REAL streamable-HTTP MCP server (FastMCP + uvicorn on a
free port): mocks would not see the streamed, unread error body.
"""
import asyncio
import json
import logging
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP

from tilellm.controller.controller import handle_agent_exception
from tilellm.controller.controller_utils import get_all_filtered_tools
from tilellm.models.llm import QuestionToLLM
from tilellm.shared.mcp_tool_calls import FixedToolArguments, tool_errors_to_result

PROFILE_URL = "https://shop-esempio.example/profiles/agent.json"
SECRET_HEADER_VALUE = "tok-segreto-0123456789"
UCP_ERROR = "UCP discovery failed: Missing profile uri"

received: list = []
wire_arguments: list = []  # raw tools/call arguments: FastMCP drops unknown ones before the tool


def _build_app():
    mcp = FastMCP("shop-esempio", stateless_http=True, json_response=True)

    @mcp.tool()
    def search_catalog(meta: dict, catalog: dict) -> str:
        """Search the shop catalog."""
        received.append(("search_catalog", {"meta": meta, "catalog": catalog}))
        return "found 3 products"

    @mcp.tool()
    def get_store_hours(day: str) -> str:
        """Opening hours."""
        received.append(("get_store_hours", {"day": day}))
        return "9-18"

    inner = mcp.streamable_http_app()

    async def app(scope, receive, send):
        """422 + JSON-RPC error on tools/call without the expected profile."""
        if scope["type"] != "http" or scope["method"] != "POST":
            return await inner(scope, receive, send)
        body = b""
        more = True
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
    server = uvicorn.Server(uvicorn.Config(_build_app(), host="127.0.0.1", port=port, log_level="warning"))
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
def _clear_received():
    received.clear()
    wire_arguments.clear()


def _question(url, parameters=None):
    server = {"transport": "streamable_http", "url": url, "headers": {"x-shop-token": SECRET_HEADER_VALUE}}
    if parameters is not None:
        server["parameters"] = parameters
    return QuestionToLLM(question="q", llm_key="sk-test", llm="openai", servers={"shop": server})


UCP_PARAMS = {"tool_arguments": {"meta": {"ucp-agent": {"profile": PROFILE_URL}}}}


async def _tools(question):
    client = question.create_mcp_client()
    tools = await get_all_filtered_tools(client, question.servers)
    return {t.name: t for t in tools}


@pytest.mark.asyncio
async def test_422_from_server_becomes_a_readable_tool_message(mcp_url):
    tools = await _tools(_question(mcp_url))

    result = await tools["search_catalog"].ainvoke({"meta": {}, "catalog": {"query": "scarpe"}})

    assert isinstance(result, str)
    assert "422" in result
    assert UCP_ERROR in result
    assert "search_catalog" in result
    assert "client=abc" not in result  # no URL query in what the model sees


@pytest.mark.asyncio
async def test_tool_arguments_are_merged_into_the_call(mcp_url):
    tools = await _tools(_question(mcp_url, UCP_PARAMS))

    result = await tools["search_catalog"].ainvoke({"catalog": {"query": "scarpe"}})

    assert "found 3 products" in str(result)
    assert received == [("search_catalog", {
        "meta": {"ucp-agent": {"profile": PROFILE_URL}},
        "catalog": {"query": "scarpe"},
    })]


@pytest.mark.asyncio
async def test_configured_key_replaces_the_model_value_entirely(mcp_url):
    """The model must not be able to add sub-keys under a configured key."""
    tools = await _tools(_question(mcp_url, UCP_PARAMS))

    await tools["search_catalog"].ainvoke({
        "meta": {"ucp-agent": {"profile": "https://inventato.example/x", "extra": 1}, "lang": "it"},
        "catalog": {"query": "scarpe"},
    })

    assert received[0][1]["meta"] == {"ucp-agent": {"profile": PROFILE_URL}}
    assert wire_arguments[0][1]["meta"] == {"ucp-agent": {"profile": PROFILE_URL}}
    assert received[0][1]["catalog"] == {"query": "scarpe"}  # keys not configured are untouched


@pytest.mark.asyncio
async def test_configured_keys_are_hidden_from_the_model_schema(mcp_url):
    question = _question(mcp_url, UCP_PARAMS)
    tools = await _tools(question)

    schema = tools["search_catalog"].args_schema
    assert "meta" not in schema["properties"]
    assert "meta" not in schema.get("required", [])
    assert "catalog" in schema["properties"] and "catalog" in schema["required"]

    # without config the schema is untouched
    plain = await _tools(_question(mcp_url))
    assert "meta" in plain["search_catalog"].args_schema["required"]


@pytest.mark.asyncio
async def test_tool_without_meta_in_schema_gets_no_extra_arguments(mcp_url):
    tools = await _tools(_question(mcp_url, UCP_PARAMS))

    await tools["get_store_hours"].ainvoke({"day": "monday"})

    assert wire_arguments == [("get_store_hours", {"day": "monday"})]
    assert received == [("get_store_hours", {"day": "monday"})]


def test_parameters_are_not_passed_to_the_connection():
    question = _question("https://shop-esempio.example/mcp", UCP_PARAMS)

    client = question.create_mcp_client()

    assert "parameters" not in client.connections["shop"]


def test_agent_exception_group_shows_root_cause():
    request = httpx.Request("POST", "https://shop-esempio.example/mcp?key=abc")
    response = httpx.Response(422, request=request, json={"jsonrpc": "2.0", "error": {"code": 1, "message": UCP_ERROR}})
    root = httpx.HTTPStatusError("Client error '422'", request=request, response=response)
    group = BaseExceptionGroup("unhandled errors in a TaskGroup", [root])

    body = json.loads(handle_agent_exception(group, "test").body)

    assert "TaskGroup" not in body["answer"]
    assert "422" in body["answer"] and UCP_ERROR in body["answer"]
    assert "key=abc" not in body["answer"]


@pytest.mark.asyncio
async def test_header_values_never_reach_the_logs(mcp_url, caplog):
    caplog.set_level(logging.DEBUG)
    tools = await _tools(_question(mcp_url))

    await tools["search_catalog"].ainvoke({"meta": {}, "catalog": {}})

    assert "x-shop-token" in caplog.text  # header names are fine
    assert SECRET_HEADER_VALUE not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [
    asyncio.CancelledError(),
    BaseExceptionGroup("unhandled errors in a TaskGroup", [asyncio.CancelledError()]),
])
async def test_cancellation_is_not_turned_into_a_tool_message(exc):
    async def handler(_request):
        raise exc

    with pytest.raises(BaseException) as raised:
        await tool_errors_to_result(SimpleNamespace(name="search_catalog", server_name="shop"), handler)
    assert raised.value is exc


@pytest.mark.asyncio
async def test_tool_arguments_no_tool_declares_are_warned(mcp_url, caplog):
    caplog.set_level(logging.DEBUG)
    tools = await _tools(_question(mcp_url, {"tool_arguments": {"sconosciuta": "VALORE-NON-LOGGARE"}}))

    assert "meta" in tools["search_catalog"].args_schema["properties"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("shop" in m and "sconosciuta" in m for m in warnings), warnings
    assert "VALORE-NON-LOGGARE" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [["meta"], "meta", 42])
async def test_tool_arguments_not_a_dict_is_ignored_without_losing_tools(mcp_url, caplog, bad):
    caplog.set_level(logging.DEBUG)
    tools = await _tools(_question(mcp_url, {"tool_arguments": bad}))

    assert set(tools) == {"search_catalog", "get_store_hours"}
    assert "meta" in tools["search_catalog"].args_schema["required"]
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("tool_arguments" in m and "shop" in m for m in errors), errors


@pytest.mark.asyncio
async def test_error_inside_tool_arguments_interceptor_becomes_a_tool_message(mcp_url, monkeypatch):
    """Interceptor order: the error handler must wrap FixedToolArguments."""
    async def broken(self, request, handler):
        raise RuntimeError("interceptor rotto")

    monkeypatch.setattr(FixedToolArguments, "__call__", broken)
    tools = await _tools(_question(mcp_url, UCP_PARAMS))

    result = await tools["search_catalog"].ainvoke({"catalog": {"query": "scarpe"}})

    assert "search_catalog failed" in str(result) and "interceptor rotto" in str(result)
    assert received == []


def test_agent_error_detail_stays_in_the_log(caplog):
    caplog.set_level(logging.DEBUG)
    exc = RuntimeError("tool error: connessione a https://interno.example/x?key=SEGRETO-ERR fallita")

    response = handle_agent_exception(exc, "test")
    body = json.loads(response.body)

    assert response.status_code == 400  # classification unchanged
    assert "SEGRETO-ERR" not in body["answer"] and "interno.example" not in body["answer"]
    assert "SEGRETO-ERR" in caplog.text  # detail is for the operator


def test_agent_error_with_status_shows_only_the_status():
    class ProviderError(Exception):
        status_code = 429

    exc = ProviderError("rate limit for key sk-FINTA-123 on model x")

    body = json.loads(handle_agent_exception(exc, "test").body)

    assert "HTTP 429" in body["answer"]
    assert "sk-FINTA-123" not in body["answer"]
