"""
get_all_filtered_tools must reject a header value containing a CR/LF (illegal
per RFC 7230) BEFORE attempting the MCP connection, instead of letting httpx/
h11 raise LocalProtocolError 40 stack frames deep inside an ExceptionGroup.

Real incident (2026-09-23, prod 0.12.3): a server's `headers` config carried
the full text of a customer email (with an "Attachments:" markdown list,
newlines included) instead of a short value — almost certainly an upstream
templating bug substituting the wrong variable into a header meant to hold
something small. get_all_filtered_tools already isolates per-server failures
(one bad server doesn't take down the whole /api/ask call); this fix keeps
that behavior but makes the failure fast and diagnosable instead of a deep
stack trace naming h11 internals instead of the offending header.
"""
from unittest.mock import AsyncMock

import pytest

from tilellm.controller.controller_utils import (
    _describe_mcp_connection_error,
    get_all_filtered_tools,
)
from tilellm.models.base import ServerConfig


def _config(headers=None, transport="streamable_http", url="https://example.com/mcp"):
    return ServerConfig(transport=transport, url=url, headers=headers)


@pytest.mark.asyncio
async def test_header_with_newline_is_rejected_without_a_network_attempt():
    """The exact real-world shape: a multi-line email body (with a markdown
    'Attachments:' list) ended up as a header value."""
    mcp_client = AsyncMock()
    servers = {
        "Tiledesk Data Table": _config(
            headers={"x-context": "Testo body email test\n> \n\n\nAttachments:\n[7.pdf](https://x/7.pdf)"}
        )
    }

    tools = await get_all_filtered_tools(mcp_client, servers)

    assert tools == []
    mcp_client.get_tools.assert_not_awaited()  # never attempted — rejected before the network call


@pytest.mark.asyncio
async def test_header_with_bare_cr_is_also_rejected():
    mcp_client = AsyncMock()
    servers = {"s": _config(headers={"x-id": "abc\rdef"})}

    tools = await get_all_filtered_tools(mcp_client, servers)

    assert tools == []
    mcp_client.get_tools.assert_not_awaited()


@pytest.mark.asyncio
async def test_clean_headers_still_reach_get_tools():
    """The check must not false-positive on ordinary header values."""
    mcp_client = AsyncMock()
    fake_tool = AsyncMock(name="x")
    fake_tool.name = "fake_tool"
    mcp_client.get_tools = AsyncMock(return_value=[fake_tool])
    servers = {"s": _config(headers={"x-composio-user-id": "abc-123"})}

    tools = await get_all_filtered_tools(mcp_client, servers)

    assert tools == [fake_tool]
    mcp_client.get_tools.assert_awaited_once_with(server_name="s")


@pytest.mark.asyncio
async def test_no_headers_at_all_still_works():
    mcp_client = AsyncMock()
    fake_tool = AsyncMock()
    fake_tool.name = "fake_tool"
    mcp_client.get_tools = AsyncMock(return_value=[fake_tool])
    servers = {"s": _config(headers=None)}

    tools = await get_all_filtered_tools(mcp_client, servers)

    assert tools == [fake_tool]


@pytest.mark.asyncio
async def test_one_bad_server_does_not_block_other_servers():
    """The existing per-server isolation must survive this fix unchanged."""
    mcp_client = AsyncMock()
    good_tool = AsyncMock()
    good_tool.name = "good_tool"
    mcp_client.get_tools = AsyncMock(return_value=[good_tool])
    servers = {
        "bad": _config(headers={"x": "line1\nline2"}),
        "good": _config(headers={"x": "clean-value"}),
    }

    tools = await get_all_filtered_tools(mcp_client, servers)

    assert tools == [good_tool]
    mcp_client.get_tools.assert_awaited_once_with(server_name="good")


def test_describe_mcp_connection_error_hints_at_illegal_header_value():
    """Defense in depth: if a control character ever slips past the proactive
    check (e.g. a future code path that builds an MCP client differently),
    the error log must still point at the real cause, not just h11 internals."""
    exc = Exception("httpx.LocalProtocolError: Illegal header value b'foo\\nbar'")

    hint = _describe_mcp_connection_error(exc)

    assert "header" in hint.lower()
    assert "ritorno a capo" in hint.lower() or "newline" in hint.lower()
