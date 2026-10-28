"""
The MCP server URL comes from the request payload: it passes the outbound URL
guard before any connection, so a request cannot point the service at cloud
metadata or loopback. A blocked server is skipped; the other servers still work.
"""
from unittest.mock import AsyncMock

import pytest

from tilellm.controller.controller_utils import get_all_filtered_tools
from tilellm.models.llm import QuestionToLLM


def _question(servers):
    return QuestionToLLM(question="ciao", llm="openai", llm_key="sk-FINTA", servers=servers)


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://127.0.0.1:6379/mcp",
    "http://localhost:8000/mcp",
])
async def test_blocked_mcp_url_is_never_contacted(url, monkeypatch):
    monkeypatch.delenv("OUTBOUND_URL_ALLOWED_HOSTS", raising=False)
    question = _question({"interno": {"transport": "streamable_http", "url": url}})
    client = question.create_mcp_client()
    client.get_tools = AsyncMock(return_value=[])

    tools = await get_all_filtered_tools(client, question.servers)

    assert tools == []
    client.get_tools.assert_not_called()


@pytest.mark.asyncio
async def test_other_servers_still_load(monkeypatch):
    monkeypatch.delenv("OUTBOUND_URL_ALLOWED_HOSTS", raising=False)
    question = _question({
        "interno": {"transport": "streamable_http", "url": "http://169.254.169.254/mcp"},
        "pubblico": {"transport": "streamable_http", "url": "https://mcp-esempio.example/mcp"},
    })
    client = question.create_mcp_client()
    client.get_tools = AsyncMock(return_value=[])

    await get_all_filtered_tools(client, question.servers)

    assert [c.kwargs.get("server_name") for c in client.get_tools.call_args_list] == ["pubblico"]
