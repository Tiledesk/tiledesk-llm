"""
An MCP server config arrives in the request body (`QuestionToLLM.servers`).
With transport "stdio" the MCP client spawns `command args` as a process inside
our container: whoever controls the payload runs code on the pod. Only network
transports are accepted.
"""
import pytest
from pydantic import ValidationError

from tilellm.models.base import ServerConfig


@pytest.mark.parametrize("transport", ["stdio", "websocket", "STDIO", ""])
def test_non_network_transports_are_rejected(transport):
    with pytest.raises(ValidationError, match="transport"):
        ServerConfig(transport=transport, url="https://example.com/mcp", command="sh", args=["-c", "id"])


@pytest.mark.parametrize("transport", ["streamable_http", "sse"])
def test_network_transports_are_accepted(transport):
    assert ServerConfig(transport=transport, url="https://example.com/mcp").transport == transport


def test_command_is_never_forwarded_to_the_mcp_client():
    cfg = ServerConfig(transport="streamable_http", url="https://example.com/mcp", command="sh", args=["-c", "id"])

    dumped = cfg.model_dump(exclude_unset=True, exclude={"enabled_tools"})

    assert "command" not in dumped and "args" not in dumped
