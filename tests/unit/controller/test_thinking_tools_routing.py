#!/usr/bin/env python3
"""
Routing tests: /api/thinking must support MCP servers + internal tools exactly
like /api/ask, using the reasoning-capable chat model.

The tool path is shared: both endpoints funnel into `_dispatch_mcp_agent`
(simple-vs-complex), differing only in which model injector builds `chat_model`.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

import tilellm.__main__ as main_mod
from tilellm.controller import controller as ctrl
from tilellm.models import QuestionToLLM
from tilellm.models.base import ServerConfig


def _q(question, servers=None, tools=None):
    return QuestionToLLM(
        question=question,
        llm="deepseek",
        llm_key="test-key",
        servers=servers or {},
        tools=tools,
    )


_MCP = {"srv": ServerConfig(transport="sse", url="http://localhost:9999/sse")}


# ---------------------------------------------------------------------------
# _dispatch_mcp_agent: simple string -> simple core; list/json-list -> complex core
# ---------------------------------------------------------------------------

class TestDispatchMcpAgent:
    @pytest.mark.asyncio
    async def test_simple_string_calls_simple_core(self):
        simple = AsyncMock(return_value="SIMPLE")
        complex_ = AsyncMock(return_value="COMPLEX")
        with patch.object(ctrl, "ask_mcp_agent_llm_simple", simple), \
             patch.object(ctrl, "ask_mcp_agent_llm", complex_):
            res = await ctrl._dispatch_mcp_agent(_q("ciao", servers=_MCP), chat_model="M")
        assert res == "SIMPLE"
        simple.assert_awaited_once()
        complex_.assert_not_called()

    @pytest.mark.asyncio
    async def test_json_list_string_calls_complex_core(self):
        payload = json.dumps([{"type": "text", "text": "hi"}])
        simple = AsyncMock(return_value="SIMPLE")
        complex_ = AsyncMock(return_value="COMPLEX")
        with patch.object(ctrl, "ask_mcp_agent_llm_simple", simple), \
             patch.object(ctrl, "ask_mcp_agent_llm", complex_):
            res = await ctrl._dispatch_mcp_agent(_q(payload, servers=_MCP), chat_model="M")
        assert res == "COMPLEX"
        complex_.assert_awaited_once()
        simple.assert_not_called()

    @pytest.mark.asyncio
    async def test_native_list_calls_complex_core(self):
        q = _q([{"type": "text", "text": "hi"}], servers=_MCP)
        simple = AsyncMock(return_value="SIMPLE")
        complex_ = AsyncMock(return_value="COMPLEX")
        with patch.object(ctrl, "ask_mcp_agent_llm_simple", simple), \
             patch.object(ctrl, "ask_mcp_agent_llm", complex_):
            res = await ctrl._dispatch_mcp_agent(q, chat_model="M")
        assert res == "COMPLEX"
        complex_.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_model_forwarded_to_core(self):
        simple = AsyncMock(return_value="SIMPLE")
        with patch.object(ctrl, "ask_mcp_agent_llm_simple", simple):
            await ctrl._dispatch_mcp_agent(_q("ciao", servers=_MCP), chat_model="REASONING_MODEL")
        # core must receive the already-injected model, not build its own
        _, kwargs = simple.call_args
        passed = simple.call_args.args + tuple(kwargs.values())
        assert "REASONING_MODEL" in passed


# ---------------------------------------------------------------------------
# /api/thinking routing: tools -> reasoning MCP agent; none -> plain reasoning
# ---------------------------------------------------------------------------

class TestThinkingEndpointRouting:
    @pytest.mark.asyncio
    async def test_no_tools_uses_plain_reasoning(self):
        reason = AsyncMock(return_value="REASON")
        agent = AsyncMock(return_value="AGENT")
        with patch.object(main_mod, "ask_reason_llm", reason), \
             patch.object(main_mod, "ask_mcp_agent_reason", agent):
            res = await main_mod.post_ask_to_llm_reason_main(_q("ciao"))
        assert res == "REASON"
        reason.assert_awaited_once()
        agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_servers_use_reasoning_agent(self):
        reason = AsyncMock(return_value="REASON")
        agent = AsyncMock(return_value="AGENT")
        with patch.object(main_mod, "ask_reason_llm", reason), \
             patch.object(main_mod, "ask_mcp_agent_reason", agent):
            res = await main_mod.post_ask_to_llm_reason_main(_q("ciao", servers=_MCP))
        assert res == "AGENT"
        agent.assert_awaited_once()
        reason.assert_not_called()

    @pytest.mark.asyncio
    async def test_internal_tools_use_reasoning_agent(self):
        reason = AsyncMock(return_value="REASON")
        agent = AsyncMock(return_value="AGENT")
        with patch.object(main_mod, "ask_reason_llm", reason), \
             patch.object(main_mod, "ask_mcp_agent_reason", agent):
            res = await main_mod.post_ask_to_llm_reason_main(_q("ciao", tools=["calculator"]))
        assert res == "AGENT"
        agent.assert_awaited_once()


# ---------------------------------------------------------------------------
# /api/ask regression: same tool path, standard (non-reasoning) injector
# ---------------------------------------------------------------------------

class TestAskEndpointRegression:
    @pytest.mark.asyncio
    async def test_no_tools_uses_plain_ask(self):
        plain = AsyncMock(return_value="ASK")
        agent = AsyncMock(return_value="AGENT")
        with patch.object(main_mod, "ask_to_llm", plain), \
             patch.object(main_mod, "ask_mcp_agent", agent):
            res = await main_mod.post_ask_to_llm_main(_q("ciao"))
        assert res == "ASK"
        plain.assert_awaited_once()
        agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_servers_use_standard_agent(self):
        plain = AsyncMock(return_value="ASK")
        agent = AsyncMock(return_value="AGENT")
        with patch.object(main_mod, "ask_to_llm", plain), \
             patch.object(main_mod, "ask_mcp_agent", agent):
            res = await main_mod.post_ask_to_llm_main(_q("ciao", servers=_MCP))
        assert res == "AGENT"
        agent.assert_awaited_once()
        plain.assert_not_called()
