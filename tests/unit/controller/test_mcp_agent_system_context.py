#!/usr/bin/env python3
"""
Regression test for the MCP tool agent path dropping system_context and
chat_history_dict (Tiledesk/tiledesk-llm PR #6): ask_mcp_agent_llm_simple built
its system_prompt from MCP_BASE64_MANAGEMENT_TEMPLATE alone and fed the agent
only the current question, ignoring both question.system_context and
question.chat_history_dict. Mirrors what the non-MCP path already did.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tilellm.controller import controller as ctrl
from tilellm.models import QuestionToLLM
from tilellm.models.chat import ChatEntry


def _question(**overrides):
    kwargs = dict(
        question="What's the weather?",
        llm="openai",
        llm_key="test-key",
        system_context="You are Bob, a pirate-themed support bot.",
        chat_history_dict={"0": ChatEntry(question="Hi", answer="Ahoy!")},
    )
    kwargs.update(overrides)
    return QuestionToLLM(**kwargs)


@pytest.mark.asyncio
async def test_mcp_agent_simple_keeps_system_prompt_and_history():
    captured = {}

    async def fake_ainvoke(agent_input):
        captured["agent_input"] = agent_input
        return {"messages": agent_input["messages"]}

    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(side_effect=fake_ainvoke)

    def fake_create_agent(*, model, tools, system_prompt, middleware):
        captured["system_prompt"] = system_prompt
        return fake_agent

    question = _question()

    with patch.object(ctrl, "get_all_filtered_tools", AsyncMock(return_value=[])), \
         patch("langchain.agents.create_agent", side_effect=fake_create_agent):
        await ctrl.ask_mcp_agent_llm_simple(question, chat_model=MagicMock())

    assert "You are Bob, a pirate-themed support bot." in captured["system_prompt"]

    contents = [getattr(m, "content", m) for m in captured["agent_input"]["messages"]]
    assert any("Hi" == c for c in contents), contents
    assert any("Ahoy!" == c for c in contents), contents


@pytest.mark.asyncio
async def test_mcp_agent_simple_without_history_or_custom_context_still_works():
    captured = {}

    async def fake_ainvoke(agent_input):
        captured["agent_input"] = agent_input
        return {"messages": agent_input["messages"]}

    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(side_effect=fake_ainvoke)

    def fake_create_agent(*, model, tools, system_prompt, middleware):
        captured["system_prompt"] = system_prompt
        return fake_agent

    question = _question(system_context="", chat_history_dict=None)

    with patch.object(ctrl, "get_all_filtered_tools", AsyncMock(return_value=[])), \
         patch("langchain.agents.create_agent", side_effect=fake_create_agent):
        await ctrl.ask_mcp_agent_llm_simple(question, chat_model=MagicMock())

    # No caller system_context -> only the base64-management template.
    assert "MCP" in captured["system_prompt"] or "base64" in captured["system_prompt"].lower()
    assert captured["agent_input"]["messages"] == ["What's the weather?"]


@pytest.mark.asyncio
async def test_mcp_agent_complex_keeps_system_prompt_and_history():
    """Same regression on the multimodal path (ask_mcp_agent_llm, used when the
    question is a list): it kept system_context but sent only the current
    message, dropping chat_history_dict."""
    captured = {}

    async def fake_ainvoke(agent_input):
        captured["agent_input"] = agent_input
        return {"messages": agent_input["messages"]}

    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(side_effect=fake_ainvoke)

    def fake_create_agent(*, model, tools, system_prompt, middleware):
        captured["system_prompt"] = system_prompt
        return fake_agent

    question = _question(question='[{"type": "text", "text": "What\'s the weather?"}]')

    with patch.object(ctrl, "get_all_filtered_tools", AsyncMock(return_value=[])), \
         patch("langchain.agents.create_agent", side_effect=fake_create_agent):
        await ctrl.ask_mcp_agent_llm(question, chat_model=MagicMock())

    assert "You are Bob, a pirate-themed support bot." in captured["system_prompt"]

    contents = [getattr(m, "content", m) for m in captured["agent_input"]["messages"]]
    assert contents[0] == "Hi", contents
    assert contents[1] == "Ahoy!", contents
    assert len(contents) == 3, contents  # history + the current message, in order
