#!/usr/bin/env python3
"""
Claude Opus 5, Claude Sonnet 5, Claude Fable 5, Opus 4.8 and Opus 4.7 reject an
explicit `temperature`/`top_p` outright (HTTP 400 "temperature is deprecated
for this model") — see the Anthropic model migration notes. get_llm_params()
always included temperature/top_p for provider="anthropic" regardless of which
Claude model was targeted, so every ChatAnthropic() builder in
shared/utility.py and shared/situated_context.py broke on these models.

Fixed with a single choke point, strip_unsupported_anthropic_sampling_params()
in shared/llm_config.py, called right before each ChatAnthropic(**client_config)
construction (7 call sites) — the model string is already resolved at that
point in every builder.
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tilellm.shared.llm_config import strip_unsupported_anthropic_sampling_params
from tilellm.shared.utility import _create_llm_instance, _create_standard_llm_instance


def test_strips_temperature_and_top_p_for_no_sampling_models():
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7"):
        params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}
        strip_unsupported_anthropic_sampling_params(model, params)
        assert "temperature" not in params, model
        assert "top_p" not in params, model
        assert params["max_tokens"] == 512, model


def test_leaves_temperature_untouched_for_older_claude_models():
    params = {"temperature": 0.7, "top_p": 0.9}
    strip_unsupported_anthropic_sampling_params("claude-opus-4-6", params)
    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.9


def _anthropic_question(model="claude-opus-5"):
    q = Mock()
    q.llm = "anthropic"
    q.model = Mock(provider=Mock(value="anthropic"))
    q.temperature = 0.7
    q.top_p = 0.9
    q.max_tokens = 512
    q.thinking = None
    return q


@pytest.mark.asyncio
async def test_create_llm_instance_drops_temperature_for_opus_5():
    with patch("tilellm.shared.utility.get_llm_params", return_value={"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}), \
         patch("tilellm.shared.utility._get_llm_config_for_client", AsyncMock(return_value={"api_key": "k", "model": "claude-opus-5"})), \
         patch("langchain_anthropic.ChatAnthropic") as mock_chat:
        await _create_llm_instance(_anthropic_question())

    _, kwargs = mock_chat.call_args
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert kwargs["model"] == "claude-opus-5"


@pytest.mark.asyncio
async def test_create_standard_llm_instance_drops_temperature_for_opus_5():
    with patch("tilellm.shared.utility.get_llm_params", return_value={"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}), \
         patch("tilellm.shared.utility._get_llm_config_for_client", AsyncMock(return_value={"api_key": "k", "model": "claude-opus-5"})), \
         patch("langchain_anthropic.ChatAnthropic") as mock_chat:
        await _create_standard_llm_instance(_anthropic_question())

    _, kwargs = mock_chat.call_args
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


@pytest.mark.asyncio
async def test_create_llm_instance_keeps_temperature_for_older_claude_model():
    with patch("tilellm.shared.utility.get_llm_params", return_value={"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}), \
         patch("tilellm.shared.utility._get_llm_config_for_client", AsyncMock(return_value={"api_key": "k", "model": "claude-opus-4-6"})), \
         patch("langchain_anthropic.ChatAnthropic") as mock_chat:
        await _create_llm_instance(_anthropic_question(model="claude-opus-4-6"))

    _, kwargs = mock_chat.call_args
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
