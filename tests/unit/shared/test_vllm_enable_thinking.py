#!/usr/bin/env python3
"""
docs/MIGLIORIE_DA_FARE.md P1#8: situated_context.py disables thinking mode for
vllm-served "thinking" models (Qwen3 and similar) via
extra_body={"chat_template_kwargs": {"enable_thinking": False}} — otherwise the
model spends all of max_tokens reasoning and returns empty content, silently
(e.g. FalkorDB community reports end up empty, no exception raised).

The two standard (non-reasoning) LLM builders in shared/utility.py,
_create_llm_instance (feeds inject_llm_chat_async — graph extraction/QA) and
_create_standard_llm_instance (feeds inject_llm_async), built ChatOpenAI for
vllm without this flag. Fixed by adding the same extra_body there. The
reasoning-dedicated builders (inject_reason_llm_async's vllm branch) are left
untouched on purpose — thinking must stay on for /api/thinking.
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tilellm.shared.utility import _create_llm_instance, _create_standard_llm_instance


def _vllm_question():
    q = Mock()
    q.llm = "vllm"
    q.model = Mock(provider=Mock(value="vllm"), url="http://vllm:8000/v1")
    q.temperature = 0.0
    q.top_p = 1.0
    q.max_tokens = 512
    q.thinking = None
    return q


@pytest.mark.asyncio
async def test_create_llm_instance_disables_thinking_for_vllm():
    with patch("tilellm.shared.utility.get_llm_params", return_value={}), \
         patch("tilellm.shared.utility._get_llm_config_for_client", AsyncMock(return_value={"api_key": "k", "model": "qwen3"})), \
         patch("langchain_openai.ChatOpenAI") as mock_chat:
        await _create_llm_instance(_vllm_question())

    _, kwargs = mock_chat.call_args
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_create_standard_llm_instance_disables_thinking_for_vllm():
    with patch("tilellm.shared.utility.get_llm_params", return_value={}), \
         patch("tilellm.shared.utility._get_llm_config_for_client", AsyncMock(return_value={"api_key": "k", "model": "qwen3"})), \
         patch("langchain_openai.ChatOpenAI") as mock_chat:
        await _create_standard_llm_instance(_vllm_question())

    _, kwargs = mock_chat.call_args
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_create_llm_instance_leaves_openai_untouched():
    q = _vllm_question()
    q.llm = "openai"
    q.model.provider.value = "openai"

    with patch("tilellm.shared.utility.get_llm_params", return_value={}), \
         patch("tilellm.shared.utility._get_llm_config_for_client", AsyncMock(return_value={"api_key": "k", "model": "gpt-4o-mini"})), \
         patch("langchain_openai.ChatOpenAI") as mock_chat:
        await _create_llm_instance(q)

    _, kwargs = mock_chat.call_args
    assert "extra_body" not in kwargs
