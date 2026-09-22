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

The "vllm" provider slot is also used for any custom OpenAI-compatible endpoint
(e.g. Cerebras via a custom base_url), not just genuine self-hosted vLLM. A
strict backend that doesn't recognize chat_template_kwargs 400s instead of
ignoring it (Cerebras: "property 'chat_template_kwargs' is unsupported").
_VllmChatOpenAI (subclass used only for this provider slot) retries once
without extra_body on that specific error — self-healing, no caller action
needed. model.disable_thinking_mode=False remains as an explicit upfront
opt-out for callers who already know a backend will reject it.
"""
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import openai
import pytest
from pydantic import SecretStr

from tilellm.models.embedding import LlmEmbeddingModel
from tilellm.shared.utility import (
    ChatOpenAI,
    _build_llm_cache_key,
    _build_standard_llm_cache_key,
    _create_llm_instance,
    _create_standard_llm_instance,
    _VllmChatOpenAI,
)


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
         patch("tilellm.shared.utility._VllmChatOpenAI") as mock_chat:
        await _create_llm_instance(_vllm_question())

    _, kwargs = mock_chat.call_args
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_create_standard_llm_instance_disables_thinking_for_vllm():
    with patch("tilellm.shared.utility.get_llm_params", return_value={}), \
         patch("tilellm.shared.utility._get_llm_config_for_client", AsyncMock(return_value={"api_key": "k", "model": "qwen3"})), \
         patch("tilellm.shared.utility._VllmChatOpenAI") as mock_chat:
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


@pytest.mark.asyncio
async def test_create_llm_instance_honors_disable_thinking_mode_false():
    """Cerebras and similar strict OpenAI-compatible backends reject unknown
    body fields (400 on chat_template_kwargs) — model.disable_thinking_mode=False
    must skip the extra_body injection even though provider is still "vllm"."""
    q = _vllm_question()
    q.model.disable_thinking_mode = False

    with patch("tilellm.shared.utility.get_llm_params", return_value={}), \
         patch("tilellm.shared.utility._get_llm_config_for_client", AsyncMock(return_value={"api_key": "k", "model": "cerebras-model"})), \
         patch("tilellm.shared.utility._VllmChatOpenAI") as mock_chat:
        await _create_llm_instance(q)

    _, kwargs = mock_chat.call_args
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_create_standard_llm_instance_honors_disable_thinking_mode_false():
    q = _vllm_question()
    q.model.disable_thinking_mode = False

    with patch("tilellm.shared.utility.get_llm_params", return_value={}), \
         patch("tilellm.shared.utility._get_llm_config_for_client", AsyncMock(return_value={"api_key": "k", "model": "cerebras-model"})), \
         patch("tilellm.shared.utility._VllmChatOpenAI") as mock_chat:
        await _create_standard_llm_instance(q)

    _, kwargs = mock_chat.call_args
    assert "extra_body" not in kwargs


def _vllm_model(**overrides):
    kwargs = dict(
        provider="vllm",
        name="qwen3",
        api_key=SecretStr("sk-test"),
        url="https://api.cerebras.ai/v1",
    )
    kwargs.update(overrides)
    return LlmEmbeddingModel(**kwargs)


class _ChatCacheKeyQuestion:
    """Stand-in for inject_llm_chat_async's question (LlmEmbeddingModel path)."""

    def __init__(self, model):
        self.llm = "vllm"
        self.model = model


class _StandardCacheKeyQuestion:
    """Stand-in for inject_llm_async's question (has llm_key, not gptkey)."""

    def __init__(self, model):
        self.llm = "vllm"
        self.model = model
        self.llm_key = SecretStr("sk-test")


def test_llm_cache_key_differs_by_disable_thinking_mode():
    """A cached ChatOpenAI built before disable_thinking_mode=False was set must
    not be silently reused once the caller opts out — same TimedCache staleness
    bug class that let a broken client survive a Cerebras 400 fix."""
    default_key = asyncio.run(_build_llm_cache_key(_ChatCacheKeyQuestion(_vllm_model())))
    opted_out_key = asyncio.run(
        _build_llm_cache_key(_ChatCacheKeyQuestion(_vllm_model(disable_thinking_mode=False)))
    )
    assert default_key != opted_out_key


def test_standard_llm_cache_key_differs_by_disable_thinking_mode():
    default_key = asyncio.run(_build_standard_llm_cache_key(_StandardCacheKeyQuestion(_vllm_model())))
    opted_out_key = asyncio.run(
        _build_standard_llm_cache_key(_StandardCacheKeyQuestion(_vllm_model(disable_thinking_mode=False)))
    )
    assert default_key != opted_out_key


def _cerebras_bad_request_error(message="chat_template_kwargs: property 'chat_template_kwargs' is unsupported"):
    request = httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"message": message, "code": "wrong_api_format"})
    return openai.BadRequestError(message=message, response=response, body={"message": message, "code": "wrong_api_format"})


def _vllm_chat_openai(**overrides):
    kwargs = dict(
        api_key="k",
        model="cerebras-model",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    kwargs.update(overrides)
    return _VllmChatOpenAI(**kwargs)


@pytest.mark.asyncio
async def test_vllm_chat_openai_retries_without_extra_body_on_rejection():
    """The exact failure mode reported for Cerebras: 400 on chat_template_kwargs.
    Must self-heal with zero caller-side config — no disable_thinking_mode needed."""
    instance = _vllm_chat_openai()
    calls = []

    async def fake_agenerate(self, *args, **kwargs):
        calls.append(self.extra_body)
        if len(calls) == 1:
            raise _cerebras_bad_request_error()
        return "ok"

    with patch.object(ChatOpenAI, "_agenerate", fake_agenerate):
        result = await instance._agenerate([])

    assert result == "ok"
    assert calls == [{"chat_template_kwargs": {"enable_thinking": False}}, None]
    assert instance.extra_body is None  # stays off for subsequent calls on this cached instance


@pytest.mark.asyncio
async def test_vllm_chat_openai_reraises_unrelated_bad_request():
    """A 400 for an unrelated reason (bad api_key, malformed messages, ...) must not
    trigger a pointless retry — only the specific extra_body rejection does."""
    instance = _vllm_chat_openai()

    async def fake_agenerate(self, *args, **kwargs):
        raise _cerebras_bad_request_error(message="Invalid API key")

    with patch.object(ChatOpenAI, "_agenerate", fake_agenerate):
        with pytest.raises(openai.BadRequestError):
            await instance._agenerate([])

    assert instance.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}  # untouched


@pytest.mark.asyncio
async def test_vllm_chat_openai_no_retry_when_extra_body_already_none():
    """disable_thinking_mode=False already skipped extra_body upfront — a 400 here
    is unrelated to chat_template_kwargs and must propagate, not loop."""
    instance = _vllm_chat_openai(extra_body=None)
    calls = []

    async def fake_agenerate(self, *args, **kwargs):
        calls.append(1)
        raise _cerebras_bad_request_error()

    with patch.object(ChatOpenAI, "_agenerate", fake_agenerate):
        with pytest.raises(openai.BadRequestError):
            await instance._agenerate([])

    assert len(calls) == 1  # no retry attempted


@pytest.mark.asyncio
async def test_vllm_chat_openai_astream_retries_without_extra_body_on_rejection():
    instance = _vllm_chat_openai()
    calls = []

    async def fake_astream(self, *args, **kwargs):
        calls.append(self.extra_body)
        if len(calls) == 1:
            raise _cerebras_bad_request_error()
            yield  # pragma: no cover - makes this an async generator
        yield "chunk"

    with patch.object(ChatOpenAI, "_astream", fake_astream):
        chunks = [c async for c in instance._astream([])]

    assert chunks == ["chunk"]
    assert calls == [{"chat_template_kwargs": {"enable_thinking": False}}, None]
    assert instance.extra_body is None


@pytest.mark.asyncio
async def test_vllm_chat_openai_astream_does_not_retry_after_partial_yield():
    """The 400 happens before any chunk streams in practice (request-level
    validation) — but if a chunk somehow already streamed, retrying would
    duplicate content, so it must not retry in that case."""
    instance = _vllm_chat_openai()

    async def fake_astream(self, *args, **kwargs):
        yield "chunk"
        raise _cerebras_bad_request_error()

    with patch.object(ChatOpenAI, "_astream", fake_astream):
        with pytest.raises(openai.BadRequestError):
            _ = [c async for c in instance._astream([])]

    assert instance.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
