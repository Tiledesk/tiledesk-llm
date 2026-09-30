"""
Cached chat clients must not leak one request's generation parameters into another.

Reported on 0.12.3-rc5 with vLLM: a first request with max_tokens=10000 builds and
caches a ChatOpenAI client with max_tokens=10000 baked in. A later request for the same
model, key and URL with max_tokens=2000 got that cached client back — the cache key
held provider/model/api-key/url but not max_tokens — so vLLM still received 10000 and
rejected it (max_tokens + prompt > the model's context). temperature and top_p are baked
into the client the same way.
"""
import asyncio

import pytest
from pydantic import SecretStr

from tilellm.models.embedding import LlmEmbeddingModel
from tilellm.models import Engine
from tilellm.models.llm import QuestionAnswer, QuestionToLLM
from tilellm.shared.utility import (
    _build_llm_cache_key,
    _build_reasoning_llm_cache_key,
    _build_standard_llm_cache_key,
)


def _vllm_model():
    return LlmEmbeddingModel(provider="vllm", name="qwen", url="http://vllm:8000/v1", api_key="k")


def _to_llm(**over):
    kw = dict(question="q", llm="vllm", llm_key=SecretStr("k"), model=_vllm_model(), max_tokens=10000)
    kw.update(over)
    return QuestionToLLM(**kw)


def _qa(**over):
    kw = dict(question="q", namespace="ns", gptkey=SecretStr("k"), llm="vllm", model=_vllm_model(),
              max_tokens=10000, engine=Engine(name="pinecone", type="serverless", apikey="k",
                                               vector_size=1024, index_name="i"))
    kw.update(over)
    return QuestionAnswer(**kw)


BUILDERS = [
    pytest.param(_build_standard_llm_cache_key, _to_llm, id="inject_llm_async"),
    pytest.param(_build_reasoning_llm_cache_key, _to_llm, id="inject_reason_llm_async"),
    pytest.param(_build_llm_cache_key, _qa, id="inject_llm_chat_async"),
]


@pytest.mark.parametrize("build, question", BUILDERS)
@pytest.mark.parametrize("field, other", [("max_tokens", 2000), ("temperature", 0.7), ("top_p", 0.5)])
def test_generation_params_change_the_cache_key(build, question, field, other):
    first = asyncio.run(build(question()))
    second = asyncio.run(build(question(**{field: other})))

    assert first != second


@pytest.mark.parametrize("build, question", BUILDERS)
def test_identical_requests_still_share_the_cached_client(build, question):
    assert asyncio.run(build(question())) == asyncio.run(build(question()))
