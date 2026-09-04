"""OpenRouter provider routing and reasoning.

OpenRouter is reached through the OpenAI-compatible client; what makes it
different is what travels in the request body: the "provider" block deciding
which upstream provider serves the model, and the unified "reasoning" block.
Both live under extra_body, so they have to be merged rather than assigned.

These tests pin that translation, the filtering of values that arrive from the
database, and the cache-key fragments that keep two clients apart when only
the request body differs.
"""
import asyncio

import pytest
from pydantic import SecretStr

from tilellm.models.base import LLMEmbeddingProviders
from tilellm.models.embedding import LlmEmbeddingModel
from tilellm.models.schemas.general_schemas import ReasoningConfig
from tilellm.shared.llm_config import (
    OPENROUTER_BASE_URL,
    PROVIDER_CONFIGS,
    build_openrouter_extra_body,
    build_openrouter_reasoning,
    get_llm_params,
)
from tilellm.shared.utility import (
    _build_llm_cache_key,
    _get_llm_config_for_client,
    _routing_cache_fragment,
)


class _Question:
    """Minimal stand-in for the question objects the injectors receive."""

    def __init__(self, model, llm="openrouter", thinking=None):
        self.llm = llm
        self.model = model
        self.llm_key = SecretStr("sk-or-fallback")
        self.temperature = 0.7
        self.top_p = None
        self.max_tokens = 512
        self.thinking = thinking


def _client_config(model, llm="openrouter", thinking=None):
    question = _Question(model, llm, thinking)
    params = get_llm_params(llm, question.temperature, question.top_p, question.max_tokens)
    return asyncio.run(_get_llm_config_for_client(question, dict(params)))


def _model(**kwargs):
    kwargs.setdefault("provider", "openrouter")
    kwargs.setdefault("name", "openai/gpt-4o")
    kwargs.setdefault("api_key", SecretStr("sk-or-abc"))
    return LlmEmbeddingModel(**kwargs)


def test_openrouter_is_a_known_provider():
    assert LLMEmbeddingProviders.OPENROUTER.value == "openrouter"
    assert "openrouter" in PROVIDER_CONFIGS


def test_routing_becomes_the_provider_block():
    config = _client_config(_model(provider_routing={
        "order": ["azure", "openai"],
        "allow_fallbacks": False,
        "sort": "price",
    }))

    assert config["base_url"] == OPENROUTER_BASE_URL
    assert config["model"] == "openai/gpt-4o"
    assert config["extra_body"] == {
        "provider": {"order": ["azure", "openai"], "allow_fallbacks": False, "sort": "price"}
    }


def test_model_without_routing_carries_no_extra_body():
    """An unconfigured model must behave like any other OpenAI-compatible provider."""
    config = _client_config(_model())

    assert config["base_url"] == OPENROUTER_BASE_URL
    assert "extra_body" not in config


def test_explicit_url_wins_over_the_default_endpoint():
    config = _client_config(_model(url="https://gateway.internal/v1"))

    assert config["base_url"] == "https://gateway.internal/v1"


def test_other_providers_are_untouched():
    config = _client_config(
        LlmEmbeddingModel(provider="openai", name="gpt-4o", api_key=SecretStr("sk-1")),
        llm="openai",
        thinking=ReasoningConfig(reasoning_effort="high"),
    )

    assert "extra_body" not in config
    assert config["base_url"] == ""  # LlmEmbeddingModel.url defaults to an empty string


@pytest.mark.parametrize("thinking,expected", [
    (None, None),
    (ReasoningConfig(), None),
    (ReasoningConfig(reasoning_effort="high"), {"reasoning": {"effort": "high"}}),
    (ReasoningConfig(budget_tokens=4096), {"reasoning": {"max_tokens": 4096}}),
    (ReasoningConfig(type="disabled"), {"reasoning": {"enabled": False}}),
    (ReasoningConfig(thinkingBudget=8000), {"reasoning": {"max_tokens": 8000}}),
    (ReasoningConfig(thinkingBudget=0), {"reasoning": {"enabled": False}}),
    (ReasoningConfig(thinkingBudget=-1), None),  # Gemini's "dynamic", not a budget
    (ReasoningConfig(thinkingLevel="low"), {"reasoning": {"effort": "low"}}),
])
def test_every_provider_dialect_of_reasoning_maps_to_openrouters(thinking, expected):
    """ReasoningConfig has one field per provider; OpenRouter fronts them all."""
    assert build_openrouter_reasoning(thinking) == expected


def test_routing_and_reasoning_share_extra_body_without_clobbering():
    """Both are keys of the same object: assigning one must not drop the other."""
    config = _client_config(
        _model(provider_routing={"order": ["azure", "openai"], "sort": "price"}),
        thinking=ReasoningConfig(reasoning_effort="high"),
    )

    assert config["extra_body"] == {
        "provider": {"order": ["azure", "openai"], "sort": "price"},
        "reasoning": {"effort": "high"},
    }


def test_reasoning_alone_still_reaches_the_body():
    config = _client_config(_model(), thinking=ReasoningConfig(budget_tokens=2048))

    assert config["extra_body"] == {"reasoning": {"max_tokens": 2048}}


def test_cache_fragment_separates_differently_reasoning_clients():
    routing = {"order": ["azure"]}
    low = _client_config(_model(provider_routing=routing), thinking=ReasoningConfig(reasoning_effort="low"))
    high = _client_config(_model(provider_routing=routing), thinking=ReasoningConfig(reasoning_effort="high"))

    assert _routing_cache_fragment(low) != _routing_cache_fragment(high)


def test_the_chat_object_cache_key_separates_routing():
    """_build_llm_cache_key keys the inject_llm_async path, which builds OpenRouter clients."""
    azure = asyncio.run(_build_llm_cache_key(_Question(_model(provider_routing={"order": ["azure"]}))))
    fireworks = asyncio.run(_build_llm_cache_key(_Question(_model(provider_routing={"order": ["fireworks"]}))))
    unrouted = asyncio.run(_build_llm_cache_key(_Question(_model())))

    assert azure != fireworks
    assert not any("openrouter_body" in str(part) for part in unrouted)


@pytest.mark.parametrize("routing,expected", [
    (None, None),
    ({}, None),
    ({"order": []}, None),
    ({"sort": "bogus"}, None),
    ({"sort": "PRICE"}, {"provider": {"sort": "price"}}),
    ({"order": ["  ", "fireworks"]}, {"provider": {"order": ["fireworks"]}}),
    ({"unknown_key": "dropped", "order": ["a"]}, {"provider": {"order": ["a"]}}),
    ({"allow_fallbacks": False}, {"provider": {"allow_fallbacks": False}}),
])
def test_routing_values_from_the_database_are_filtered(routing, expected):
    """The value is stored by the dashboard, so only known-good keys are forwarded."""
    assert build_openrouter_extra_body(routing) == expected


def test_cache_fragment_separates_differently_routed_clients():
    """Same model and same key, different providers: these must not share a client."""
    azure = _client_config(_model(provider_routing={"order": ["azure"]}))
    fireworks = _client_config(_model(provider_routing={"order": ["fireworks"]}))
    unrouted = _client_config(_model())

    assert _routing_cache_fragment(azure) != _routing_cache_fragment(fireworks)
    assert _routing_cache_fragment(unrouted) is None


def test_the_config_is_accepted_by_the_client():
    from langchain_openai import ChatOpenAI

    config = _client_config(_model(provider_routing={"order": ["azure"], "sort": "latency"}))
    client = ChatOpenAI(**config)

    assert client.extra_body == {"provider": {"order": ["azure"], "sort": "latency"}}
