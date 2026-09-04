"""OpenRouter provider routing.

OpenRouter is reached through the OpenAI-compatible client; what makes it
different is the request-body "provider" block that decides which upstream
provider actually serves the model. These tests pin that translation, the
filtering of values that arrive from the database, and the cache-key fragment
that keeps two differently-routed clients apart.
"""
import asyncio

import pytest
from pydantic import SecretStr

from tilellm.models.base import LLMEmbeddingProviders
from tilellm.models.embedding import LlmEmbeddingModel
from tilellm.shared.llm_config import (
    OPENROUTER_BASE_URL,
    PROVIDER_CONFIGS,
    build_openrouter_extra_body,
    get_llm_params,
)
from tilellm.shared.utility import _get_llm_config_for_client, _routing_cache_fragment


class _Question:
    """Minimal stand-in for the question objects the injectors receive."""

    def __init__(self, model, llm="openrouter"):
        self.llm = llm
        self.model = model
        self.llm_key = SecretStr("sk-or-fallback")
        self.temperature = 0.7
        self.top_p = None
        self.max_tokens = 512


def _client_config(model, llm="openrouter"):
    question = _Question(model, llm)
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
    )

    assert "extra_body" not in config
    assert config["base_url"] == ""  # LlmEmbeddingModel.url defaults to an empty string


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
