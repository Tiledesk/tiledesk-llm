#!/usr/bin/env python3
"""
Ogni client LLM deve avere un timeout di richiesta esplicito e regolabile.

Da chiarire subito, perche' e' facile raccontarsela: NON stiamo proteggendo da
richieste appese all'infinito. Verificato leggendo la libreria: openai applica
gia' di suo `Timeout(connect=5s, read=600s)` con `max_retries=2`, quindi una
connessione morta viene rilevata comunque. Durante il blackout di rete del
2026-07-30 il worker emetteva infatti APIConnectionError e ritentava a due
livelli (openai._base_client + i 4 tentativi di graphrag_extractor).

Il punto e' un altro: quei 600s di read timeout sono impliciti, molto generosi
per una singola chiamata di estrazione, e non tarabili senza toccare il codice.
Sommati ai retry sovrapposti, una singola chunk puo' tenere occupato uno slot di
concorrenza per decine di minuti. Rendere il valore esplicito e configurabile
via LLM_REQUEST_TIMEOUT_S permette di stringere il caso peggiore.

Due percorsi distinti da coprire, perche' non condividono codice:
  utility.get_llm_params            -> QA, estrazione GraphRAG, ecc. (6 call-site)
  situated_context.build_llm_from_config -> ingestion
"""
import pytest


class TestGetLlmParamsInjectsTimeout:
    """get_llm_params e' l'unico costruttore dei parametri client in utility.py
    (6 call-site): il timeout messo qui raggiunge tutti i ChatOpenAI."""

    def test_timeout_present_by_default(self):
        from tilellm.shared.llm_config import get_llm_params

        params = get_llm_params(provider="vllm", temperature=0.0, top_p=1.0, max_tokens=1024)

        assert "timeout" in params, "nessun timeout: la richiesta puo' restare appesa per sempre"
        assert params["timeout"] > 0

    def test_timeout_is_configurable_via_env(self, monkeypatch):
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT_S", "42")
        import importlib
        import tilellm.shared.llm_config as m
        importlib.reload(m)
        try:
            assert m.get_llm_params(provider="openai", temperature=0.0, top_p=1.0,
                                    max_tokens=100)["timeout"] == 42.0
        finally:
            monkeypatch.delenv("LLM_REQUEST_TIMEOUT_S", raising=False)
            importlib.reload(m)

    def test_explicit_timeout_is_not_overridden(self):
        from tilellm.shared.llm_config import get_llm_params

        params = get_llm_params(provider="openai", temperature=0.0, top_p=1.0,
                                max_tokens=100, timeout=7)
        assert params["timeout"] == 7

    @pytest.mark.parametrize("provider", ["openai", "vllm", "anthropic", "google", "cohere", "sconosciuto"])
    def test_every_provider_gets_a_timeout(self, provider):
        from tilellm.shared.llm_config import get_llm_params

        params = get_llm_params(provider=provider, temperature=0.0, top_p=1.0, max_tokens=100)
        assert params.get("timeout"), f"provider {provider} senza timeout"


class TestSituatedContextClientHasTimeout:
    """Percorso separato da utility.py: build_llm_from_config costruisce il suo
    ChatOpenAI e non passa da get_llm_params, quindi va coperto a parte."""

    @pytest.mark.asyncio
    async def test_chat_openai_built_with_timeout(self):
        from tilellm.models.llm import SituatedContextConfig
        from tilellm.shared.situated_context import build_llm_from_config

        llm = await build_llm_from_config(
            SituatedContextConfig(enable=True, provider="vllm", model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                                  api_key="pippo", url="https://example-8000.proxy.runpod.net/v1"),
            fallback_api_key=None,
        )

        assert llm is not None
        timeout = getattr(llm, "request_timeout", None) or getattr(llm, "timeout", None)
        assert timeout, "il client situated_context resta sul default openai (600s), non tarabile"
