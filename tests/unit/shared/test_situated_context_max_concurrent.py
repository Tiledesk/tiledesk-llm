"""
SITUATED_CONTEXT_MAX_CONCURRENT (docs/MIGLIORIE_DA_FARE.md P1#9): the situated-
context enrichment concurrency was hardcoded to 5, uncontrollable, and — combined
with TASKIQ_MAX_ASYNC_TASKS — saturated RunPod on 2026-07-27. Now configurable
via env var, same pattern as PDF_MAX_CONCURRENT.
"""
import importlib
import inspect

import tilellm.shared.situated_context as situated_context


def test_default_matches_module_constant():
    """enrich_chunks_with_situated_context's max_concurrent default must stay
    wired to the module-level constant, not drift back to a bare literal."""
    sig = inspect.signature(situated_context.enrich_chunks_with_situated_context)
    assert sig.parameters["max_concurrent"].default == situated_context.SITUATED_CONTEXT_MAX_CONCURRENT


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("SITUATED_CONTEXT_MAX_CONCURRENT", "9")
    try:
        reloaded = importlib.reload(situated_context)
        assert reloaded.SITUATED_CONTEXT_MAX_CONCURRENT == 9
    finally:
        importlib.reload(situated_context)  # restore default env for the rest of the suite


def test_defaults_to_five_without_env_var(monkeypatch):
    monkeypatch.delenv("SITUATED_CONTEXT_MAX_CONCURRENT", raising=False)
    try:
        reloaded = importlib.reload(situated_context)
        assert reloaded.SITUATED_CONTEXT_MAX_CONCURRENT == 5
    finally:
        importlib.reload(situated_context)
