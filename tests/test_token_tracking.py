"""
Unit tests for tilellm.shared.token_tracking.

TDD: written before the implementation. Covers usage extraction across the
shapes LangChain returns (plain AIMessage, include_raw structured-output dict,
provider response_metadata fallback, missing usage) plus collector aggregation,
serialization shape, and always-on analytics emission.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tilellm.shared.token_tracking import (
    TokenUsageRecord,
    TokenUsageCollector,
    extract_token_usage,
    model_name_of,
    aggregate_token_usage,
)


def test_aggregate_token_usage_merges_blocks_and_skips_none():
    a = {"total": {"prompt": 700, "completion": 150, "total": 850},
         "calls": [{"op": "discretionary_judge", "model": "m", "prompt": 700, "completion": 150, "total": 850}]}
    b = {"total": {"prompt": 500, "completion": 150, "total": 650},
         "calls": [{"op": "compliance_judge", "model": "m", "prompt": 500, "completion": 150, "total": 650}]}
    out = aggregate_token_usage([a, None, b])
    assert out["total"] == {"prompt": 1200, "completion": 300, "total": 1500}
    assert len(out["calls"]) == 2


def test_aggregate_token_usage_empty():
    assert aggregate_token_usage(None) == {"total": {"prompt": 0, "completion": 0, "total": 0}, "calls": []}


def test_build_ingestion_collector_embedding_and_situated_context():
    from tilellm.shared.token_tracking import build_ingestion_collector
    c = build_ingestion_collector(
        embedding_model="text-embedding-3-small", embedding_tokens=1200,
        sc_model="gpt-4o-mini", sc_input_tokens=300, sc_output_tokens=80, sc_total_tokens=380,
    )
    out = c.to_dict()
    ops = {call["op"]: call for call in out["calls"]}
    assert ops["embedding"]["prompt"] == 1200 and ops["embedding"]["total"] == 1200
    assert ops["embedding"]["completion"] == 0
    assert ops["situated_context"]["prompt"] == 300
    assert ops["situated_context"]["completion"] == 80
    assert out["total"] == {"prompt": 1500, "completion": 80, "total": 1580}


def test_build_ingestion_collector_skips_zero_components():
    from tilellm.shared.token_tracking import build_ingestion_collector
    c = build_ingestion_collector(embedding_model="m", embedding_tokens=0)
    assert c.is_empty()


def test_model_name_of_handles_str_object_and_none():
    assert model_name_of("gpt-4o") == "gpt-4o"
    assert model_name_of(SimpleNamespace(name="custom-model")) == "custom-model"
    assert model_name_of(None) == ""


def _ai_message(usage_metadata=None, response_metadata=None):
    """Minimal AIMessage-like stand-in (we only read these two attributes)."""
    return SimpleNamespace(
        usage_metadata=usage_metadata,
        response_metadata=response_metadata or {},
    )


# ---------------------------------------------------------------------------
# extract_token_usage
# ---------------------------------------------------------------------------

def test_extract_from_aimessage_usage_metadata():
    msg = _ai_message(usage_metadata={
        "input_tokens": 700, "output_tokens": 150, "total_tokens": 850,
    })
    rec = extract_token_usage(msg, operation="discretionary_judge", model="gpt-4o")
    assert rec == TokenUsageRecord(
        operation="discretionary_judge", model="gpt-4o",
        prompt_tokens=700, completion_tokens=150, total_tokens=850, thinking_tokens=0,
    )


def test_extract_from_include_raw_structured_output():
    """with_structured_output(..., include_raw=True) returns {'raw','parsed','parsing_error'}."""
    raw = _ai_message(usage_metadata={
        "input_tokens": 500, "output_tokens": 150, "total_tokens": 650,
    })
    response = {"raw": raw, "parsed": object(), "parsing_error": None}
    rec = extract_token_usage(response, operation="legal_audit", model="gpt-4o")
    assert (rec.prompt_tokens, rec.completion_tokens, rec.total_tokens) == (500, 150, 650)


def test_extract_thinking_tokens_from_output_token_details():
    msg = _ai_message(usage_metadata={
        "input_tokens": 100, "output_tokens": 400, "total_tokens": 500,
        "output_token_details": {"reasoning": 250},
    })
    rec = extract_token_usage(msg, operation="ask", model="o1")
    assert rec.thinking_tokens == 250


def test_extract_total_defaults_to_sum_when_missing():
    msg = _ai_message(usage_metadata={"input_tokens": 10, "output_tokens": 5})
    rec = extract_token_usage(msg, operation="x", model="m")
    assert rec.total_tokens == 15


def test_extract_falls_back_to_response_metadata_openai_style():
    msg = _ai_message(usage_metadata=None, response_metadata={
        "token_usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
    })
    rec = extract_token_usage(msg, operation="x", model="m")
    assert (rec.prompt_tokens, rec.completion_tokens, rec.total_tokens) == (30, 20, 50)


def test_extract_missing_usage_returns_zeros():
    rec = extract_token_usage(_ai_message(), operation="x", model="m")
    assert (rec.prompt_tokens, rec.completion_tokens, rec.total_tokens, rec.thinking_tokens) == (0, 0, 0, 0)


def test_extract_handles_none_response():
    rec = extract_token_usage(None, operation="x", model="m")
    assert rec.total_tokens == 0


# ---------------------------------------------------------------------------
# TokenUsageCollector
# ---------------------------------------------------------------------------

def test_collector_starts_empty():
    c = TokenUsageCollector()
    assert c.is_empty()
    assert c.total() == {"prompt": 0, "completion": 0, "total": 0}


def test_collector_record_and_aggregate():
    c = TokenUsageCollector()
    c.record(_ai_message(usage_metadata={"input_tokens": 700, "output_tokens": 150, "total_tokens": 850}),
             operation="discretionary_judge", model="gpt-4o")
    c.record(_ai_message(usage_metadata={"input_tokens": 500, "output_tokens": 150, "total_tokens": 650}),
             operation="legal_audit", model="gpt-4o")
    assert not c.is_empty()
    assert c.total() == {"prompt": 1200, "completion": 300, "total": 1500}


def test_collector_to_dict_shape():
    c = TokenUsageCollector()
    c.record(_ai_message(usage_metadata={"input_tokens": 700, "output_tokens": 150, "total_tokens": 850}),
             operation="discretionary_judge", model="gpt-4o")
    out = c.to_dict()
    assert out["total"] == {"prompt": 700, "completion": 150, "total": 850}
    assert out["calls"] == [
        {"op": "discretionary_judge", "model": "gpt-4o", "prompt": 700, "completion": 150, "total": 850},
    ]


def test_collector_total_includes_thinking_only_when_present():
    c = TokenUsageCollector()
    c.record(_ai_message(usage_metadata={
        "input_tokens": 100, "output_tokens": 400, "total_tokens": 500,
        "output_token_details": {"reasoning": 250},
    }), operation="ask", model="o1")
    assert c.total()["thinking"] == 250
    assert c.to_dict()["calls"][0]["thinking"] == 250


def test_collector_merge():
    a = TokenUsageCollector()
    a.add(TokenUsageRecord("op1", "m", 10, 5, 15))
    b = TokenUsageCollector()
    b.add(TokenUsageRecord("op2", "m", 20, 10, 30))
    a.merge(b)
    assert len(a.calls) == 2
    assert a.total() == {"prompt": 30, "completion": 15, "total": 45}


# ---------------------------------------------------------------------------
# emit_analytics — always-on, one ai.token_usage event per call
# ---------------------------------------------------------------------------

def test_emit_analytics_publishes_one_event_per_call(monkeypatch):
    import tilellm.analytics as analytics
    from tilellm.shared import token_tracking

    published = []
    monkeypatch.setattr(analytics, "publish_nowait",
                        lambda et, idp, pl: published.append((et, idp, pl)))

    c = TokenUsageCollector()
    c.add(TokenUsageRecord("discretionary_judge", "gpt-4o", 700, 150, 850))
    c.add(TokenUsageRecord("legal_audit", "gpt-4o", 500, 150, 650))

    token_tracking.emit_analytics(c, id_project="proj-1", source="compliance")

    assert len(published) == 2
    assert all(et == "ai.token_usage" for et, _, _ in published)
    assert all(idp == "proj-1" for _, idp, _ in published)
    assert published[0][2]["operation"] == "discretionary_judge"
    assert published[0][2]["source"] == "compliance"
    assert published[0][2]["prompt_tokens"] == 700


def test_emit_analytics_noop_when_empty(monkeypatch):
    import tilellm.analytics as analytics
    from tilellm.shared import token_tracking

    published = []
    monkeypatch.setattr(analytics, "publish_nowait",
                        lambda et, idp, pl: published.append(1))
    token_tracking.emit_analytics(TokenUsageCollector(), id_project="p", source="compliance")
    assert published == []
