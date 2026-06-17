from tilellm.analytics import events


def test_kb_query_omits_success_when_unknown() -> None:
    event_type, payload = events.kb_query(
        kb_id="kb-1",
        kb_name="kb-1",
        query_text="how to reset password",
        chunks_retrieved=3,
        reranking_applied=False,
        latency_ms=120,
        request_id="req-1",
        success=None,
    )

    assert event_type == "kb.query_executed"
    assert "success" not in payload


def test_kb_query_includes_success_when_true() -> None:
    _, payload = events.kb_query(
        kb_id="kb-1",
        kb_name="kb-1",
        query_text="how to reset password",
        chunks_retrieved=3,
        reranking_applied=False,
        latency_ms=120,
        request_id="req-1",
        success=True,
    )

    assert payload["success"] is True


def test_kb_query_includes_success_when_false() -> None:
    _, payload = events.kb_query(
        kb_id="kb-1",
        kb_name="kb-1",
        query_text="how to reset password",
        chunks_retrieved=0,
        reranking_applied=False,
        latency_ms=120,
        request_id="req-1",
        success=False,
    )

    assert payload["success"] is False


def test_kb_query_keeps_agent_id_null_when_no_agent() -> None:
    # KB queries can legitimately run without an agent (e.g. /qa invoked by
    # tiledesk-server directly). agent_id must stay null, never coerced.
    _, payload = events.kb_query(
        kb_id="kb-1",
        kb_name="kb-1",
        query_text="how to reset password",
        chunks_retrieved=3,
        reranking_applied=False,
        latency_ms=120,
        request_id="req-1",
        agent_id=None,
    )

    assert "agent_id" in payload
    assert payload["agent_id"] is None


def test_model_call_coerces_missing_provider_to_unknown() -> None:
    # The ai.model_call contract requires provider to be a non-null string.
    # Callers pass getattr(question, "llm", None), which can be None.
    _, payload = events.model_call(
        model="gpt-4o",
        provider=None,  # type: ignore[arg-type]
        operation="ask",
        latency_ms=42,
        success=True,
    )

    assert payload["provider"] == "unknown"


def test_model_call_coerces_empty_provider_to_unknown() -> None:
    _, payload = events.model_call(
        model="gpt-4o",
        provider="",
        operation="ask",
        latency_ms=42,
        success=True,
    )

    assert payload["provider"] == "unknown"


def test_model_call_preserves_known_provider() -> None:
    _, payload = events.model_call(
        model="gpt-4o",
        provider="openai",
        operation="ask",
        latency_ms=42,
        success=True,
    )

    assert payload["provider"] == "openai"
