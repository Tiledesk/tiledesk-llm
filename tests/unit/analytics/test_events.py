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


def test_token_usage_includes_kb_id_for_rag_source() -> None:
    # rag-source token usage carries kb_id so KB-token analytics can attribute
    # tokens to a knowledge base directly, without joining on request_id.
    event_type, payload = events.token_usage(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        operation="ask",
        source="rag",
        request_id=None,
        agent_id=None,
        kb_id="my-namespace",
    )

    assert event_type == "ai.token_usage"
    assert payload["kb_id"] == "my-namespace"


def test_token_usage_keeps_kb_id_null_for_chat_source() -> None:
    # Non-KB (e.g. source='chat') usage has no knowledge base; kb_id stays null.
    _, payload = events.token_usage(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        operation="chat",
        source="chat",
    )

    assert "kb_id" in payload
    assert payload["kb_id"] is None


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
