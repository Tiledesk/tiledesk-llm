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
