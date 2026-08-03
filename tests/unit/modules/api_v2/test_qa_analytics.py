#!/usr/bin/env python3
"""
api_v2 analytics instrumentation (docs/MIGLIORIE_DA_FARE.md P1#14):
/api/v2/query and /api/v2/qa emit kb.query_executed (mirrors /api/qa in
__main__.py). token_usage/model_call are NOT duplicated at this layer: both
graphs route through rag_node -> ask_with_memory/ask_hybrid_with_memory
(controller.py), which already emit them per LLM call.
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.responses import JSONResponse

from tilellm.models.vector_store import Engine


def _engine():
    return Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx")


class TestEmitApiV2KbQuery:
    def test_success_extracts_chunk_count(self):
        from tilellm.modules.api_v2.controllers import _emit_api_v2_kb_query
        from tilellm.models import QuestionAnswer

        payload = QuestionAnswer(question="ciao", namespace="ns", engine=_engine(), id_project="proj1")
        response = JSONResponse(content={"success": True, "content_chunks": ["a", "b", "c"]})

        with patch("tilellm.modules.api_v2.controllers.analytics") as mock_analytics:
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {"fake": "payload"})
            _emit_api_v2_kb_query(payload, response, 123)

        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 3
        assert kwargs["success"] is True
        assert kwargs["latency_ms"] == 123
        mock_analytics.publish_nowait.assert_called_once_with("kb.query_executed", "proj1", {"fake": "payload"})

    def test_error_status_marks_success_false(self):
        from tilellm.modules.api_v2.controllers import _emit_api_v2_kb_query
        from tilellm.models import QuestionAnswer

        payload = QuestionAnswer(question="ciao", namespace="ns", engine=_engine(), id_project="proj1")
        response = JSONResponse(status_code=400, content={"detail": "Domanda non pertinente o non sicura."})

        with patch("tilellm.modules.api_v2.controllers.analytics") as mock_analytics:
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            _emit_api_v2_kb_query(payload, response, 5)

        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["success"] is False
        assert kwargs["chunks_retrieved"] == 0


class TestAskQuestionFullEmitsAnalytics:
    @pytest.mark.asyncio
    async def test_kb_query_emitted_after_graph_run(self):
        from tilellm.modules.api_v2.controllers import ask_question_full
        from tilellm.models import QuestionAnswer

        payload = QuestionAnswer(question="ciao", namespace="ns", engine=_engine(), id_project="proj1")
        fake_state = {"retrieval_result": AsyncMock(model_dump=lambda: {"success": True, "content_chunks": ["a"]})}

        with patch("tilellm.modules.api_v2.controllers.app") as mock_app, \
             patch("tilellm.modules.api_v2.controllers.analytics") as mock_analytics:
            mock_app.ainvoke = AsyncMock(return_value=fake_state)
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            await ask_question_full(payload)

        mock_analytics.events.kb_query.assert_called_once()
        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 1


class TestAskQuestionSimpleEmitsAnalytics:
    @pytest.mark.asyncio
    async def test_kb_query_emitted_after_graph_run(self):
        from tilellm.modules.api_v2.controllers import ask_question_simple
        from tilellm.modules.api_v2.models import QASimpleRequest

        payload = QASimpleRequest(question="ciao", namespace="ns", engine=_engine(), id_project="proj1")
        fake_state = {"retrieval_result": AsyncMock(model_dump=lambda: {"success": True, "content_chunks": []})}

        with patch("tilellm.modules.api_v2.controllers.simple_app") as mock_app, \
             patch("tilellm.modules.api_v2.controllers.analytics") as mock_analytics:
            mock_app.ainvoke = AsyncMock(return_value=fake_state)
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            await ask_question_simple(payload)

        mock_analytics.events.kb_query.assert_called_once()
        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 0
