#!/usr/bin/env python3
"""
temporal_digest analytics instrumentation (docs/MIGLIORIE_DA_FARE.md P1#14):
generate_digest emits kb.content_indexed (digests are indexed into the vector
store), query_digest/agent_query_digest emit kb.query_executed. Wired at the
logic.py layer around DigestService.generate/query/agent_query (mocked here)
rather than logic.py's own @inject_llm_chat_async/@inject_repo_async-decorated
functions being called directly — those decorators do real DI when invoked,
so the service call itself is the natural seam.

token_usage/model_call are NOT wired: DigestService makes several internal
LLM calls per request (act_type classifier batch, judge/synthesis, rollup) —
deferred, documented ceiling, see docs/MIGLIORIE_DA_FARE.md P1#14.
"""
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from tilellm.models.vector_store import Engine


def _engine():
    return Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx")


class TestGenerateDigestAnalytics:
    @pytest.mark.asyncio
    async def test_content_indexed_emitted_on_success(self):
        from tilellm.modules.temporal_digest.logic import _generate_digest_core as generate_digest
        from tilellm.modules.temporal_digest.models.schemas import DigestGenerationRequest, DigestGenerationResponse

        request = DigestGenerationRequest(
            namespace="ns", date_from=date(2026, 8, 1), engine=_engine(), id_project="proj1",
        )
        fake_response = DigestGenerationResponse(
            namespace="ns", digests=[], total_chunks_processed=12, total_windows=1,
        )

        with patch(
            "tilellm.modules.temporal_digest.logic._service.generate",
            new=AsyncMock(return_value=fake_response),
        ), patch("tilellm.modules.temporal_digest.logic.analytics") as mock_analytics:
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {"fake": "payload"})
            mock_analytics.events.get_embedding_model_name.side_effect = lambda x: str(x)
            mock_analytics.events.get_engine_value.side_effect = lambda x: getattr(x, "name", str(x))
            result = await generate_digest(request, repo=AsyncMock(), llm=AsyncMock(), llm_embeddings=AsyncMock())

        assert result.total_chunks_processed == 12
        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["chunks_indexed"] == 12
        assert kwargs["success"] is True
        assert kwargs["source_type"] == "temporal_digest_generate"
        mock_analytics.publish_nowait.assert_called_once_with("kb.content_indexed", "proj1", {"fake": "payload"})

    @pytest.mark.asyncio
    async def test_content_indexed_emitted_on_failure(self):
        from tilellm.modules.temporal_digest.logic import _generate_digest_core as generate_digest
        from tilellm.modules.temporal_digest.models.schemas import DigestGenerationRequest

        request = DigestGenerationRequest(
            namespace="ns", date_from=date(2026, 8, 1), engine=_engine(), id_project="proj1",
        )

        with patch(
            "tilellm.modules.temporal_digest.logic._service.generate",
            new=AsyncMock(side_effect=RuntimeError("vector store down")),
        ), patch("tilellm.modules.temporal_digest.logic.analytics") as mock_analytics:
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {})
            mock_analytics.events.get_embedding_model_name.side_effect = lambda x: str(x)
            mock_analytics.events.get_engine_value.side_effect = lambda x: getattr(x, "name", str(x))
            with pytest.raises(RuntimeError):
                await generate_digest(request, repo=AsyncMock(), llm=AsyncMock(), llm_embeddings=AsyncMock())

        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["success"] is False
        assert kwargs["error_message"] == "vector store down"
        assert kwargs["chunks_indexed"] == 0


class TestQueryDigestAnalytics:
    @pytest.mark.asyncio
    async def test_kb_query_emitted_on_success(self):
        from tilellm.modules.temporal_digest.logic import _query_digest_core as query_digest
        from tilellm.modules.temporal_digest.models.schemas import DigestQueryRequest, DigestQueryResponse

        request = DigestQueryRequest(question="cosa e' successo?", namespace="ns", engine=_engine(), id_project="proj1")
        fake_response = DigestQueryResponse(answer="ok", query_mode="temporal", chunk_count=4)

        with patch(
            "tilellm.modules.temporal_digest.logic._service.query",
            new=AsyncMock(return_value=fake_response),
        ), patch("tilellm.modules.temporal_digest.logic.analytics") as mock_analytics:
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {"fake": "payload"})
            result = await query_digest(request, repo=AsyncMock(), llm=AsyncMock(), llm_embeddings=AsyncMock())

        assert result.chunk_count == 4
        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 4
        assert kwargs["success"] is True
        mock_analytics.publish_nowait.assert_called_once_with("kb.query_executed", "proj1", {"fake": "payload"})


class TestAgentQueryDigestAnalytics:
    @pytest.mark.asyncio
    async def test_kb_query_emitted_on_success(self):
        from tilellm.modules.temporal_digest.logic import _agent_query_digest_core as agent_query_digest
        from tilellm.modules.temporal_digest.models.schemas import DigestAgentRequest, DigestAgentResponse

        request = DigestAgentRequest(question="e ieri?", namespace="ns", engine=_engine(), id_project="proj1")
        fake_response = DigestAgentResponse(answer="ok", query_mode="temporal", chunk_count=2, extracted_query_mode="temporal")

        with patch(
            "tilellm.modules.temporal_digest.logic._service.agent_query",
            new=AsyncMock(return_value=fake_response),
        ), patch("tilellm.modules.temporal_digest.logic.analytics") as mock_analytics:
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            result = await agent_query_digest(request, repo=AsyncMock(), llm=AsyncMock(), llm_embeddings=AsyncMock())

        assert result.chunk_count == 2
        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 2
        assert kwargs["success"] is True
