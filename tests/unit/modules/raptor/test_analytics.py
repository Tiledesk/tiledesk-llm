#!/usr/bin/env python3
"""
raptor analytics instrumentation (docs/MIGLIORIE_DA_FARE.md P1#14):
_build_raptor_tree_core emits kb.content_indexed, _retrieve_raptor_core emits
kb.query_executed (no LLM, success=None), _summarize_core's batch summary
loop emits ai.token_usage per call (no model_call — same convention as
lgraph's community summarizer / compliance judge), _raptor_qa_core emits
kb.query_executed + ai.token_usage/ai.model_call for its single foreground
synthesis call.
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tilellm.models.vector_store import Engine


def _engine():
    return Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx")


def _mock_llm(content: str = "answer") -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=Mock(content=content, usage_metadata={
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    }))
    return llm


class TestBuildRaptorTreeAnalytics:
    @pytest.mark.asyncio
    async def test_content_indexed_emitted_on_success(self):
        from tilellm.modules.raptor.controllers import _build_raptor_tree_core
        from tilellm.modules.raptor.models.models import RaptorRequest, RaptorResponse

        request = RaptorRequest(namespace="ns", doc_id="doc1", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.raptor.controllers.should_use_raptor_for_document", return_value=True), \
             patch("tilellm.modules.raptor.controllers._retrieve_document_chunks", new=AsyncMock(return_value=["c1", "c2"])), \
             patch("tilellm.modules.raptor.controllers.get_raptor_repo", new=AsyncMock()), \
             patch("tilellm.modules.raptor.controllers.RaptorService") as mock_service_cls, \
             patch("tilellm.modules.raptor.controllers.analytics") as mock_analytics:
            mock_service_cls.return_value.build_raptor_tree = AsyncMock(
                return_value=RaptorResponse(success=True, total_chunks=2, total_summaries=1),
            )
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {})
            mock_analytics.events.get_embedding_model_name.side_effect = lambda x: str(x)
            mock_analytics.events.get_engine_value.side_effect = lambda x: getattr(x, "name", str(x))
            result = await _build_raptor_tree_core(request, repo=AsyncMock(), llm=AsyncMock(), llm_embeddings=AsyncMock())

        assert result.success is True
        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["chunks_indexed"] == 2
        assert kwargs["success"] is True
        assert kwargs["source_type"] == "raptor_build"
        mock_analytics.publish_nowait.assert_called_once()

    @pytest.mark.asyncio
    async def test_content_indexed_emitted_when_activation_criteria_not_met(self):
        from tilellm.modules.raptor.controllers import _build_raptor_tree_core
        from tilellm.modules.raptor.models.models import RaptorRequest

        request = RaptorRequest(namespace="ns", doc_id="doc1", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.raptor.controllers.should_use_raptor_for_document", return_value=False), \
             patch("tilellm.modules.raptor.controllers.analytics") as mock_analytics:
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {})
            mock_analytics.events.get_embedding_model_name.side_effect = lambda x: str(x)
            mock_analytics.events.get_engine_value.side_effect = lambda x: getattr(x, "name", str(x))
            result = await _build_raptor_tree_core(request, repo=AsyncMock(), llm=AsyncMock(), llm_embeddings=AsyncMock())

        assert result.success is False
        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["success"] is False
        assert kwargs["chunks_indexed"] == 0


class TestRetrieveRaptorAnalytics:
    @pytest.mark.asyncio
    async def test_kb_query_emitted_no_llm_involved(self):
        from tilellm.modules.raptor.controllers import _retrieve_raptor_core
        from tilellm.modules.raptor.models.models import RaptorRetrievalRequest, RaptorRetrievalResult

        request = RaptorRetrievalRequest(question="q", namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.raptor.controllers.get_raptor_repo", new=AsyncMock()), \
             patch("tilellm.modules.raptor.controllers.RaptorRetriever") as mock_retriever_cls, \
             patch("tilellm.modules.raptor.controllers.analytics") as mock_analytics:
            mock_retriever_cls.return_value.retrieve = AsyncMock(return_value=RaptorRetrievalResult(
                success=True, results=[{"content": "x"}], strategy_used="collapsed_tree",
            ))
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            await _retrieve_raptor_core(request, repo=AsyncMock(), llm=AsyncMock(), llm_embeddings=AsyncMock())

        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 1
        assert kwargs["success"] is None


class TestSummarizeAnalytics:
    @pytest.mark.asyncio
    async def test_token_usage_emitted_per_group_no_model_call(self):
        from tilellm.modules.raptor.controllers import _summarize_core
        from tilellm.modules.raptor.models.models import RaptorSummaryRequest
        from langchain_core.documents import Document

        request = RaptorSummaryRequest(namespace="ns", chunk_ids=["c1", "c2"], engine=_engine(), id_project="proj1")
        chunks = [
            Document(page_content="chunk one", metadata={"id": "c1"}),
            Document(page_content="chunk two", metadata={"id": "c2"}),
        ]

        with patch("tilellm.modules.raptor.controllers._retrieve_chunks_by_ids", new=AsyncMock(return_value=chunks)), \
             patch("tilellm.analytics.events.token_usage") as mock_token_usage, \
             patch("tilellm.analytics.publish_nowait") as mock_publish, \
             patch("tilellm.modules.raptor.controllers.analytics") as mock_logic_analytics:
            mock_token_usage.return_value = ("ai.token_usage", {})
            result = await _summarize_core(request, repo=AsyncMock(), llm=_mock_llm(), llm_embeddings=AsyncMock())

        assert result.success is True
        assert result.total_groups == 1  # cluster_size=5 default, 2 chunks -> 1 group
        mock_token_usage.assert_called_once()
        _, kwargs = mock_token_usage.call_args
        assert kwargs["operation"] == "raptor_summarize"
        assert kwargs["total_tokens"] == 15
        mock_publish.assert_any_call("ai.token_usage", "proj1", {})
        mock_logic_analytics.events.model_call.assert_not_called()


class TestRaptorQaAnalytics:
    @pytest.mark.asyncio
    async def test_kb_query_and_llm_analytics_emitted_on_success(self):
        from tilellm.modules.raptor.controllers import _raptor_qa_core
        from tilellm.modules.raptor.models.models import RaptorQARequest, RaptorRetrievalResult

        request = RaptorQARequest(question="q", namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.raptor.controllers.get_raptor_repo", new=AsyncMock()), \
             patch("tilellm.modules.raptor.controllers.RaptorRetriever") as mock_retriever_cls, \
             patch("tilellm.modules.raptor.controllers.analytics") as mock_analytics, \
             patch("tilellm.analytics.events.token_usage") as mock_token_usage, \
             patch("tilellm.analytics.publish_nowait"):
            mock_retriever_cls.return_value.retrieve = AsyncMock(return_value=RaptorRetrievalResult(
                success=True, results=[{"content": "x", "level": 0}], strategy_used="collapsed_tree",
            ))
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            mock_analytics.events.model_call.return_value = ("ai.model_call", {})
            mock_token_usage.return_value = ("ai.token_usage", {})
            result = await _raptor_qa_core(request, repo=AsyncMock(), llm=_mock_llm(), llm_embeddings=AsyncMock())

        assert result.success is True
        _, kq_kwargs = mock_analytics.events.kb_query.call_args
        assert kq_kwargs["chunks_retrieved"] == 1
        assert kq_kwargs["success"] is True
        _, mc_kwargs = mock_analytics.events.model_call.call_args
        assert mc_kwargs["success"] is True
        assert mc_kwargs["operation"] == "raptor_qa"
        mock_token_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_kb_query_success_none_when_no_results(self):
        from tilellm.modules.raptor.controllers import _raptor_qa_core
        from tilellm.modules.raptor.models.models import RaptorQARequest, RaptorRetrievalResult

        request = RaptorQARequest(question="q", namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.raptor.controllers.get_raptor_repo", new=AsyncMock()), \
             patch("tilellm.modules.raptor.controllers.RaptorRetriever") as mock_retriever_cls, \
             patch("tilellm.modules.raptor.controllers.analytics") as mock_analytics:
            mock_retriever_cls.return_value.retrieve = AsyncMock(return_value=RaptorRetrievalResult(
                success=False, results=[], strategy_used="collapsed_tree", error="nothing found",
            ))
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            result = await _raptor_qa_core(request, repo=AsyncMock(), llm=_mock_llm(), llm_embeddings=AsyncMock())

        assert result.success is False
        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 0
        assert kwargs["success"] is None
