#!/usr/bin/env python3
"""
lgraph analytics instrumentation (docs/MIGLIORIE_DA_FARE.md P1#14): every
write path emits kb.content_indexed, every single-query read path emits
kb.query_executed + ai.token_usage/ai.model_call for its LLM synthesis call,
and the community-summarizer batch loop emits ai.token_usage per call (same
token_usage-only convention as the compliance judge — no model_call per item).
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


class TestBuildLgraphAnalytics:
    @pytest.mark.asyncio
    async def test_content_indexed_emitted_on_success(self):
        from tilellm.modules.lgraph.logic import _build_lgraph_core as build_lgraph
        from tilellm.modules.lgraph.models.schemas import LGraphBuildRequest

        repo = AsyncMock()
        items = AsyncMock(matches=[
            AsyncMock(id="c1", text="t", metadata_id="d1", metadata_source="s1", metadata={}),
        ])
        repo.get_all_obj_namespace = AsyncMock(return_value=items)
        request = LGraphBuildRequest(namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.build_chunk_entity_matrix", return_value=({}, {})), \
             patch("tilellm.modules.lgraph.logic.build_light_graph", new=AsyncMock(return_value={
                 "chunks_processed": 1, "entities_created": 2,
                 "entity_chunk_edges": 2, "entity_entity_edges": 0,
             })), \
             patch("tilellm.modules.lgraph.logic.analytics") as mock_analytics:
            mock_falkor.return_value = AsyncMock()
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {"fake": "payload"})
            await build_lgraph(request, repo=repo)

        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["success"] is True
        assert kwargs["chunks_indexed"] == 1
        mock_analytics.publish_nowait.assert_called_once_with("kb.content_indexed", "proj1", {"fake": "payload"})

    @pytest.mark.asyncio
    async def test_content_indexed_emitted_on_empty_namespace(self):
        from tilellm.modules.lgraph.logic import _build_lgraph_core as build_lgraph
        from tilellm.modules.lgraph.models.schemas import LGraphBuildRequest

        repo = AsyncMock()
        repo.get_all_obj_namespace = AsyncMock(return_value=AsyncMock(matches=[]))
        request = LGraphBuildRequest(namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.lgraph.logic.analytics") as mock_analytics:
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {})
            result = await build_lgraph(request, repo=repo)

        assert result["status"] == "empty"
        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["success"] is True
        assert kwargs["chunks_indexed"] == 0

    @pytest.mark.asyncio
    async def test_content_indexed_emitted_on_failure(self):
        from tilellm.modules.lgraph.logic import _build_lgraph_core as build_lgraph
        from tilellm.modules.lgraph.models.schemas import LGraphBuildRequest

        repo = AsyncMock()
        repo.get_all_obj_namespace = AsyncMock(side_effect=RuntimeError("db down"))
        request = LGraphBuildRequest(namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.lgraph.logic.analytics") as mock_analytics:
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {})
            with pytest.raises(RuntimeError):
                await build_lgraph(request, repo=repo)

        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["success"] is False
        assert kwargs["error_message"] == "db down"


class TestSummarizeCommunitiesAnalytics:
    @pytest.mark.asyncio
    async def test_content_indexed_and_token_usage_emitted(self):
        """content_indexed goes through logic.py's own `analytics` import;
        token_usage goes through shared/token_tracking.py's emit_analytics,
        which does its own late `import tilellm.analytics` — so the two need
        patching at different targets (both resolve to the same real module,
        but a Mock bound to one name doesn't affect a fresh import elsewhere)."""
        from tilellm.modules.lgraph.logic import _summarize_communities_lgraph_core as summarize_communities_lgraph
        from tilellm.modules.lgraph.models.schemas import LGraphCommunitySummarizationRequest

        request = LGraphCommunitySummarizationRequest(namespace="ns", engine=_engine(), id_project="proj1")

        async def fake_generate(**kwargs):
            collector = kwargs["token_usage_collector"]
            fake_response = AsyncMock(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
            collector.record(fake_response, operation="lgraph_community_summary", model="gpt-4o-mini")
            return {"status": "success", "communities_processed": 1, "communities_indexed": 1, "errors": 0}

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic._get_vector_repo_for_engine", new=AsyncMock()), \
             patch(
                 "tilellm.modules.lgraph.services.community_summarizer.generate_community_summaries",
                 new=AsyncMock(side_effect=fake_generate),
             ), \
             patch("tilellm.modules.lgraph.logic.analytics") as mock_logic_analytics, \
             patch("tilellm.analytics.events.token_usage") as mock_token_usage, \
             patch("tilellm.analytics.publish_nowait") as mock_publish:
            mock_falkor.return_value = AsyncMock()
            mock_logic_analytics.events.content_indexed.return_value = ("kb.content_indexed", {})
            mock_logic_analytics.events.get_embedding_model_name.side_effect = lambda x: str(x)
            mock_logic_analytics.events.get_engine_value.side_effect = lambda x: getattr(x, "name", str(x))
            mock_token_usage.return_value = ("ai.token_usage", {})

            await summarize_communities_lgraph(request, llm=AsyncMock(), llm_embeddings=AsyncMock())

        _, kwargs = mock_logic_analytics.events.content_indexed.call_args
        assert kwargs["chunks_indexed"] == 1

        mock_token_usage.assert_called_once()
        _, tu_kwargs = mock_token_usage.call_args
        assert tu_kwargs["total_tokens"] == 15
        assert tu_kwargs["operation"] == "lgraph_community_summary"
        mock_publish.assert_any_call("ai.token_usage", "proj1", {})


class TestSearchLgraphAnalytics:
    @pytest.mark.asyncio
    async def test_kb_query_emitted_no_llm_involved(self):
        from tilellm.modules.lgraph.logic import search_lgraph
        from tilellm.modules.lgraph.models.schemas import LGraphSearchRequest

        request = LGraphSearchRequest(question="chi ha firmato?", namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[("acme", "ORG")]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=[
                 {"chunk_id": "c1", "text": "t", "metadata_id": "d1", "source": "s1", "ppr_score": 0.5},
             ])), \
             patch("tilellm.modules.lgraph.logic.analytics") as mock_analytics:
            mock_falkor.return_value = AsyncMock()
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            await search_lgraph(request)

        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 1
        assert kwargs["reranking_applied"] is False
        assert kwargs["success"] is None  # no LLM synthesis on /search — no opinion on success
        mock_analytics.publish_nowait.assert_called_once_with("kb.query_executed", "proj1", {})


class TestQaLgraphAnalytics:
    @pytest.mark.asyncio
    async def test_kb_query_and_llm_analytics_emitted_on_success(self):
        from tilellm.modules.lgraph.logic import _qa_lgraph_core
        from tilellm.modules.lgraph.models.schemas import LGraphQARequest

        request = LGraphQARequest(question="q", namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[("acme", "ORG")]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=[
                 {"chunk_id": "c1", "text": "t", "metadata_id": "d1", "source": "s1", "ppr_score": 0.5},
             ])), \
             patch("tilellm.modules.lgraph.logic.analytics") as mock_analytics, \
             patch("tilellm.analytics.events.token_usage") as mock_token_usage, \
             patch("tilellm.analytics.publish_nowait") as mock_tt_publish:
            mock_falkor.return_value = AsyncMock()
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            mock_analytics.events.model_call.return_value = ("ai.model_call", {})
            mock_token_usage.return_value = ("ai.token_usage", {})
            await _qa_lgraph_core(request, llm=_mock_llm())

        _, kq_kwargs = mock_analytics.events.kb_query.call_args
        assert kq_kwargs["chunks_retrieved"] == 1
        assert kq_kwargs["success"] is True

        _, mc_kwargs = mock_analytics.events.model_call.call_args
        assert mc_kwargs["success"] is True
        assert mc_kwargs["operation"] == "lgraph_qa"

        mock_token_usage.assert_called_once()
        _, tu_kwargs = mock_token_usage.call_args
        assert tu_kwargs["total_tokens"] == 15
        mock_tt_publish.assert_any_call("ai.token_usage", "proj1", {})

    @pytest.mark.asyncio
    async def test_kb_query_success_none_when_no_chunks(self):
        from tilellm.modules.lgraph.logic import _qa_lgraph_core
        from tilellm.modules.lgraph.models.schemas import LGraphQARequest

        request = LGraphQARequest(question="q", namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=[])), \
             patch("tilellm.modules.lgraph.logic.analytics") as mock_analytics:
            mock_falkor.return_value = AsyncMock()
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            await _qa_lgraph_core(request, llm=AsyncMock())

        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 0
        assert kwargs["success"] is None

    @pytest.mark.asyncio
    async def test_model_call_success_false_on_llm_error(self):
        from tilellm.modules.lgraph.logic import _qa_lgraph_core
        from tilellm.modules.lgraph.models.schemas import LGraphQARequest

        request = LGraphQARequest(question="q", namespace="ns", engine=_engine(), id_project="proj1")
        broken_llm = AsyncMock()
        broken_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[("acme", "ORG")]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=[
                 {"chunk_id": "c1", "text": "t", "metadata_id": "d1", "source": "s1", "ppr_score": 0.5},
             ])), \
             patch("tilellm.modules.lgraph.logic.analytics") as mock_analytics:
            mock_falkor.return_value = AsyncMock()
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            mock_analytics.events.model_call.return_value = ("ai.model_call", {})
            result = await _qa_lgraph_core(request, llm=broken_llm)

        _, mc_kwargs = mock_analytics.events.model_call.call_args
        assert mc_kwargs["success"] is False
        assert mc_kwargs["error_type"] == "RuntimeError"
        _, kq_kwargs = mock_analytics.events.kb_query.call_args
        assert kq_kwargs["success"] is False
        assert "[Errore LLM:" in result.answer


class TestQaLgraphHybridAnalytics:
    @pytest.mark.asyncio
    async def test_kb_query_and_llm_analytics_emitted(self):
        from tilellm.modules.lgraph.logic import _qa_lgraph_hybrid_core
        from tilellm.modules.lgraph.models.schemas import LGraphHybridRequest
        from tilellm.models.schemas import RetrievalChunksResult

        repo = AsyncMock()
        repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
            success=True, namespace="ns", chunks=["t1"], metadata=[{}], chunk_ids=["v1"],
        ))
        request = LGraphHybridRequest(question="q", namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=[
                 {"chunk_id": "v1", "text": "t1", "metadata_id": "d1", "source": "s1", "ppr_score": 0.5},
             ])), \
             patch("tilellm.modules.lgraph.logic.analytics") as mock_analytics, \
             patch("tilellm.analytics.events.token_usage") as mock_token_usage, \
             patch("tilellm.analytics.publish_nowait"):
            mock_falkor.return_value = AsyncMock()
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            mock_analytics.events.model_call.return_value = ("ai.model_call", {})
            mock_token_usage.return_value = ("ai.token_usage", {})
            await _qa_lgraph_hybrid_core(request, repo=repo, llm=_mock_llm())

        _, kq_kwargs = mock_analytics.events.kb_query.call_args
        assert kq_kwargs["chunks_retrieved"] == 1
        assert kq_kwargs["success"] is True
        _, mc_kwargs = mock_analytics.events.model_call.call_args
        assert mc_kwargs["operation"] == "lgraph_hybrid_qa"
