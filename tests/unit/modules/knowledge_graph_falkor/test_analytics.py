#!/usr/bin/env python3
"""
falkor analytics instrumentation (docs/MIGLIORIE_DA_FARE.md P1#14).

Unlike lgraph, falkor's QA-style logic.py functions (query_graph,
context_fusion_graph_search, multimodal_search, advanced_qa_search,
agentic_qa_search) are thin validate-then-delegate wrappers around a
services/ method that does the actual retrieval + LLM synthesis — the LLM
call itself isn't visible in logic.py, and the result dict shape is
heterogeneous across the 5 endpoints (no single canonical 'chunks_retrieved'
field). All 5 route through the same two shared wrappers
(_emit_falkor_kb_query_wrapped / _emit_falkor_content_indexed_wrapped,
also used by create_graph/add_document_to_graph) — testing those wrappers
directly covers every call site's actual logic without needing to
DI-bypass each of the 7 outer functions individually.

token_usage/model_call are NOT wired here: they would require instrumenting
each service's internal llm.ainvoke call sites (services/community_graph_
service.py, services/advanced_qa_service.py, etc.) — deferred, documented
ceiling, see docs/MIGLIORIE_DA_FARE.md P1#14.
"""
from unittest.mock import AsyncMock, patch

import pytest

from tilellm.models.vector_store import Engine


def _engine():
    return Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx")


class TestExtractFalkorChunkCount:
    def test_non_dict_returns_zero(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _extract_falkor_chunk_count
        assert _extract_falkor_chunk_count(None) == 0
        assert _extract_falkor_chunk_count("oops") == 0

    def test_reports_used_key(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _extract_falkor_chunk_count
        assert _extract_falkor_chunk_count({"reports_used": 4}) == 4

    def test_nested_scores_local_chunks(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _extract_falkor_chunk_count
        assert _extract_falkor_chunk_count({"scores": {"local_chunks": 7}}) == 7

    def test_sources_list_length(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _extract_falkor_chunk_count
        assert _extract_falkor_chunk_count({"sources": [1, 2, 3]}) == 3

    def test_no_known_key_returns_zero(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _extract_falkor_chunk_count
        assert _extract_falkor_chunk_count({"answer": "hi"}) == 0


class TestExtractFalkorIndexedCount:
    def test_chunks_processed_key(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _extract_falkor_indexed_count
        assert _extract_falkor_indexed_count({"chunks_processed": 42}) == 42

    def test_entities_created_fallback(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _extract_falkor_indexed_count
        assert _extract_falkor_indexed_count({"entities_created": 5}) == 5


class TestFalkorQuestionText:
    def test_string_question(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _falkor_question_text
        assert _falkor_question_text(AsyncMock(question="ciao")) == "ciao"

    def test_multimodal_question_list(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _falkor_question_text
        from tilellm.models.schemas.multimodal_content import TextContent
        req = AsyncMock(question=[TextContent(text="ciao multimodale")])
        assert _falkor_question_text(req) == "ciao multimodale"


class TestEmitFalkorKbQueryWrapped:
    @pytest.mark.asyncio
    async def test_success_emits_with_extracted_chunk_count(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _emit_falkor_kb_query_wrapped

        async def fake_call():
            return {"answer": "ok", "reports_used": 3}

        request = AsyncMock(namespace="ns", question="q", request_id="req1", id_project="proj1")

        with patch("tilellm.modules.knowledge_graph_falkor.logic.analytics") as mock_analytics:
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {"fake": "payload"})
            result = await _emit_falkor_kb_query_wrapped(fake_call(), request, reranking_applied=True)

        assert result == {"answer": "ok", "reports_used": 3}
        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["chunks_retrieved"] == 3
        assert kwargs["reranking_applied"] is True
        assert kwargs["success"] is True
        mock_analytics.publish_nowait.assert_called_once_with("kb.query_executed", "proj1", {"fake": "payload"})

    @pytest.mark.asyncio
    async def test_status_error_marks_success_false(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _emit_falkor_kb_query_wrapped

        async def fake_call():
            return {"answer": "bad", "status": "error"}

        request = AsyncMock(namespace="ns", question="q", request_id=None, id_project="proj1")

        with patch("tilellm.modules.knowledge_graph_falkor.logic.analytics") as mock_analytics:
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            await _emit_falkor_kb_query_wrapped(fake_call(), request)

        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_exception_reraised_and_analytics_still_emitted(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _emit_falkor_kb_query_wrapped

        async def fake_call():
            raise RuntimeError("service down")

        request = AsyncMock(namespace="ns", question="q", request_id=None, id_project="proj1")

        with patch("tilellm.modules.knowledge_graph_falkor.logic.analytics") as mock_analytics:
            mock_analytics.events.kb_query.return_value = ("kb.query_executed", {})
            with pytest.raises(RuntimeError):
                await _emit_falkor_kb_query_wrapped(fake_call(), request)

        _, kwargs = mock_analytics.events.kb_query.call_args
        assert kwargs["success"] is False
        assert kwargs["chunks_retrieved"] == 0


class TestEmitFalkorContentIndexedWrapped:
    @pytest.mark.asyncio
    async def test_success_emits_with_extracted_count(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _emit_falkor_content_indexed_wrapped

        async def fake_call():
            return {"status": "success", "chunks_processed": 10}

        request = AsyncMock(namespace="ns", engine=_engine(), embedding="text-embedding-3-small",
                             request_id=None, id_project="proj1")

        with patch("tilellm.modules.knowledge_graph_falkor.logic.analytics") as mock_analytics:
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {"fake": "payload"})
            mock_analytics.events.get_embedding_model_name.side_effect = lambda x: str(x)
            mock_analytics.events.get_engine_value.side_effect = lambda x: getattr(x, "name", str(x))
            result = await _emit_falkor_content_indexed_wrapped(fake_call(), request, source_type="falkor_create_graph")

        assert result["chunks_processed"] == 10
        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["chunks_indexed"] == 10
        assert kwargs["success"] is True
        assert kwargs["source_type"] == "falkor_create_graph"
        mock_analytics.publish_nowait.assert_called_once_with("kb.content_indexed", "proj1", {"fake": "payload"})

    @pytest.mark.asyncio
    async def test_status_failed_marks_success_false(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _emit_falkor_content_indexed_wrapped

        async def fake_call():
            return {"status": "failed"}

        request = AsyncMock(namespace="ns", engine=_engine(), embedding="text-embedding-3-small",
                             request_id=None, id_project="proj1")

        with patch("tilellm.modules.knowledge_graph_falkor.logic.analytics") as mock_analytics:
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {})
            mock_analytics.events.get_embedding_model_name.side_effect = lambda x: str(x)
            mock_analytics.events.get_engine_value.side_effect = lambda x: getattr(x, "name", str(x))
            await _emit_falkor_content_indexed_wrapped(fake_call(), request, source_type="falkor_create_graph")

        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["success"] is False


class TestAddDocumentToGraphCore:
    @pytest.mark.asyncio
    async def test_content_indexed_emitted_on_success(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _add_document_to_graph_core
        from tilellm.modules.knowledge_graph_falkor.models.schemas import AddDocumentRequest

        request = AddDocumentRequest(
            metadata_id="doc1", namespace="ns", engine=_engine(), id_project="proj1",
        )

        with patch("tilellm.modules.knowledge_graph_falkor.logic.ensure_initialized", new=AsyncMock()), \
             patch("tilellm.modules.knowledge_graph_falkor.logic.graph_rag_service") as mock_rag, \
             patch("tilellm.modules.knowledge_graph_falkor.logic.COMMUNITY_GRAPH_AVAILABLE", False), \
             patch("tilellm.modules.knowledge_graph_falkor.logic.analytics") as mock_analytics:
            mock_rag.add_document_to_graph = AsyncMock(return_value={
                "status": "success", "chunks_processed": 3, "entities_extracted": 2,
            })
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {})
            mock_analytics.events.get_embedding_model_name.side_effect = lambda x: str(x)
            mock_analytics.events.get_engine_value.side_effect = lambda x: getattr(x, "name", str(x))
            result = await _add_document_to_graph_core(request, repo=AsyncMock(), llm=AsyncMock(), llm_embeddings=AsyncMock())

        assert result["chunks_processed"] == 3
        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["chunks_indexed"] == 3
        assert kwargs["success"] is True
        assert kwargs["source_type"] == "falkor_add_document"

    @pytest.mark.asyncio
    async def test_content_indexed_emitted_on_validation_failure(self):
        from tilellm.modules.knowledge_graph_falkor.logic import _add_document_to_graph_core
        from tilellm.modules.knowledge_graph_falkor.models.schemas import AddDocumentRequest

        request = AddDocumentRequest(metadata_id="doc1", namespace="ns", engine=_engine(), id_project="proj1")

        with patch("tilellm.modules.knowledge_graph_falkor.logic.ensure_initialized", new=AsyncMock()), \
             patch("tilellm.modules.knowledge_graph_falkor.logic.analytics") as mock_analytics:
            mock_analytics.events.content_indexed.return_value = ("kb.content_indexed", {})
            mock_analytics.events.get_embedding_model_name.side_effect = lambda x: str(x)
            mock_analytics.events.get_engine_value.side_effect = lambda x: getattr(x, "name", str(x))
            with pytest.raises(ValueError):
                await _add_document_to_graph_core(request, repo=AsyncMock(), llm=None, llm_embeddings=AsyncMock())

        _, kwargs = mock_analytics.events.content_indexed.call_args
        assert kwargs["success"] is False
        assert "LLM configuration is required" in kwargs["error_message"]
