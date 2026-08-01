#!/usr/bin/env python3
"""
GraphQAAdvancedResponse (falkor's /api/kg-falkor/hybrid — the REAL production
endpoint used by the debt_recovery benchmark, distinct from the legacy Neo4j
/api/kg module which has its own parallel community_graph_service.py) had zero
document/page provenance — run_local_retrieval only pulled `id`/`text`/`score`
from each match, discarding `source`/`file_name`/`page_number` even though the
full metadata dict was already available (m.get("metadata", {})).

_chunk_from_match / _citations_from_local_chunks are extracted as pure functions
so the fix is unit-testable without mocking the full 400-line context_fusion_search
pipeline (embeddings, cached vector store wrapper, graph expansion, reranker, LLM).
"""
from unittest.mock import AsyncMock, patch

import pytest

from tilellm.modules.knowledge_graph_falkor.services.community_graph_service import (
    _chunk_from_match,
    _citations_from_local_chunks,
)


class TestChunkFromMatch:
    def test_extracts_full_provenance(self):
        m = {
            "id": "chunk1",
            "score": 0.83,
            "metadata": {"text": "hello world", "source": "s1", "file_name": "delibera.pdf", "page_number": 4},
        }
        c = _chunk_from_match(m)
        assert c == {
            "id": "chunk1", "text": "hello world", "score": 0.83,
            "source": "s1", "file_name": "delibera.pdf", "page_number": 4,
        }

    def test_falls_back_to_page_content_when_text_missing(self):
        m = {"id": "c1", "score": 0.5, "metadata": {}, "page_content": "fallback text"}
        c = _chunk_from_match(m)
        assert c["text"] == "fallback text"

    def test_missing_metadata_keys_default_to_none(self):
        m = {"id": "c1", "score": 0.1, "metadata": {"text": "x"}}
        c = _chunk_from_match(m)
        assert c["source"] is None
        assert c["file_name"] is None
        assert c["page_number"] is None


class TestCitationsFromLocalChunks:
    def test_builds_deduplicated_source_list(self):
        chunks = [
            {"id": "c1", "text": "a", "score": 0.9, "source": "s1", "file_name": "a.pdf", "page_number": 1},
            {"id": "c2", "text": "b", "score": 0.8, "source": "s1", "file_name": "a.pdf", "page_number": 2},
            {"id": "c3", "text": "c", "score": 0.7, "source": "s2", "file_name": "b.pdf", "page_number": 1},
        ]
        citations = _citations_from_local_chunks(chunks)
        assert citations == [
            {"source": "s1", "file_name": "a.pdf", "page_number": 1},
            {"source": "s1", "file_name": "a.pdf", "page_number": 2},
            {"source": "s2", "file_name": "b.pdf", "page_number": 1},
        ]

    def test_empty_chunks_returns_empty(self):
        assert _citations_from_local_chunks([]) == []

    def test_skips_chunks_without_source(self):
        chunks = [{"id": "c1", "text": "a", "score": 0.5, "source": None, "file_name": None, "page_number": None}]
        assert _citations_from_local_chunks(chunks) == []


class TestGraphQaHybridEndpointWiresSources:
    @pytest.mark.asyncio
    async def test_sources_reach_the_response(self):
        from tilellm.modules.knowledge_graph_falkor.controllers import graph_qa_hybrid
        from tilellm.modules.knowledge_graph_falkor.models.schemas import GraphQAAdvancedRequest
        from tilellm.models import Engine

        fake_result = {
            "answer": "ans", "entities": [], "relationships": [],
            "retrieval_strategy": "integrated_hybrid_technical", "scores": {},
            "expanded_nodes": [], "expanded_relationships": [],
            "sources": [{"source": "s1", "file_name": "delibera.pdf", "page_number": 4}],
            "chat_history_dict": {},
        }
        request = GraphQAAdvancedRequest(
            question="q", namespace="ns",
            engine=Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx"),
        )

        with patch(
            "tilellm.modules.knowledge_graph_falkor.controllers.kg_logic.context_fusion_graph_search",
            new=AsyncMock(return_value=fake_result),
        ):
            response = await graph_qa_hybrid(request)

        assert response.sources == [{"source": "s1", "file_name": "delibera.pdf", "page_number": 4}]

    @pytest.mark.asyncio
    async def test_sources_default_empty_when_absent(self):
        """Other endpoints sharing GraphQAAdvancedResponse (advanced_qa_search,
        agentic_qa_search) don't populate 'sources' — must default to [], not error."""
        from tilellm.modules.knowledge_graph_falkor.controllers import graph_qa_hybrid
        from tilellm.modules.knowledge_graph_falkor.models.schemas import GraphQAAdvancedRequest
        from tilellm.models import Engine

        fake_result = {"answer": "ans", "entities": [], "relationships": [], "scores": {},
                        "expanded_nodes": [], "expanded_relationships": [], "chat_history_dict": {}}
        request = GraphQAAdvancedRequest(
            question="q", namespace="ns",
            engine=Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx"),
        )

        with patch(
            "tilellm.modules.knowledge_graph_falkor.controllers.kg_logic.context_fusion_graph_search",
            new=AsyncMock(return_value=fake_result),
        ):
            response = await graph_qa_hybrid(request)

        assert response.sources == []
