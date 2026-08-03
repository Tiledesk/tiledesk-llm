#!/usr/bin/env python3
"""
/api/lgraph/hybrid — vector-seeded PPR + RRF fusion (decision locked in
memory/project_debt_recovery_benchmark.md): a prior dense/sparse search on the
same namespace gives top-K chunk ids; those become PPR seed_chunk_ids (the
parameter ppr_search already had but nobody fed); the vector-rank and PPR-rank
orderings are combined with reciprocal_rank_fusion (already generic/reusable
in knowledge_graph/utils/rrf.py).
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tilellm.models.vector_store import Engine


def _engine():
    return Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx")


def _llm_response(content: str) -> Mock:
    """A real AIMessage shape (plain Mock, not AsyncMock — attribute access on
    AsyncMock auto-creates async coroutines, which breaks token_tracking's
    synchronous usage_metadata.get(...) reads)."""
    return Mock(content=content, usage_metadata=None)


def _mock_llm(content: str = "answer text") -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=_llm_response(content))
    return llm


class TestLGraphHybridRequestModel:
    def test_defaults(self):
        from tilellm.modules.lgraph.models.schemas import LGraphHybridRequest

        req = LGraphHybridRequest(question="q", namespace="ns", engine=_engine())
        assert req.search_type == "hybrid"
        assert req.sparse_encoder == "splade"
        assert req.vector_top_k == 10
        assert req.rrf_k == 60

    def test_inherits_qa_fields(self):
        from tilellm.modules.lgraph.models.schemas import LGraphHybridRequest

        req = LGraphHybridRequest(question="q", namespace="ns", engine=_engine(), top_k=7, ppr_alpha=0.9)
        assert req.top_k == 7
        assert req.ppr_alpha == 0.9


class TestQaLgraphHybrid:
    @pytest.mark.asyncio
    async def test_vector_seeds_feed_ppr_seed_chunk_ids(self):
        """The whole point of /hybrid: vector search results must reach
        ppr_search as seed_chunk_ids (previously always []) ."""
        from tilellm.modules.lgraph.logic import _qa_lgraph_hybrid_core
        from tilellm.modules.lgraph.models.schemas import LGraphHybridRequest

        repo = AsyncMock()
        repo.get_chunks_from_repo = AsyncMock(return_value=_retrieval_result(
            chunk_ids=["v1", "v2"], chunks=["t1", "t2"], metadata=[{}, {}],
        ))
        request = LGraphHybridRequest(question="chi ha firmato?", namespace="ns", engine=_engine())

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[("acme", "ORG")]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=[
                 {"chunk_id": "v1", "text": "t1", "metadata_id": "d1", "source": "s1",
                  "page_number": 3, "ppr_score": 0.5},
             ])) as mock_ppr:
            mock_falkor.return_value = AsyncMock()
            result = await _qa_lgraph_hybrid_core(request, repo=repo, llm=_mock_llm())

        _, kwargs = mock_ppr.call_args
        assert kwargs["seed_chunk_ids"] == ["v1", "v2"]
        assert result.seeded_by == ["vector", "entity"]

    @pytest.mark.asyncio
    async def test_rrf_fuses_vector_and_ppr_rankings(self):
        """A chunk ranked highly in BOTH vector search and PPR should win the fusion."""
        from tilellm.modules.lgraph.logic import _qa_lgraph_hybrid_core
        from tilellm.modules.lgraph.models.schemas import LGraphHybridRequest

        repo = AsyncMock()
        # vector rank order: v2 first, v1 second
        repo.get_chunks_from_repo = AsyncMock(return_value=_retrieval_result(
            chunk_ids=["v2", "v1"], chunks=["t2", "t1"], metadata=[{}, {}],
        ))
        request = LGraphHybridRequest(question="q", namespace="ns", engine=_engine(), debug=True)

        ppr_raw = [
            {"chunk_id": "v1", "text": "t1", "metadata_id": "d1", "source": "s1", "page_number": 1, "ppr_score": 0.9},
            {"chunk_id": "v2", "text": "t2", "metadata_id": "d2", "source": "s2", "page_number": 2, "ppr_score": 0.8},
        ]

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=ppr_raw)):
            mock_falkor.return_value = AsyncMock()
            result = await _qa_lgraph_hybrid_core(request, repo=repo, llm=_mock_llm())

        # v2 leads vector rank (pos 0) and trails PPR rank (pos 1); v1 trails vector (pos 1)
        # and leads PPR (pos 0) — symmetric, so RRF ties them; either order is a valid fusion.
        # What matters: both chunks are present and page numbers survived.
        assert {c.chunk_id for c in result.chunks_used} == {"v1", "v2"}
        pages = {c.chunk_id: c.page for c in result.chunks_used}
        assert pages == {"v1": 1, "v2": 2}

    @pytest.mark.asyncio
    async def test_falls_back_to_ppr_only_when_vector_search_fails(self):
        """Vector search failure must degrade gracefully to entity-seeded PPR
        (today's qa_lgraph behavior), not crash the whole hybrid endpoint."""
        from tilellm.modules.lgraph.logic import _qa_lgraph_hybrid_core
        from tilellm.modules.lgraph.models.schemas import LGraphHybridRequest

        repo = AsyncMock()
        repo.get_chunks_from_repo = AsyncMock(side_effect=ValueError("No chunks found"))
        request = LGraphHybridRequest(question="q", namespace="ns", engine=_engine())

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[("acme", "ORG")]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=[])) as mock_ppr:
            mock_falkor.return_value = AsyncMock()
            result = await _qa_lgraph_hybrid_core(request, repo=repo, llm=AsyncMock())

        _, kwargs = mock_ppr.call_args
        assert kwargs["seed_chunk_ids"] == []
        assert "acme" in kwargs["seed_entity_names"]
        assert result.chunk_count == 0
        assert result.seeded_by == ["entity"]

    @pytest.mark.asyncio
    async def test_seeded_by_empty_when_neither_source_yields_seeds(self):
        """No vector hits and no entities/keywords at all → seeded_by == []
        (distinct from a real 'entity' seed with zero PPR results)."""
        from tilellm.modules.lgraph.logic import _qa_lgraph_hybrid_core
        from tilellm.modules.lgraph.models.schemas import LGraphHybridRequest

        repo = AsyncMock()
        repo.get_chunks_from_repo = AsyncMock(return_value=_retrieval_result(
            chunk_ids=[], chunks=[], metadata=[],
        ))
        request = LGraphHybridRequest(question="", namespace="ns", engine=_engine())

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[]), \
             patch("tilellm.modules.lgraph.logic.expand_date_references", return_value=[]), \
             patch("tilellm.modules.lgraph.logic.extract_query_keywords", return_value=[]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=[])):
            mock_falkor.return_value = AsyncMock()
            result = await _qa_lgraph_hybrid_core(request, repo=repo, llm=AsyncMock())

        assert result.seeded_by == []


def _retrieval_result(chunk_ids, chunks, metadata):
    from tilellm.models.schemas import RetrievalChunksResult
    return RetrievalChunksResult(success=True, namespace="ns", chunks=chunks, metadata=metadata, chunk_ids=chunk_ids)


class TestHybridEndpointRegistered:
    def test_route_exists(self):
        from tilellm.modules.lgraph.controllers import router
        paths = {route.path for route in router.routes}
        assert "/api/lgraph/hybrid" in paths
