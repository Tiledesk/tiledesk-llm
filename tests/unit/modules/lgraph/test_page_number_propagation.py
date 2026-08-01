#!/usr/bin/env python3
"""
page_number must survive the full lgraph pipeline: build_lgraph (reads it from
RepositoryQueryResult.metadata, see Task #4) -> build_light_graph (stores it on
LChunk nodes) -> _load_subgraph/ppr_search (reads it back into ranked results) ->
ChunkResult (audit trail exposed via LGraphQAResponse.chunks_used, debug=True).

Without this, lgraph's answers can never say "which page" — even after fixing
flat RAG's citation path (Tasks #1/#2), the graph pipeline discarded page_number
at the RepositoryQueryResult->chunk-dict narrowing step.
"""
from unittest.mock import AsyncMock, patch

import pytest

from tilellm.modules.lgraph.models.schemas import ChunkResult
from tilellm.modules.lgraph.services.graph_builder import build_light_graph
from tilellm.modules.lgraph.services.ppr_retriever import _load_subgraph, ppr_search


class TestChunkResultPageField:
    def test_page_defaults_none(self):
        assert ChunkResult(chunk_id="c1", text="t", metadata_id="d1", source="s", ppr_score=0.5).page is None

    def test_page_explicit(self):
        c = ChunkResult(chunk_id="c1", text="t", metadata_id="d1", source="s", ppr_score=0.5, page=7)
        assert c.page == 7


def _repo_with_query_sequence(*rows_per_call):
    """Fake repo._execute_query returning a fixed sequence of row-lists per call."""
    repo = AsyncMock()
    repo._execute_query = AsyncMock(side_effect=list(rows_per_call))
    return repo


class TestLoadSubgraphPageNumber:
    @pytest.mark.asyncio
    async def test_seed_chunk_carries_page_number(self):
        # Query order in _load_subgraph: entity_q, chunk_q, nb_q
        repo = _repo_with_query_sequence(
            [],  # entity_q: no seed entities
            [{"id": 1, "chunk_id": "c1", "text": "hello", "metadata_id": "d1",
              "source": "s1", "page_number": 4}],  # chunk_q
            [],  # nb_q: no neighbours
        )

        G, chunk_data, seeds = await _load_subgraph(
            repo, "gname", "ns", "idx", seed_entity_names=[], seed_chunk_ids=["c1"],
        )

        assert chunk_data[1]["page_number"] == 4
        assert G.nodes["chunk::c1"]["page_number"] == 4

    @pytest.mark.asyncio
    async def test_neighbour_chunk_carries_page_number(self):
        repo = _repo_with_query_sequence(
            [{"id": 10, "name": "acme"}],  # entity_q: one seed entity
            [],  # chunk_q: no seed chunks
            [{  # nb_q: neighbour chunk of the seed entity
                "s_id": 10, "s_labels": ["LEntity"], "n_id": 2, "n_labels": ["LChunk"],
                "rel_type": "HAS_ENTITY", "weight": 1.0, "s_key": "acme", "n_key": "c2",
                "n_text": "world", "n_metadata_id": "d2", "n_source": "s2", "n_page_number": 9,
            }],
        )

        G, chunk_data, seeds = await _load_subgraph(
            repo, "gname", "ns", "idx", seed_entity_names=["acme"], seed_chunk_ids=[],
        )

        assert G.nodes["chunk::c2"]["page_number"] == 9


class TestPprSearchPageNumber:
    @pytest.mark.asyncio
    async def test_result_includes_page_number(self):
        repo = _repo_with_query_sequence(
            [],
            [{"id": 1, "chunk_id": "c1", "text": "hello", "metadata_id": "d1",
              "source": "s1", "page_number": 4}],
            [],
        )

        results = await ppr_search(
            repo, "ns", "idx", seed_chunk_ids=["c1"], seed_entity_names=[],
            top_k=5, alpha=0.85, max_iter=50, graph_name="gname",
        )

        assert results[0]["page_number"] == 4


class TestBuildLightGraphPageNumber:
    @pytest.mark.asyncio
    async def test_chunk_node_stores_page_number(self):
        """The LChunk MERGE query must persist page_number so _load_subgraph
        can read it back (c.page_number in the RETURN clause)."""
        repo = AsyncMock()
        repo._execute_query = AsyncMock(return_value=[{"id": 1, "chunk_id": "c1"}])
        repo.delete_graph = AsyncMock()

        chunks = [{"id": "c1", "text": "hello", "metadata_id": "d1",
                   "source": "s1", "page_number": 12}]

        await build_light_graph(
            repo, chunks=chunks, chunk_entities={}, entity_doc_freq={},
            namespace="ns", index_name="idx", npmi_threshold=0.1, npmi_min_count=2,
        )

        # First _execute_query call is the LChunk MERGE — inspect its bound params
        call_args = repo._execute_query.call_args_list[0]
        query_text = call_args.args[0] if call_args.args else call_args.kwargs["q"]
        params = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["params"]

        assert "page_number" in query_text
        assert params["nodes"][0]["page_number"] == 12


class TestBuildLgraphReadsPageNumberFromMetadata:
    def test_matches_to_chunks_extracts_page_number(self):
        """_matches_to_chunks must pull page_number from RepositoryQueryResult.metadata
        (Task #4's generic passthrough) — without this, page_number never enters the
        graph pipeline at all, no matter what graph_builder/ppr_retriever do.
        Pure function, no DI/repo connection needed to test it."""
        from tilellm.models.schemas import RepositoryQueryResult
        from tilellm.modules.lgraph.logic import _matches_to_chunks

        matches = [
            RepositoryQueryResult(
                id="c1", metadata_id="d1", metadata_source="s1", text="hello",
                metadata={"id": "d1", "source": "s1", "page_number": 3},
            ),
        ]

        chunks = _matches_to_chunks(matches)

        assert chunks[0]["page_number"] == 3

    def test_matches_to_chunks_handles_missing_metadata(self):
        from tilellm.models.schemas import RepositoryQueryResult
        from tilellm.modules.lgraph.logic import _matches_to_chunks

        matches = [RepositoryQueryResult(id="c1", metadata_id="d1", metadata_source="s1")]
        chunks = _matches_to_chunks(matches)
        assert chunks[0]["page_number"] is None


class TestSearchLgraphChunkResultPage:
    @pytest.mark.asyncio
    async def test_page_propagated_into_chunk_result(self):
        from tilellm.modules.lgraph.logic import search_lgraph
        from tilellm.modules.lgraph.models.schemas import LGraphSearchRequest
        from tilellm.models.vector_store import Engine

        request = LGraphSearchRequest(
            question="chi ha firmato la delibera?",
            namespace="ns",
            engine=Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx"),
        )

        with patch("tilellm.modules.lgraph.logic._get_falkor_repo") as mock_falkor, \
             patch("tilellm.modules.lgraph.logic.extract_entities", return_value=[("delibera", "MISC")]), \
             patch("tilellm.modules.lgraph.logic.ppr_search", new=AsyncMock(return_value=[
                 {"chunk_id": "c1", "text": "t", "metadata_id": "d1", "source": "s1",
                  "page_number": 5, "ppr_score": 0.9},
             ])):
            mock_falkor.return_value = AsyncMock()
            result = await search_lgraph(request)

        assert result.chunks[0].page == 5
