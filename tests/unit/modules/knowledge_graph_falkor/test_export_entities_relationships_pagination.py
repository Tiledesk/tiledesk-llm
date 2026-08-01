#!/usr/bin/env python3
"""
_export_entities_to_dataframe / _export_relationships_to_dataframe ran their
own unpaginated Cypher query, silently truncated by FalkorDB's default
resultset_size=10000 cap (no error, no warning). Observed live on namespace
256237 (2026-07-29): task reported nodes_created=10908, relationships_created
=15297, but entities.parquet/relationships.parquet saved to MinIO only had
10000 rows each (908 entities and 5297 relationships — ~35% — silently lost).

get_all_nodes_and_relationships (repository layer) already solves this with
SKIP/LIMIT pagination (used correctly by clustering). Fix: the two export
methods must delegate to it instead of re-querying, so the parquet snapshot
always matches the live graph.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest


def _service():
    from tilellm.modules.knowledge_graph_falkor.services.community_graph_service import (
        CommunityGraphService,
    )
    return CommunityGraphService(graph_rag_service=MagicMock())


class TestExportEntitiesDelegatesToPaginatedFetch:
    @pytest.mark.asyncio
    async def test_uses_get_all_nodes_and_relationships_not_raw_query(self):
        service = _service()
        repo = MagicMock()
        repo._execute_query = AsyncMock(
            side_effect=AssertionError("must not call the unpaginated raw query anymore")
        )
        repo.get_all_nodes_and_relationships = AsyncMock(return_value={
            "nodes": [
                {"id": "1", "label": "ORGANIZATION", "properties": {"name": "Banca ABC", "description": "d1"}},
                {"id": "2", "label": "PERSON", "properties": {"name": "Mario Bianchi", "description": "d2"}},
            ],
            "relationships": [],
        })

        df = await service._export_entities_to_dataframe(repo, namespace="43282", index_name="idx", graph_name="43282-debt_recovery")

        repo.get_all_nodes_and_relationships.assert_called_once()
        assert len(df) == 2
        assert set(df["name"]) == {"Banca ABC", "Mario Bianchi"}
        assert "node_id" in df.columns
        assert "labels" in df.columns

    @pytest.mark.asyncio
    async def test_includes_more_than_10000_rows_when_present(self):
        service = _service()
        repo = MagicMock()
        repo._execute_query = AsyncMock(side_effect=AssertionError("must not call raw query"))
        nodes = [
            {"id": str(i), "label": "CONTRACT", "properties": {"name": f"n{i}"}}
            for i in range(10908)
        ]
        repo.get_all_nodes_and_relationships = AsyncMock(return_value={"nodes": nodes, "relationships": []})

        df = await service._export_entities_to_dataframe(repo, namespace="43282", index_name="idx", graph_name="43282-debt_recovery")

        assert len(df) == 10908


class TestExportRelationshipsDelegatesToPaginatedFetch:
    @pytest.mark.asyncio
    async def test_uses_get_all_nodes_and_relationships_not_raw_query(self):
        service = _service()
        repo = MagicMock()
        repo._execute_query = AsyncMock(
            side_effect=AssertionError("must not call the unpaginated raw query anymore")
        )
        repo.get_all_nodes_and_relationships = AsyncMock(return_value={
            "nodes": [],
            "relationships": [
                {"id": "10", "type": "HAS_LOAN", "properties": {"amount": 1000},
                 "source_id": "1", "target_id": "2"},
            ],
        })

        df = await service._export_relationships_to_dataframe(repo, namespace="43282", index_name="idx", graph_name="43282-debt_recovery")

        repo.get_all_nodes_and_relationships.assert_called_once()
        assert len(df) == 1
        assert df.iloc[0]["relationship_type"] == "HAS_LOAN"
        assert df.iloc[0]["source_id"] == "1"
        assert df.iloc[0]["target_id"] == "2"

    @pytest.mark.asyncio
    async def test_includes_more_than_10000_rows_when_present(self):
        service = _service()
        repo = MagicMock()
        repo._execute_query = AsyncMock(side_effect=AssertionError("must not call raw query"))
        rels = [
            {"id": str(i), "type": "RELATED_TO", "properties": {}, "source_id": "1", "target_id": "2"}
            for i in range(15297)
        ]
        repo.get_all_nodes_and_relationships = AsyncMock(return_value={"nodes": [], "relationships": rels})

        df = await service._export_relationships_to_dataframe(repo, namespace="43282", index_name="idx", graph_name="43282-debt_recovery")

        assert len(df) == 15297
