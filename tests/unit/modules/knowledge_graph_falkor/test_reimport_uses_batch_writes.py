#!/usr/bin/env python3
"""
_reimport must bulk-load through the repository's existing batch writers instead
of issuing one query per node and one per relationship.

On namespace 256237 (2026-07-30) a restore meant 10 000 create_node calls plus
7 311 create_relationship calls — 17 311 round trips — after which FalkorDB died
("Connection closed by server") and came back on an older RDB snapshot, losing
the entire restore. batch_create_nodes / batch_create_relationships already exist
and are what import_from_vector_store uses (UNWIND, batch_size=100).

Relationships are re-linked by entity NAME (the snapshot carries source_entity /
target_entity) via the normalized_name→node_id map batch_create_nodes returns —
the stored source_id/target_id are ids of the wiped graph and cannot be trusted.
"""
import io
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


def _parquet(rows) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    return buf.getvalue()


def _repo():
    repo = MagicMock()
    repo.delete_nodes_by_metadata = AsyncMock(return_value={"nodes_deleted": 0})
    repo.batch_create_nodes = AsyncMock(return_value={"banca abc": "500", "mario bianchi": "501"})
    repo.batch_create_relationships = AsyncMock(return_value=1)
    repo.save_community_report = AsyncMock(return_value="r1")
    repo._normalize_name = lambda s: s.strip().lower()
    repo.create_node = AsyncMock(side_effect=AssertionError("must not create nodes one by one"))
    repo.create_relationship = AsyncMock(side_effect=AssertionError("must not create relationships one by one"))
    return repo


NODES = _parquet([
    {"id": "7", "label": "ORGANIZATION", "name": "Banca ABC", "description": "creditore", "source_ids": "[]"},
    {"id": "8", "label": "PERSON", "name": "Mario Bianchi", "description": "debitore", "source_ids": "[]"},
])
RELS = _parquet([
    {"id": "9", "type": "HAS_LOAN", "source_id": "7", "target_id": "8",
     "source_entity": "Banca ABC", "target_entity": "Mario Bianchi",
     "description": "prestito", "source_ids": "[]"},
])


class TestReimportUsesBatchWriters:
    @pytest.mark.asyncio
    async def test_nodes_go_through_batch_create_nodes(self):
        from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import GraphOptimizer

        repo = _repo()
        optimizer = GraphOptimizer(repository=repo, minio_storage_service=MagicMock())

        await optimizer._reimport(
            namespace="256237", graph_name="256237-debt_recovery",
            nodes_bytes=NODES, rels_bytes=RELS, community_reports=[],
        )

        repo.batch_create_nodes.assert_awaited_once()
        _, kwargs = repo.batch_create_nodes.call_args
        entities = kwargs["entities"]
        assert len(entities) == 2
        assert {e["entity_name"] for e in entities} == {"Banca ABC", "Mario Bianchi"}
        assert {e["entity_type"] for e in entities} == {"ORGANIZATION", "PERSON"}
        assert kwargs["namespace"] == "256237"
        assert kwargs["graph_name"] == "256237-debt_recovery"

    @pytest.mark.asyncio
    async def test_relationships_go_through_batch_and_are_keyed_by_name(self):
        from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import GraphOptimizer

        repo = _repo()
        optimizer = GraphOptimizer(repository=repo, minio_storage_service=MagicMock())

        await optimizer._reimport(
            namespace="256237", graph_name="256237-debt_recovery",
            nodes_bytes=NODES, rels_bytes=RELS, community_reports=[],
        )

        repo.batch_create_relationships.assert_awaited_once()
        _, kwargs = repo.batch_create_relationships.call_args
        rels = kwargs["relationships"]
        assert len(rels) == 1
        assert rels[0]["relationship_type"] == "HAS_LOAN"
        assert rels[0]["src_id"] == "Banca ABC"
        assert rels[0]["tgt_id"] == "Mario Bianchi"
        # resolution map comes from batch_create_nodes' return value
        assert kwargs["entity_node_map"] == {"banca abc": "500", "mario bianchi": "501"}

    @pytest.mark.asyncio
    async def test_community_entities_remapped_via_name_map(self):
        """Old snapshot ids must still resolve to the newly created node ids."""
        from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import GraphOptimizer

        repo = _repo()
        optimizer = GraphOptimizer(repository=repo, minio_storage_service=MagicMock())

        await optimizer._reimport(
            namespace="256237", graph_name="256237-debt_recovery",
            nodes_bytes=NODES, rels_bytes=RELS,
            community_reports=[{"community_id": "L0_C1", "level": 0, "entities": '["7", "8", "999"]'}],
        )

        _, kwargs = repo.save_community_report.call_args
        # 7→500, 8→501; 999 was never in the snapshot and must be dropped
        assert kwargs["report"]["entities"] == ["500", "501"]
