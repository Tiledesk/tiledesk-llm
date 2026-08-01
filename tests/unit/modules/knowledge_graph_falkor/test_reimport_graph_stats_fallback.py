#!/usr/bin/env python3
"""
reimport_graph (POST /api/kg-falkor/reimport) only knew about
GraphOptimizer's own graph_snapshots/ format. When that's missing (e.g.
namespace 256237, 2026-07-29: FalkorDB graph wiped, only the /create-run's
auto-saved community_graph_service._save_stats snapshot exists), it must
fall back to that snapshot via load_stats_snapshot + the schema bridge.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from minio.error import S3Error

from tilellm.models.vector_store import Engine


def _engine():
    return Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="debt-recovery-hybrid")


def _not_found():
    return S3Error(response=MagicMock(), code="NoSuchKey", message="not found",
                    resource="x", request_id="req", host_id="host")


class TestReimportGraphStatsFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_stats_snapshot_when_graph_snapshot_missing(self):
        from tilellm.modules.knowledge_graph_falkor.models.schemas import GraphReimportRequest
        from tilellm.modules.knowledge_graph_falkor import logic

        request = GraphReimportRequest(
            namespace="256237",
            engine=_engine(),
            graph_db_name="256237-debt_recovery",
            creation_prompt="debt_recovery",
            snapshot_timestamp="20260729_151509",
        )

        minio = MagicMock()
        minio.load_graph_snapshot.side_effect = _not_found()
        minio.load_stats_snapshot.return_value = {
            "entities": b"ENTITIES_BYTES",
            "relationships": b"RELATIONSHIPS_BYTES",
            "community_reports": [{"community_id": "L0_C1"}],
            "timestamp": "20260729_151509",
        }

        optimizer_instance = MagicMock()
        optimizer_instance._reimport = AsyncMock(return_value=None)

        with patch.object(logic, "graph_service", MagicMock()), \
             patch.object(logic, "repository", MagicMock()), \
             patch("tilellm.modules.knowledge_graph_falkor.services.minio_storage.get_minio_storage_service", return_value=minio), \
             patch("tilellm.modules.knowledge_graph_falkor.services.graph_optimizer.GraphOptimizer", return_value=optimizer_instance), \
             patch("tilellm.modules.knowledge_graph_falkor.services.graph_optimizer.convert_stats_snapshot_to_optimizer_format",
                   return_value=(b"NODES_BYTES", b"RELS_BYTES")) as convert_mock:

            result = await logic.reimport_graph.__wrapped__(request, repo=MagicMock())

        minio.load_stats_snapshot.assert_called_once_with(
            namespace="256237", index_name="debt-recovery-hybrid", index_type=_engine().type,
            timestamp="20260729_151509",
        )
        convert_mock.assert_called_once_with(b"ENTITIES_BYTES", b"RELATIONSHIPS_BYTES")
        optimizer_instance._reimport.assert_called_once()
        _, call_kwargs = optimizer_instance._reimport.call_args
        assert call_kwargs["nodes_bytes"] == b"NODES_BYTES"
        assert call_kwargs["rels_bytes"] == b"RELS_BYTES"
        assert call_kwargs["community_reports"] == [{"community_id": "L0_C1"}]
        assert result["status"] == "success"
        assert result["snapshot_timestamp"] == "20260729_151509"
