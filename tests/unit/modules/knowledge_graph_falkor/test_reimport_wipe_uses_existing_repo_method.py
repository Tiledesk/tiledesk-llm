#!/usr/bin/env python3
"""
GraphOptimizer._reimport called self.repository.delete_all_nodes(...) — a method
that exists on no repository in the codebase (grep: the call site is its only
occurrence). Dead code never exercised until POST /api/kg-falkor/reimport was
run for the first time (2026-07-30, namespace 256237):
    AttributeError: 'AsyncFalkorGraphRepository' object has no attribute
    'delete_all_nodes'. Did you mean: 'delete_node'?

Fix: use delete_nodes_by_metadata(namespace=..., graph_name=...) — the real,
already-paginated wipe. Passing ONLY namespace (no engine_name/engine_type) is
deliberate: it matches CommunityReport nodes too (they carry namespace but are
saved with engine_name=None/engine_type=None), so the wipe is complete before
the snapshot — entities AND community reports — is restored.
"""
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


def _parquet(rows) -> bytes:
    import io
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    return buf.getvalue()


class TestReimportWipe:
    @pytest.mark.asyncio
    async def test_wipes_via_delete_nodes_by_metadata(self):
        from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import GraphOptimizer

        repo = MagicMock()
        repo.delete_nodes_by_metadata = AsyncMock(return_value={"nodes_deleted": 42})
        repo.batch_create_nodes = AsyncMock(return_value={"banca abc": "new-1"})
        repo.batch_create_relationships = AsyncMock(return_value=1)
        repo.save_community_report = AsyncMock(return_value="r1")
        repo._normalize_name = lambda s: s.strip().lower()
        # delete_all_nodes must not be reached; a MagicMock would silently succeed,
        # so make any use of it an explicit failure.
        repo.delete_all_nodes = MagicMock(side_effect=AssertionError("delete_all_nodes does not exist on any repository"))

        optimizer = GraphOptimizer(repository=repo, minio_storage_service=MagicMock())

        await optimizer._reimport(
            namespace="256237",
            graph_name="256237-debt_recovery",
            nodes_bytes=_parquet([{"id": "1", "label": "ORGANIZATION", "name": "Banca ABC", "source_ids": "[]"}]),
            rels_bytes=_parquet([{"id": "9", "type": "HAS_LOAN", "source_id": "1", "target_id": "1"}]),
            community_reports=[{"community_id": "L0_C1", "level": 0}],
        )

        repo.delete_nodes_by_metadata.assert_awaited_once_with(
            namespace="256237", graph_name="256237-debt_recovery"
        )
        repo.batch_create_nodes.assert_awaited()
        repo.save_community_report.assert_awaited_once()
