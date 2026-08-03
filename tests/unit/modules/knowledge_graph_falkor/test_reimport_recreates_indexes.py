#!/usr/bin/env python3
"""
docs/MIGLIORIE_DA_FARE.md P1#7: _reimport wipes the FalkorDB graph and
restores it from Parquet, but never recreated the indexes afterward (wipe
via delete_nodes_by_metadata does not preserve index definitions in
FalkorDB). Fixed by calling repository.ensure_indexes_for_graph(graph_name)
at the end of _reimport, same mechanism used by normal graph creation and
by P1#6's CommunityReport.community_id/.level index.
"""
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


def _parquet(rows) -> bytes:
    import io
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    return buf.getvalue()


class TestReimportRecreatesIndexes:
    @pytest.mark.asyncio
    async def test_reimport_calls_ensure_indexes_for_graph(self):
        from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import GraphOptimizer

        repo = MagicMock()
        repo.delete_nodes_by_metadata = AsyncMock(return_value={"nodes_deleted": 42})
        repo.batch_create_nodes = AsyncMock(return_value={"banca abc": "new-1"})
        repo.batch_create_relationships = AsyncMock(return_value=1)
        repo.save_community_report = AsyncMock(return_value="r1")
        repo.ensure_indexes_for_graph = AsyncMock(return_value=None)
        repo._normalize_name = lambda s: s.strip().lower()

        optimizer = GraphOptimizer(repository=repo, minio_storage_service=MagicMock())

        await optimizer._reimport(
            namespace="256237",
            graph_name="256237-debt_recovery",
            nodes_bytes=_parquet([{"id": "1", "label": "ORGANIZATION", "name": "Banca ABC", "source_ids": "[]"}]),
            rels_bytes=_parquet([{"id": "9", "type": "HAS_LOAN", "source_id": "1", "target_id": "1"}]),
            community_reports=[{"community_id": "L0_C1", "level": 0}],
        )

        repo.ensure_indexes_for_graph.assert_awaited_once_with("256237-debt_recovery")
