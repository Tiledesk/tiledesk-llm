#!/usr/bin/env python3
"""
Restoring community reports during _reimport must rebuild BELONGS_TO_COMMUNITY
against the NEW node ids, and must tolerate parquet round-tripped payloads.

Two defects surfaced on the first real /reimport run (256237, 2026-07-30):

1. `report["entities"]` comes back from parquet as a JSON *string*
   ('["0", "1", ...]'), so save_community_report iterated it character by
   character:  ValueError: invalid literal for int() with base 10: '['

2. Worse, and silent: those ids are the OLD FalkorDB ids of the graph that
   _reimport just wiped. Nodes are recreated with fresh ids, so linking by the
   stored id would attach entities to whatever unrelated node now happens to
   own that integer — corrupt edges rather than missing ones. _reimport already
   builds an old→new map for relationships; community reports must use it too.
"""
import io
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


def _parquet(rows) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    return buf.getvalue()


class TestSaveCommunityReportAcceptsJsonEncodedEntities:
    @pytest.mark.asyncio
    async def test_json_string_entities_are_parsed(self):
        from tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository import (
            AsyncFalkorGraphRepository,
        )

        repo = AsyncFalkorGraphRepository.__new__(AsyncFalkorGraphRepository)
        calls = []

        async def fake_execute(query, params=None, **kwargs):
            calls.append((query, params or {}))
            if "MERGE (c:CommunityReport" in query:
                return [{"node_id": 99}]
            return []

        repo._execute_query = fake_execute

        await repo.save_community_report(
            community_id="L0_C1",
            report={"title": "t", "summary": "s", "entities": '["11", "22"]'},
            level=0,
            namespace="256237",
            graph_name="256237-debt_recovery",
        )

        link_calls = [p for q, p in calls if "BELONGS_TO_COMMUNITY" in q]
        assert link_calls, "entities were never linked"
        assert link_calls[0]["entity_ids"] == [11, 22]


class TestReimportRemapsCommunityEntityIds:
    @pytest.mark.asyncio
    async def test_entities_are_remapped_to_new_node_ids(self):
        from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import GraphOptimizer

        repo = MagicMock()
        repo.delete_nodes_by_metadata = AsyncMock(return_value={"nodes_deleted": 0})
        repo.batch_create_relationships = AsyncMock(return_value=0)
        repo.save_community_report = AsyncMock(return_value="r1")
        repo.ensure_indexes_for_graph = AsyncMock(return_value=None)
        repo._normalize_name = lambda s: s.strip().lower()
        # Old ids 7/8 in the snapshot; recreation assigns 500/501.
        repo.batch_create_nodes = AsyncMock(
            return_value={"banca abc": "500", "mario bianchi": "501"}
        )

        optimizer = GraphOptimizer(repository=repo, minio_storage_service=MagicMock())

        await optimizer._reimport(
            namespace="256237",
            graph_name="256237-debt_recovery",
            nodes_bytes=_parquet([
                {"id": "7", "label": "ORGANIZATION", "name": "Banca ABC", "source_ids": "[]"},
                {"id": "8", "label": "PERSON", "name": "Mario Bianchi", "source_ids": "[]"},
            ]),
            rels_bytes=_parquet([]),
            community_reports=[{"community_id": "L0_C1", "level": 0, "entities": '["7", "8"]'}],
        )

        repo.save_community_report.assert_awaited_once()
        _, kwargs = repo.save_community_report.call_args
        entities = kwargs["report"]["entities"]
        assert entities == ["500", "501"], f"stale ids leaked into the restored report: {entities}"

    @pytest.mark.asyncio
    async def test_unresolvable_entities_are_dropped_not_passed_through(self):
        from tilellm.modules.knowledge_graph_falkor.services.graph_optimizer import GraphOptimizer

        repo = MagicMock()
        repo.delete_nodes_by_metadata = AsyncMock(return_value={"nodes_deleted": 0})
        repo.batch_create_relationships = AsyncMock(return_value=0)
        repo.save_community_report = AsyncMock(return_value="r1")
        repo.ensure_indexes_for_graph = AsyncMock(return_value=None)
        repo._normalize_name = lambda s: s.strip().lower()
        repo.batch_create_nodes = AsyncMock(return_value={"banca abc": "500"})

        optimizer = GraphOptimizer(repository=repo, minio_storage_service=MagicMock())

        await optimizer._reimport(
            namespace="256237",
            graph_name="256237-debt_recovery",
            nodes_bytes=_parquet([{"id": "7", "label": "ORGANIZATION", "name": "Banca ABC", "source_ids": "[]"}]),
            rels_bytes=_parquet([]),
            # "999" was truncated out of the entities parquet by the resultset_size cap
            community_reports=[{"community_id": "L0_C1", "level": 0, "entities": ["7", "999"]}],
        )

        _, kwargs = repo.save_community_report.call_args
        assert kwargs["report"]["entities"] == ["500"]
