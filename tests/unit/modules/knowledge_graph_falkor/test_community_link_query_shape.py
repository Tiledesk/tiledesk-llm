#!/usr/bin/env python3
"""
The BELONGS_TO_COMMUNITY link query must bind the report node once, outside the
UNWIND, and must chunk large entity lists.

It used to read:

    UNWIND $entity_ids as entity_id
    MATCH (e) WHERE id(e) = entity_id
    MATCH (c) WHERE id(c) = $report_id      <-- inside the UNWIND scope
    MERGE (e)-[:BELONGS_TO_COMMUNITY]->(c)

so the report node was re-scanned for every single entity. On namespace 256237
(2026-07-30) one community carries 6 972 entities against a 10 000-node graph —
~70 M scans in one query — and FalkorDB died mid-restore ("Connection closed by
server"), losing the whole reimport back to the last RDB snapshot. Same
cartesian-explosion class that get_all_nodes_and_relationships already documents
avoiding by splitting its queries.

This was latent: before the entities payload was parsed correctly the link query
never ran at all, so fixing the parsing is what first exposed it.
"""
from unittest.mock import AsyncMock

import pytest


def _repo():
    from tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository import (
        AsyncFalkorGraphRepository,
    )
    return AsyncFalkorGraphRepository.__new__(AsyncFalkorGraphRepository)


async def _capture(repo, entities):
    calls = []

    async def fake_execute(query, params=None, **kwargs):
        calls.append((query, params or {}))
        if "MERGE (c:CommunityReport" in query:
            return [{"node_id": 99}]
        return []

    repo._execute_query = fake_execute
    await repo.save_community_report(
        community_id="L0_C1",
        report={"title": "t", "summary": "s", "entities": entities},
        level=0,
        namespace="256237",
        graph_name="256237-debt_recovery",
    )
    return [(q, p) for q, p in calls if "BELONGS_TO_COMMUNITY" in q]


class TestLinkQueryShape:
    @pytest.mark.asyncio
    async def test_report_node_is_bound_before_the_unwind(self):
        repo = _repo()
        link_calls = await _capture(repo, [1, 2, 3])

        assert link_calls, "no link query was issued"
        query = link_calls[0][0]
        assert "UNWIND" in query
        match_c = query.index("id(c) = $report_id")
        unwind = query.index("UNWIND")
        assert match_c < unwind, (
            "the report node must be matched once BEFORE the UNWIND, otherwise it is "
            f"re-scanned per entity:\n{query}"
        )

    @pytest.mark.asyncio
    async def test_large_entity_lists_are_chunked(self):
        repo = _repo()
        link_calls = await _capture(repo, list(range(6972)))

        assert len(link_calls) > 1, "6972 entities must not go out as a single query"
        sizes = [len(p["entity_ids"]) for _, p in link_calls]
        assert sum(sizes) == 6972, f"entities lost or duplicated while chunking: {sum(sizes)}"
        assert max(sizes) <= 1000, f"chunk too large: {max(sizes)}"

    @pytest.mark.asyncio
    async def test_small_lists_still_go_out_in_one_query(self):
        repo = _repo()
        link_calls = await _capture(repo, [7, 8, 9])

        assert len(link_calls) == 1
        assert link_calls[0][1]["entity_ids"] == [7, 8, 9]
