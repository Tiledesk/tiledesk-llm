#!/usr/bin/env python3
"""
docs/MIGLIORIE_DA_FARE.md P1#6: the community report MERGE matches on
(community_id, level), and search_community_reports' CONTAINS scan doesn't
help there — no index backed those lookups. Fixed by indexing
CommunityReport.community_id/.level explicitly, alongside (not instead of)
the existing free-text SEARCHABLE_PROPERTIES indexes, kept for future
full-text search use cases.
"""
from unittest.mock import AsyncMock, Mock

import pytest

from tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository import AsyncFalkorGraphRepository
from tilellm.modules.knowledge_graph_falkor.repository.falkor_repository import FalkorGraphRepository


def _queries_from(mock_execute) -> list:
    return [call.args[0] for call in mock_execute.call_args_list]


class TestAsyncEnsureIndexesForGraph:
    @pytest.mark.asyncio
    async def test_community_report_structured_properties_indexed(self):
        repo = AsyncFalkorGraphRepository.__new__(AsyncFalkorGraphRepository)
        repo._execute_query = AsyncMock(return_value=[])

        await repo._ensure_indexes_for_graph("gname")

        queries = _queries_from(repo._execute_query)
        assert "CREATE INDEX FOR (n:CommunityReport) ON (n.community_id)" in queries
        assert "CREATE INDEX FOR (n:CommunityReport) ON (n.level)" in queries

    @pytest.mark.asyncio
    async def test_free_text_indexes_still_created(self):
        """P1#6 asked to keep the free-text SEARCHABLE_PROPERTIES indexes for
        future use, not remove them — only add the structured one."""
        repo = AsyncFalkorGraphRepository.__new__(AsyncFalkorGraphRepository)
        repo._execute_query = AsyncMock(return_value=[])

        await repo._ensure_indexes_for_graph("gname")

        queries = _queries_from(repo._execute_query)
        assert "CREATE INDEX FOR (n:Entity) ON (n.text)" in queries
        assert "CREATE INDEX FOR (n:CommunityReport) ON (n.summary)" in queries


class _FakeSyncRepo:
    """FalkorGraphRepository is abstract (an unimplemented method predates
    this change, unrelated to it) — call _ensure_indexes_for_graph unbound
    against a minimal duck-typed stand-in instead of instantiating the class."""
    SEARCHABLE_PROPERTIES = FalkorGraphRepository.SEARCHABLE_PROPERTIES
    COMMUNITY_REPORT_INDEXED_PROPERTIES = FalkorGraphRepository.COMMUNITY_REPORT_INDEXED_PROPERTIES

    def __init__(self):
        self._execute_query = Mock(return_value=[])


class TestSyncEnsureIndexesForGraph:
    def test_community_report_structured_properties_indexed(self):
        fake = _FakeSyncRepo()
        FalkorGraphRepository._ensure_indexes_for_graph(fake, "gname")

        queries = _queries_from(fake._execute_query)
        assert "CREATE INDEX FOR (n:CommunityReport) ON (n.community_id)" in queries
        assert "CREATE INDEX FOR (n:CommunityReport) ON (n.level)" in queries

    def test_free_text_indexes_still_created(self):
        fake = _FakeSyncRepo()
        FalkorGraphRepository._ensure_indexes_for_graph(fake, "gname")

        queries = _queries_from(fake._execute_query)
        assert "CREATE INDEX FOR (n:Entity) ON (n.text)" in queries
        assert "CREATE INDEX FOR (n:CommunityReport) ON (n.summary)" in queries
