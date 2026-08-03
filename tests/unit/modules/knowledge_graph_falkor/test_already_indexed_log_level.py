#!/usr/bin/env python3
"""
docs/MIGLIORIE_DA_FARE.md P2#11: _ensure_indexes_for_graph re-issues CREATE INDEX on every
startup/reimport for indexes that already exist, and _execute_query logged those failures at
ERROR — dozens of red lines that look like real breakage (mistaken for one during 2026-07-30
debugging) but are caught and degraded to debug one level up. Fixed by downgrading only the
"already indexed" case to debug inside _execute_query itself, where the log line actually fires.
"""
from unittest.mock import MagicMock, patch

import pytest

from tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository import AsyncFalkorGraphRepository
from tilellm.modules.knowledge_graph_falkor.repository.falkor_repository import FalkorGraphRepository


class TestAsyncExecuteQueryLogLevel:
    @pytest.mark.asyncio
    async def test_already_indexed_logs_at_debug_not_error(self):
        repo = AsyncFalkorGraphRepository.__new__(AsyncFalkorGraphRepository)
        graph = MagicMock()
        graph.name = "gname"
        graph.query = _async_raise(Exception("Attribute 'text' is already indexed"))
        repo._get_graph = MagicMock(return_value=graph)

        with patch("tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.logger") as mock_logger:
            with pytest.raises(Exception):
                await repo._execute_query("CREATE INDEX FOR (n:Entity) ON (n.text)")
            mock_logger.error.assert_not_called()
            mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_real_failure_still_logs_at_error(self):
        repo = AsyncFalkorGraphRepository.__new__(AsyncFalkorGraphRepository)
        graph = MagicMock()
        graph.name = "gname"
        graph.query = _async_raise(Exception("connection reset by peer"))
        repo._get_graph = MagicMock(return_value=graph)

        with patch("tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.logger") as mock_logger:
            with pytest.raises(Exception):
                await repo._execute_query("MATCH (n) RETURN n")
            mock_logger.error.assert_called()


class _FakeSyncRepo:
    """FalkorGraphRepository is abstract (missing find_nodes_by_source_id, predates and is
    unrelated to this change — see test_community_report_index.py) — exercise
    _execute_query unbound against a minimal duck-typed stand-in instead."""
    def __init__(self, graph):
        self._graph = graph

    def _get_graph(self, namespace=None, graph_name=None):
        return self._graph

    def _convert_database_value(self, value):
        return value


class TestSyncExecuteQueryLogLevel:
    def test_already_indexed_logs_at_debug_not_error(self):
        graph = MagicMock()
        graph.name = "gname"
        graph.query.side_effect = Exception("Attribute 'text' is already indexed")
        fake = _FakeSyncRepo(graph)

        with patch("tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.logger") as mock_logger:
            with pytest.raises(Exception):
                FalkorGraphRepository._execute_query(fake, "CREATE INDEX FOR (n:Entity) ON (n.text)")
            mock_logger.error.assert_not_called()
            mock_logger.debug.assert_called()

    def test_real_failure_still_logs_at_error(self):
        graph = MagicMock()
        graph.name = "gname"
        graph.query.side_effect = Exception("connection reset by peer")
        fake = _FakeSyncRepo(graph)

        with patch("tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.logger") as mock_logger:
            with pytest.raises(Exception):
                FalkorGraphRepository._execute_query(fake, "MATCH (n) RETURN n")
            mock_logger.error.assert_called()


def _async_raise(exc):
    async def _f(*args, **kwargs):
        raise exc
    return _f
