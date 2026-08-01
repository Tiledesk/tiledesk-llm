#!/usr/bin/env python3
"""
generate_hierarchical_reports (the public, lock-acquiring entry point used by
standalone /hierarchical and /leiden-cluster) delegates to the private
_generate_hierarchical_reports_locked, but passed the wrong keyword name
(graph_name instead of graph_name_to_use) — a pre-existing bug that was always
masked when called nested under create_community_graph's own lock (it never
reached this line, failing earlier on lock re-entrancy instead — see
test_create_community_graph_lock_reentrancy.py). Surfaced when calling
POST /api/kg-falkor/hierarchical directly on a graph with no prior lock
(namespace 43282, 2026-07-29):
  TypeError: _generate_hierarchical_reports_locked() got an unexpected
  keyword argument 'graph_name'
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@asynccontextmanager
async def _noop_lock(name, ttl=None):
    yield


class TestGenerateHierarchicalReportsParamName:
    @pytest.mark.asyncio
    async def test_delegates_with_correct_keyword(self):
        from tilellm.modules.knowledge_graph_falkor.services.community_graph_service import (
            CommunityGraphService,
        )

        service = CommunityGraphService(graph_rag_service=MagicMock())
        service._generate_hierarchical_reports_locked = AsyncMock(return_value={})

        with patch(
            "tilellm.modules.knowledge_graph_falkor.services.community_graph_service._graph_create_lock",
            _noop_lock,
        ):
            await service.generate_hierarchical_reports(
                namespace="43282",
                graph_name="43282-debt_recovery",
                engine=MagicMock(),
                llm=MagicMock(),
                vector_store_repo=MagicMock(),
            )

        service._generate_hierarchical_reports_locked.assert_called_once()
        _, kwargs = service._generate_hierarchical_reports_locked.call_args
        assert kwargs["graph_name_to_use"] == "43282-debt_recovery"
        assert "graph_name" not in kwargs
