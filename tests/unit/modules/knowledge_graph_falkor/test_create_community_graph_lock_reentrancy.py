#!/usr/bin/env python3
"""
create_community_graph holds a Redis distributed lock (_graph_create_lock) on
graph_db_name for its whole duration (extraction + clustering). It then calls
generate_hierarchical_reports, which independently acquires the SAME lock —
a non-reentrant self-deadlock. Every real run failed with "Graph creation for
'<graph>' is already running on another worker" right after extraction
finished, on the very first attempt at clustering (observed live on
namespace 43282, a clean run with no prior crash — see
memory/project_debt_recovery_benchmark.md, 2026-07-29).

Fix: create_community_graph must call the already-locked private variant
(_generate_hierarchical_reports_locked) instead of the public lock-acquiring
wrapper. Other callers of generate_hierarchical_reports (logic.py, standalone
/hierarchical and /leiden-cluster endpoints) are top-level entry points with
no pre-existing lock and must keep using the public method.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@asynccontextmanager
async def _noop_lock(name, ttl=None):
    yield


class TestCreateCommunityGraphLockReentrancy:
    @pytest.mark.asyncio
    async def test_uses_already_locked_variant_for_hierarchical_reports(self):
        from tilellm.modules.knowledge_graph_falkor.services.community_graph_service import (
            CommunityGraphService,
        )

        graph_rag_service = MagicMock()
        graph_rag_service.vector_store_repository = MagicMock()
        graph_rag_service.import_from_vector_store = AsyncMock(
            return_value={"nodes_created": 1, "relationships_created": 1}
        )

        service = CommunityGraphService(graph_rag_service=graph_rag_service)
        service._generate_hierarchical_reports_locked = AsyncMock(return_value={})
        service.generate_hierarchical_reports = AsyncMock(
            side_effect=AssertionError(
                "create_community_graph must not re-acquire the graph lock via "
                "the public generate_hierarchical_reports — it already holds it"
            )
        )

        with patch(
            "tilellm.modules.knowledge_graph_falkor.services.community_graph_service._graph_create_lock",
            _noop_lock,
        ):
            result = await service.create_community_graph(
                namespace="43282",
                engine=MagicMock(type="serverless", index_name="idx"),
                creation_prompt="debt_recovery",
                vector_store_repo=MagicMock(),
                llm=MagicMock(),
                limit=10,
                overwrite=True,
            )

        service._generate_hierarchical_reports_locked.assert_called_once()
        service.generate_hierarchical_reports.assert_not_called()
        assert result["status"] == "success"
