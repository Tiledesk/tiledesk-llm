#!/usr/bin/env python3
"""
task_falkor_graph_create must ignore a re-delivery of a task that already
completed, because its first action is destructive (overwrite=True wipes all
entity nodes + MinIO checkpoints before re-extracting).

Root cause of the 256237 data loss (2026-07-29, see
memory/project_debt_recovery_benchmark.md): taskiq_redis's
RedisStreamBroker.listen() runs `xautoclaim(min_idle_time=idle_timeout)` on
EVERY loop iteration (~2s). Our IDLE_TIMEOUT_MS is 3 600 000 (60 min) but a
GraphRAG extraction on a large namespace takes ~2 h, so the still-unacked
in-flight message crossed the 60-min idle threshold and was re-delivered as
the SAME task_id. With --max-async-tasks 1 the duplicate queued behind the
original, then started 7 ms after it finished and wiped 10 908 freshly
extracted nodes. No new stream entry is created by xautoclaim (xlen and
entries-read both stayed at 1), which is why the duplicate was invisible in
the queue.

Raising IDLE_TIMEOUT_MS alone only moves the threshold; this guard makes the
task idempotent no matter which re-delivery path fires — mirroring the check
_startup_reclaim already performs for the startup reclaim path.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ctx(task_id: str):
    ctx = MagicMock()
    ctx.message.task_id = task_id
    return ctx


@pytest.fixture
def falkor_logic_mock():
    logic = MagicMock()
    logic.create_graph = AsyncMock(return_value={"status": "success", "nodes_created": 10908})
    return logic


class TestFalkorCreateDuplicateDeliveryGuard:
    @pytest.mark.asyncio
    async def test_skips_when_result_already_exists(self, falkor_logic_mock):
        from tilellm.modules.task_executor import tasks

        backend = MagicMock()
        backend.is_result_ready = AsyncMock(return_value=True)

        with patch.object(tasks.broker, "result_backend", backend), \
             patch("tilellm.modules.knowledge_graph_falkor.logic.create_graph",
                   falkor_logic_mock.create_graph):
            result = await tasks.task_falkor_graph_create.__wrapped__(
                {"namespace": "256237", "engine": {"name": "qdrant", "index_name": "idx"}},
                ctx=_ctx("22ff045e43f1496bb834ce6635742c09"),
            )

        falkor_logic_mock.create_graph.assert_not_called()
        assert result["status"] == "skipped_duplicate"

    @pytest.mark.asyncio
    async def test_runs_normally_when_no_previous_result(self, falkor_logic_mock):
        from tilellm.modules.task_executor import tasks

        backend = MagicMock()
        backend.is_result_ready = AsyncMock(return_value=False)

        with patch.object(tasks.broker, "result_backend", backend), \
             patch("tilellm.modules.knowledge_graph_falkor.logic.create_graph",
                   falkor_logic_mock.create_graph):
            result = await tasks.task_falkor_graph_create.__wrapped__(
                {"namespace": "256237", "engine": {"name": "qdrant", "index_name": "idx"}},
                ctx=_ctx("fresh-task-id"),
            )

        falkor_logic_mock.create_graph.assert_called_once()
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_proceeds_when_guard_check_itself_fails(self, falkor_logic_mock):
        """A broken/unreachable result backend must not block real work."""
        from tilellm.modules.task_executor import tasks

        backend = MagicMock()
        backend.is_result_ready = AsyncMock(side_effect=ConnectionError("redis down"))

        with patch.object(tasks.broker, "result_backend", backend), \
             patch("tilellm.modules.knowledge_graph_falkor.logic.create_graph",
                   falkor_logic_mock.create_graph):
            result = await tasks.task_falkor_graph_create.__wrapped__(
                {"namespace": "256237", "engine": {"name": "qdrant", "index_name": "idx"}},
                ctx=_ctx("some-task-id"),
            )

        falkor_logic_mock.create_graph.assert_called_once()
        assert result["status"] == "success"
