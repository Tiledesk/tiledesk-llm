#!/usr/bin/env python3
"""
POST /api/lgraph/community_summaries was synchronous — one LLM call per Leiden
community, blocking the HTTP request for graphs with many communities (the same
reason /build and /leiden already run via TaskIQ). Made async, mirroring the
existing build/leiden dispatch pattern exactly: controller enqueues
task_lgraph_community_summaries.kiq(...) and returns {task_id, status: "queued"};
the actual work (summarize_communities_lgraph) now runs in the worker, polled via
the existing GET /api/lgraph/tasks/{id}.
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from tilellm.models import Engine
from tilellm.modules.lgraph.models.schemas import LGraphCommunitySummarizationRequest


def _request():
    return LGraphCommunitySummarizationRequest(
        namespace="asl-bari",
        engine=Engine(name="qdrant", index_name="regionepuglia"),
        gptkey="sk-test",
    )


class TestCommunitySummariesDispatch:
    @pytest.mark.asyncio
    async def test_dispatches_via_taskiq_and_returns_task_id(self):
        from tilellm.modules.lgraph import controllers

        mock_task = Mock(task_id="abc-123")
        mock_kiq = AsyncMock(return_value=mock_task)

        with patch.object(controllers, "ENABLE_TASKIQ", True), \
             patch.object(controllers, "task_lgraph_community_summaries", Mock(kiq=mock_kiq)):
            result = await controllers.generate_community_summaries(_request())

        assert result.task_id == "abc-123"
        assert result.status == "queued"
        mock_kiq.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_501_when_taskiq_disabled(self):
        from tilellm.modules.lgraph import controllers

        with patch.object(controllers, "ENABLE_TASKIQ", False):
            with pytest.raises(HTTPException) as exc_info:
                await controllers.generate_community_summaries(_request())

        assert exc_info.value.status_code == 501


class TestCommunitySummariesTask:
    @pytest.mark.asyncio
    async def test_task_runs_summarize_communities_lgraph_and_returns_payload(self, monkeypatch):
        from tilellm.modules.task_executor import tasks

        fake_response = Mock()
        fake_response.model_dump.return_value = {"status": "success", "communities_processed": 3}
        monkeypatch.setattr(
            tasks, "summarize_communities_lgraph", AsyncMock(return_value=fake_response)
        )

        request_dict = _request().model_dump(mode="python")
        request_dict["gptkey"] = "sk-test"  # SecretStr round-trips as plain str via the queue

        result = await tasks.task_lgraph_community_summaries(request_dict)

        assert result == {"status": "success", "communities_processed": 3}
