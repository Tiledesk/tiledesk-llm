#!/usr/bin/env python3
"""
POST /api/lgraph/community_summaries crashed on every real call:

    AttributeError: 'LGraphCommunitySummarizationRequest' object has no attribute 'debug'

_create_callback_handler (called from inject_llm_chat_async's async_wrapper) read
question.debug unconditionally. Unlike LGraphQARequest/ItemSingle, which do carry a
.debug field, LGraphCommunitySummarizationRequest never got one — the same class of
gap as the earlier ItemSingle/.llm crash (docs/MIGLIORIE_DA_FARE.md), just on a
different attribute. Fixed at the decorator level (getattr with a False default)
rather than patching every model that happens to be missing the field — protects
every current and future inject_llm_chat_async caller, not just this one.
"""
from unittest.mock import AsyncMock, Mock

import pytest

from tilellm.shared.utility import _create_callback_handler


class TestCreateCallbackHandlerMissingDebugAttr:
    def test_no_debug_attribute_returns_none_instead_of_crashing(self):
        question = Mock(spec=["llm"])  # no .debug at all, mirrors LGraphCommunitySummarizationRequest
        question.llm = "openai"

        result = _create_callback_handler(question, llm=Mock())

        assert result is None

    def test_debug_true_still_builds_a_handler(self):
        question = Mock()
        question.debug = True
        question.llm = "openai"

        result = _create_callback_handler(question, llm=Mock())

        assert result is not None


class TestSummarizeCommunitiesLgraphWrapper:
    @pytest.mark.asyncio
    async def test_wrapper_does_not_crash_on_missing_debug_attribute(self, monkeypatch):
        from tilellm.modules.lgraph import logic as lgraph_logic
        from tilellm.modules.lgraph.models.schemas import LGraphCommunitySummarizationRequest
        from tilellm.models import Engine

        monkeypatch.setattr(
            lgraph_logic, "_summarize_communities_lgraph_core", AsyncMock(return_value="core-result")
        )
        monkeypatch.setattr(
            "tilellm.shared.utility._create_llm_instance", AsyncMock(return_value=Mock())
        )
        monkeypatch.setattr(
            "tilellm.shared.utility._create_embedding_instance", AsyncMock(return_value="fake-embeddings")
        )

        request = LGraphCommunitySummarizationRequest(
            namespace="asl-bari",
            engine=Engine(name="qdrant", index_name="regionepuglia"),
            gptkey="sk-test",
        )

        result = await lgraph_logic.summarize_communities_lgraph(request)
        assert result == "core-result"
