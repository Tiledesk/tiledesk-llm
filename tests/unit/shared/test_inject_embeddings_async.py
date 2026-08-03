#!/usr/bin/env python3
"""
/api/v2/ingestion and /api/ingest/md both crashed on the first real (non-unit-test) call:

    AttributeError: 'ItemSingle' object has no attribute 'llm'

Both ingest_v2 and ingest_md were decorated with @inject_llm_chat_async, which
unconditionally reads question.llm (and builds a chat LLM nobody downstream uses —
ingestion only needs llm_embeddings; situated_context, when enabled, builds its own LLM
from config in _apply_situated_context). ItemSingle/IngestConfig are plain BaseModel, not
QuestionAnswer-derived, so they have no .llm field. Existing tests never caught this
because they patch ingest_document/_ingest_v2_core directly, bypassing the decorator
stack entirely — the exact gap that let this ship.

Fix: new inject_embeddings_async decorator (only builds llm_embeddings, no chat LLM),
used by both ingest_md and ingest_v2 instead of inject_llm_chat_async.
"""
from unittest.mock import AsyncMock, Mock

import pytest

from tilellm.models import Engine
from tilellm.shared.utility import inject_embeddings_async


class _EmbeddingOnlyQuestion:
    """Mirrors ItemSingle/IngestConfig's shape: has .embedding/.gptkey, no .llm."""
    def __init__(self):
        self.embedding = "text-embedding-3-small"
        self.gptkey = Mock(get_secret_value=Mock(return_value="sk-test"))


class TestInjectEmbeddingsAsync:
    @pytest.mark.asyncio
    async def test_injects_embeddings_without_requiring_llm_attribute(self, monkeypatch):
        monkeypatch.setattr(
            "tilellm.shared.utility._create_embedding_instance",
            AsyncMock(return_value="fake-embeddings"),
        )

        received = {}

        @inject_embeddings_async
        async def handler(question, llm_embeddings=None, embedding_config_key=None, **kwargs):
            received["llm_embeddings"] = llm_embeddings
            received["embedding_config_key"] = embedding_config_key
            return "ok"

        result = await handler(_EmbeddingOnlyQuestion())

        assert result == "ok"
        assert received["llm_embeddings"] == "fake-embeddings"
        assert received["embedding_config_key"] is not None


class TestIngestionWrappersDoNotRequireLlmAttribute:
    @pytest.mark.asyncio
    async def test_ingest_v2_wrapper(self, monkeypatch):
        from tilellm.models import ItemSingle
        from tilellm.modules.api_v2.services import ingestion_v2_service as svc

        monkeypatch.setattr(svc, "_ingest_v2_core", AsyncMock(return_value="core-result"))
        monkeypatch.setattr(
            "tilellm.shared.utility._create_embedding_instance",
            AsyncMock(return_value="fake-embeddings"),
        )

        item = ItemSingle(
            id="doc1", type="text", content="hello world",
            namespace="ns", engine=Engine(), gptkey="sk-test",
        )

        result = await svc.ingest_v2(item)
        assert result == "core-result"

    @pytest.mark.asyncio
    async def test_ingest_md_wrapper(self, monkeypatch):
        from tilellm.modules.ingestion.ingest import service as svc

        monkeypatch.setattr(svc, "ingest_document", AsyncMock(return_value="core-result"))
        monkeypatch.setattr(
            "tilellm.shared.utility._create_embedding_instance",
            AsyncMock(return_value="fake-embeddings"),
        )

        request = svc.IngestMdRequest(
            id="doc1", namespace="ns", engine=Engine(), md="# hello", gptkey="sk-test",
        )

        result = await svc.ingest_md(request)
        assert result == "core-result"
