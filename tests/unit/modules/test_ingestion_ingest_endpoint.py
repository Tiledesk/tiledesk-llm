#!/usr/bin/env python3
"""
POST /api/ingest/md — thin routing: DI (llm_embeddings, repo) -> ingest_document().
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from tilellm.modules.ingestion.ingest.models import IngestMdRequest, IngestMdResult

_ENGINE = {"name": "pinecone", "type": "serverless", "apikey": "k", "vector_size": 1536, "index_name": "idx"}


def _req(**over):
    kw = dict(id="doc1", namespace="ns", engine=_ENGINE, md="---\ntype: document\n---\n\nbody\n")
    kw.update(over)
    return IngestMdRequest(**kw)


class TestIngestMdEndpoint:
    @pytest.mark.asyncio
    async def test_success_returns_result(self):
        from tilellm.modules.ingestion.controllers import ingest_md_endpoint

        expected = IngestMdResult(id="doc1", namespace="ns", chunks_indexed=3, chunk_ids=["a", "b", "c"])
        with patch(
            "tilellm.modules.ingestion.controllers.ingest_md",
            new=AsyncMock(return_value=expected),
        ):
            result = await ingest_md_endpoint(_req())

        assert result == expected

    @pytest.mark.asyncio
    async def test_value_error_becomes_400(self):
        from tilellm.modules.ingestion.controllers import ingest_md_endpoint

        with patch(
            "tilellm.modules.ingestion.controllers.ingest_md",
            new=AsyncMock(side_effect=ValueError("bad source")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ingest_md_endpoint(_req())
        assert exc_info.value.status_code == 400


class TestRouteRegistered:
    def test_ingest_md_route_exists(self):
        from tilellm.modules.ingestion.controllers import router
        paths = {route.path for route in router.routes}
        assert "/api/ingest/md" in paths
