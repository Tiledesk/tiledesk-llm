"""
_query_temporal used to hardcode qa.top_k = 20, ignoring request.top_k
(unlike _query_semantic, which already respected it via _build_qa).
Also locks in that DigestQueryRequest silently ignores unknown fields
(e.g. a caller mistakenly passing lgraph's vector_top_k) instead of
raising a validation error.
"""
from datetime import date
from unittest.mock import AsyncMock

import pytest

from tilellm.models.vector_store import Engine
from tilellm.modules.temporal_digest.models.schemas import DigestQueryRequest
from tilellm.modules.temporal_digest.services.digest_service import DigestService


def _engine():
    return Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx")


class _FakeRetrieval:
    chunks: list = []
    metadata: list = []


@pytest.mark.asyncio
async def test_query_temporal_respects_request_top_k():
    request = DigestQueryRequest(
        question="cosa e' successo?",
        namespace="asl-bari",
        engine=_engine(),
        top_k=42,
    )
    repo = AsyncMock()
    repo.get_chunks_from_repo.return_value = _FakeRetrieval()

    await DigestService()._query_temporal(request, repo=repo, llm=AsyncMock())

    used_qa = repo.get_chunks_from_repo.call_args[0][0]
    assert used_qa.top_k == 42


def test_digest_query_request_ignores_unknown_fields():
    request = DigestQueryRequest(
        question="q",
        namespace="ns",
        engine=_engine(),
        vector_top_k=99,
    )
    assert not hasattr(request, "vector_top_k")
