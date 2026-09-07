"""
_fetch_source_chunks used to filter by date via a vector-store metadata
range query ({date_metadata_field: {"$gte": ..., "$lte": ...}}). Real
production bug (2026-08-08, ASL Bari corpus on Pinecone):

1. Pinecone's $gte/$lte require a *number* — a plain ISO date string 400s
   outright ("the $gte operator must be followed by a number, got string
   instead"), so this path never worked on Pinecone at all.
2. Even fixing the type, the date lives under whatever field name/format the
   tenant's ingestion used (this corpus: "data_determina", DD/MM/YYYY —
   default date_metadata_field is "date", which doesn't exist here).

Fix: fetch the whole namespace (get_all_obj_namespace, no cap since the
2026-08-06 fix) and filter by date in Python — works regardless of vector
store backend, field format, or whether the field is ISO or DD/MM/YYYY.
"""
from datetime import date
from unittest.mock import AsyncMock

import pytest

from tilellm.models.schemas import RepositoryItems, RepositoryQueryResult
from tilellm.models.vector_store import Engine
from tilellm.modules.temporal_digest.models.schemas import DigestGenerationRequest
from tilellm.modules.temporal_digest.services.digest_service import DigestService


def _engine():
    return Engine(name="pinecone", type="serverless", apikey="k", index_name="idx")


def _match(id_, text, metadata):
    return RepositoryQueryResult(id=id_, text=text, metadata=metadata)


def _request(**over):
    kw = dict(namespace="aslbari", date_from=date(2026, 7, 27), engine=_engine())
    kw.update(over)
    return DigestGenerationRequest(**kw)


@pytest.mark.asyncio
async def test_filters_by_iso_date_field():
    repo = AsyncMock()
    repo.get_all_obj_namespace.return_value = RepositoryItems(matches=[
        _match("c1", "dentro range", {"date": "2026-07-27"}),
        _match("c2", "fuori range", {"date": "2026-08-01"}),
    ])
    request = _request(date_metadata_field="date")

    result = await DigestService()._fetch_source_chunks(
        repo=repo, request=request, date_from_str="2026-07-25", date_to_str="2026-07-31",
    )

    assert result.chunks == ["dentro range"]


@pytest.mark.asyncio
async def test_filters_by_italian_date_format():
    """The real ASL Bari corpus: data_determina in DD/MM/YYYY, not ISO."""
    repo = AsyncMock()
    repo.get_all_obj_namespace.return_value = RepositoryItems(matches=[
        _match("c1", "27 luglio", {"data_determina": "27/07/2026"}),
        _match("c2", "agosto", {"data_determina": "01/08/2026"}),
    ])
    request = _request(date_metadata_field="data_determina")

    result = await DigestService()._fetch_source_chunks(
        repo=repo, request=request, date_from_str="2026-07-25", date_to_str="2026-07-31",
    )

    assert result.chunks == ["27 luglio"]


@pytest.mark.asyncio
async def test_excludes_digest_type_chunks():
    repo = AsyncMock()
    repo.get_all_obj_namespace.return_value = RepositoryItems(matches=[
        _match("c1", "atto reale", {"date": "2026-07-27"}),
        _match("c2", "digest gia' indicizzato", {"date": "2026-07-27", "digest_type": "digest"}),
    ])
    request = _request(date_metadata_field="date")

    result = await DigestService()._fetch_source_chunks(
        repo=repo, request=request, date_from_str="2026-07-25", date_to_str="2026-07-31",
    )

    assert result.chunks == ["atto reale"]


@pytest.mark.asyncio
async def test_chunks_with_missing_or_unparseable_date_excluded():
    repo = AsyncMock()
    repo.get_all_obj_namespace.return_value = RepositoryItems(matches=[
        _match("c1", "senza data", {}),
        _match("c2", "data sporca", {"date": "non-una-data"}),
    ])
    request = _request(date_metadata_field="date")

    result = await DigestService()._fetch_source_chunks(
        repo=repo, request=request, date_from_str="2026-07-25", date_to_str="2026-07-31",
    )

    assert result.chunks == []


@pytest.mark.asyncio
async def test_retrieval_failure_returns_empty_not_raise():
    repo = AsyncMock()
    repo.get_all_obj_namespace.side_effect = RuntimeError("boom")
    request = _request(date_metadata_field="date")

    result = await DigestService()._fetch_source_chunks(
        repo=repo, request=request, date_from_str="2026-07-25", date_to_str="2026-07-31",
    )

    assert result.chunks == []
