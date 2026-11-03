"""
POST /api/crawl/discover — the pages and files a type=crawl would index, no
vector store write. Same limits as the crawl fields of ItemSingle.
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from tilellm.models.crawl import CrawlDiscoverRequest
from tilellm.modules.ingestion.controllers import crawl_discover_endpoint
from tilellm.shared.outbound_url import BlockedURLError

RESULT = {"pages": [{"url": "https://www.example.org/", "title": "Home", "kind": "page", "depth": 0}],
          "truncated": False, "fetched": 1, "skipped": {"robots": 0}}


@pytest.mark.asyncio
async def test_crawl_options_reach_discover_site():
    req = CrawlDiscoverRequest(source="https://www.example.org/", max_depth=1, max_pages=7,
                               ignore_robots=True, crawl_file_types=["pdf"],
                               browser_headers={"Authorization": "Bearer FINTO"})
    with patch("tilellm.modules.ingestion.controllers.discover_site",
               new=AsyncMock(return_value=RESULT)) as discover:
        result = await crawl_discover_endpoint(req)
    discover.assert_awaited_once_with("https://www.example.org/", max_depth=1, max_pages=7,
                                      ignore_robots=True, file_types=["pdf"],
                                      browser_headers={"Authorization": "Bearer FINTO"},
                                      max_seconds=300)
    assert result.pages[0].title == "Home"


@pytest.mark.asyncio
async def test_max_seconds_reaches_discover_site():
    req = CrawlDiscoverRequest(source="https://www.example.org/", max_seconds=120)
    with patch("tilellm.modules.ingestion.controllers.discover_site",
               new=AsyncMock(return_value=RESULT)) as discover:
        await crawl_discover_endpoint(req)
    assert discover.await_args.kwargs["max_seconds"] == 120


def test_discover_request_max_seconds_bounds():
    assert CrawlDiscoverRequest(source="https://x.org").max_seconds == 300
    for bad in (29, 1801):
        with pytest.raises(ValidationError):
            CrawlDiscoverRequest(source="https://x.org", max_seconds=bad)


@pytest.mark.asyncio
async def test_refused_start_url_is_a_400():
    req = CrawlDiscoverRequest(source="http://169.254.169.254/")
    with patch("tilellm.modules.ingestion.controllers.discover_site",
               new=AsyncMock(side_effect=BlockedURLError("refused"))):
        with pytest.raises(HTTPException) as exc:
            await crawl_discover_endpoint(req)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("bad", [
    {"max_depth": 11}, {"max_pages": 501}, {"max_pages": 0}, {"crawl_file_types": ["exe"]},
    {"namespace": "ns"},  # an ingestion field: rejected, not silently ignored
])
def test_request_limits(bad):
    with pytest.raises(ValidationError):
        CrawlDiscoverRequest(source="https://www.example.org/", **bad)


def test_item_single_keeps_the_same_limits():
    from tilellm.models import Engine
    from tilellm.models.llm import ItemSingle
    engine = Engine(name="qdrant", type="serverless", apikey="", vector_size=1024, index_name="c",
                    host="localhost", port=6333)
    for bad in ({"max_depth": 11}, {"max_pages": 501}, {"crawl_file_types": ["exe"]}):
        with pytest.raises(ValidationError):
            ItemSingle(id="s", source="https://www.example.org/", type="crawl", engine=engine, **bad)
