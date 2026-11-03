"""Fail fast on unreachable hosts: classifier, per-host breaker, no browser fallbacks."""
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
import urllib3.exceptions as u3

from tilellm.tools import document_tools as dt
from tilellm.tools import unreachable_hosts as uh
from tilellm.tools.unreachable_hosts import UnreachableHostError, is_unreachable_host_error

URL = "https://www.dead.example/a"
URL2 = "https://WWW.dead.example/b"


@pytest.fixture(autouse=True)
def _clean():
    uh.reset()
    yield
    uh.reset()


def _conn_error():
    try:
        try:
            raise u3.NewConnectionError(None, "boom")
        except u3.NewConnectionError as e:
            raise requests.exceptions.ConnectionError(e)
    except requests.exceptions.ConnectionError as outer:
        return outer


def _http_error(code):
    r = requests.Response()
    r.status_code = code
    return requests.exceptions.HTTPError(response=r)


def test_classifier_connection_level():
    assert is_unreachable_host_error(requests.exceptions.ConnectTimeout())
    assert is_unreachable_host_error(_conn_error())
    assert is_unreachable_host_error(socket.gaierror(-2, "unknown"))
    assert is_unreachable_host_error(ConnectionRefusedError())
    try:
        try:
            raise socket.gaierror(-2, "x")
        except OSError as e:
            raise ValueError("wrapped") from e
    except ValueError as w:
        assert is_unreachable_host_error(w)


def test_classifier_not_connection_level():
    assert not is_unreachable_host_error(_http_error(404))
    assert not is_unreachable_host_error(requests.exceptions.ReadTimeout())
    assert not is_unreachable_host_error(requests.exceptions.SSLError())
    assert not is_unreachable_host_error(ValueError("All fallbacks failed"))


@pytest.mark.asyncio
async def test_connect_timeout_skips_fallbacks_and_trips_breaker():
    scrape = AsyncMock(side_effect=requests.exceptions.ConnectTimeout("t"))
    with patch.object(dt, "_handle_trafilatura_scrape", scrape), \
         patch.object(dt, "robust_fallback", AsyncMock()) as fb, \
         patch.object(dt, "handle_unstructured_loader", AsyncMock()) as unst:
        with pytest.raises(requests.exceptions.ConnectTimeout):
            await dt.get_content_by_url(URL, 0)
        fb.assert_not_awaited()
        unst.assert_not_awaited()
        assert scrape.await_count == 1
        with pytest.raises(UnreachableHostError, match="host www.dead.example unreachable"):
            await dt.get_content_by_url(URL2, 0)
        assert scrape.await_count == 1
        fb.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_connection_error_still_uses_fallback():
    scrape = AsyncMock(side_effect=_http_error(500))
    fallback_docs = [MagicMock()]
    with patch.object(dt, "_handle_trafilatura_scrape", scrape), \
         patch.object(dt, "handle_unstructured_loader", AsyncMock(side_effect=_http_error(500))), \
         patch.object(dt, "robust_fallback", AsyncMock(return_value=fallback_docs)) as fb:
        assert await dt.get_content_by_url(URL, 0) == fallback_docs
        fb.assert_awaited_once()


@pytest.mark.asyncio
async def test_breaker_expires_and_success_clears():
    ok = [dt.Document(page_content="x" * 200, metadata={"source": URL})]
    with patch.object(uh.time, "monotonic", return_value=1000.0):
        uh.record_unreachable(URL)
    scrape = AsyncMock(return_value=ok)
    with patch.object(dt, "_handle_trafilatura_scrape", scrape):
        with patch.object(uh.time, "monotonic", return_value=1000.0 + uh.UNREACHABLE_HOST_TTL_SECONDS - 1):
            with pytest.raises(UnreachableHostError):
                await dt.get_content_by_url(URL, 0)
        scrape.assert_not_awaited()
        with patch.object(uh.time, "monotonic", return_value=1000.0 + uh.UNREACHABLE_HOST_TTL_SECONDS + 1):
            assert await dt.get_content_by_url(URL, 0)
        scrape.assert_awaited_once()
    assert uh._down == {}


@pytest.mark.asyncio
async def test_success_clears_breaker():
    uh.record_unreachable(URL)
    uh._down["other.example"] = 0.0
    uh.clear_unreachable(URL2)
    assert "www.dead.example" not in uh._down and "other.example" in uh._down


@pytest.mark.asyncio
async def test_trafilatura_scrape_connect_error_raises_and_records():
    with patch.object(dt, "outbound_requests_get", side_effect=requests.exceptions.ConnectTimeout("t")) as get:
        with pytest.raises(requests.exceptions.ConnectTimeout):
            await dt._handle_trafilatura_scrape(URL, headers={"user-agent": "x"})
    assert get.call_args.kwargs["timeout"] == (10, 30)
    with pytest.raises(UnreachableHostError):
        uh.check_host_reachable(URL)


@pytest.mark.asyncio
async def test_trafilatura_scrape_http_error_returns_empty():
    resp = MagicMock()
    resp.raise_for_status.side_effect = _http_error(500)
    with patch.object(dt, "outbound_requests_get", return_value=resp):
        assert await dt._handle_trafilatura_scrape(URL, headers={"user-agent": "x"}) == []
    uh.check_host_reachable(URL)


def test_breaker_bounded():
    for i in range(1005):
        uh.record_unreachable(f"https://h{i}.example/")
    assert len(uh._down) == 1000
