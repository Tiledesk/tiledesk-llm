"""Fail fast on unreachable hosts: classifier, per-host breaker, no browser fallbacks."""
import errno
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
        with pytest.raises(requests.exceptions.ConnectTimeout):
            await dt.get_content_by_url(URL2, 0)
        assert scrape.await_count == 2  # 1st failure does not trip the breaker
        with pytest.raises(UnreachableHostError, match="host www.dead.example unreachable"):
            await dt.get_content_by_url(URL2, 0)
        assert scrape.await_count == 2
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
async def test_trafilatura_scrape_connect_error_raises():
    with patch.object(dt, "outbound_requests_get", side_effect=requests.exceptions.ConnectTimeout("t")) as get:
        with pytest.raises(requests.exceptions.ConnectTimeout):
            await dt._handle_trafilatura_scrape(URL, headers={"user-agent": "x"})
    assert get.call_args.kwargs["timeout"] == (10, 30)
    assert uh._hosts == {}  # recorded once, by get_content_by_url, not here


@pytest.mark.asyncio
async def test_trafilatura_scrape_http_error_returns_empty():
    resp = MagicMock()
    resp.raise_for_status.side_effect = _http_error(500)
    with patch.object(dt, "outbound_requests_get", return_value=resp):
        assert await dt._handle_trafilatura_scrape(URL, headers={"user-agent": "x"}) == []
    uh.check_host_reachable(URL)



# ---- additional coverage -------------------------------------------------

class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    c = _Clock()
    with patch.object(uh.time, "monotonic", c):
        yield c


def _trip(url=URL):
    uh.record_unreachable(url)
    uh.record_unreachable(url)


def test_classifier_more_cases():
    assert is_unreachable_host_error(u3.NameResolutionError("h", None, socket.gaierror(-2, "x")))
    mre = u3.MaxRetryError(None, "/a", reason=u3.NewConnectionError(None, "boom"))
    assert is_unreachable_host_error(mre)
    assert is_unreachable_host_error(TimeoutError(errno.ETIMEDOUT, "Connection timed out"))
    assert not is_unreachable_host_error(TimeoutError("read timed out"))
    for m in ("net::ERR_CONNECTION_TIMED_OUT", "net::ERR_CONNECTION_REFUSED",
              "net::ERR_NAME_NOT_RESOLVED", "net::ERR_ADDRESS_UNREACHABLE"):
        assert is_unreachable_host_error(ValueError(f"Navigation failed. Error: Page.goto: {m} at {URL}"))
    assert not is_unreachable_host_error(Exception("Page.goto: Timeout 60000ms exceeded."))


def test_threshold(clock):
    uh.record_unreachable(URL)
    uh.check_host_reachable(URL)  # 1 failure: not tripped
    clock.t += 121  # outside the window
    uh.record_unreachable(URL)
    uh.check_host_reachable(URL)
    clock.t += 10  # within the window of the previous one
    uh.record_unreachable(URL)
    with pytest.raises(UnreachableHostError):
        uh.check_host_reachable(URL)


def test_eai_again_not_counted(clock):
    exc = socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    assert is_unreachable_host_error(exc)  # still fails fast for the URL
    uh.record_unreachable(URL, exc)
    uh.record_unreachable(URL, exc)
    uh.check_host_reachable(URL)
    assert uh._hosts == {}


def test_probe_half_open(clock):
    _trip()
    clock.t += 30
    with pytest.raises(UnreachableHostError):
        uh.check_host_reachable(URL)
    clock.t += 31  # 61 s since last attempt
    uh.check_host_reachable(URL)  # the probe passes
    with pytest.raises(UnreachableHostError):  # concurrent callers skip
        uh.check_host_reachable(URL)
    uh.record_unreachable(URL)  # probe failed: re-armed
    clock.t += 30
    with pytest.raises(UnreachableHostError):
        uh.check_host_reachable(URL)
    clock.t += 31
    uh.check_host_reachable(URL)
    uh.clear_unreachable(URL)  # probe succeeded
    uh.check_host_reachable(URL)
    uh.check_host_reachable(URL)


def test_ttl_expiry(clock):
    _trip()
    for _ in range(9):
        clock.t += 61
        uh.check_host_reachable(URL)  # probes, then
        uh.record_unreachable(URL)  # re-armed (down_since moves) -> never expires while failing
    uh.reset()
    _trip()
    clock.t += uh.UNREACHABLE_HOST_TTL_SECONDS + 1
    uh.check_host_reachable(URL)
    assert uh._hosts == {}


def test_failed_host_from_request():
    e = requests.exceptions.ConnectTimeout(request=requests.Request("GET", "https://Other.example:8080/x"))
    assert uh.failed_host(e, URL) == "other.example:8080"
    e2 = requests.exceptions.ConnectTimeout(request=requests.Request("GET", "https://WWW.Dead.example:443/x"))
    assert uh.failed_host(e2, URL) == uh._host(URL) == "www.dead.example"
    pool = MagicMock(host="Www.Dead.Example", port=443, scheme="https")
    assert uh.failed_host(u3.MaxRetryError(pool, "/a"), "https://x/") == "www.dead.example"
    assert uh.failed_host(ValueError(), URL) == "www.dead.example"


def test_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("X_BAD", "abc")
    assert uh._env_int("X_BAD", 7) == 7
    monkeypatch.setenv("X_BAD", "-3")
    assert uh._env_int("X_BAD", 7) == 7


def test_breaker_bounded_evicts_oldest():
    for i in range(1005):
        uh.record_unreachable(f"https://h{i}.example/")
    assert len(uh._hosts) == 1000
    assert "h0.example" not in uh._hosts and "h1004.example" in uh._hosts


@pytest.mark.asyncio
async def test_scrape_type_1_connect_error_fast_fails():
    with patch.object(dt, "_handle_trafilatura_scrape", AsyncMock(side_effect=requests.exceptions.ConnectTimeout("t"))), \
         patch.object(dt, "handle_unstructured_loader", AsyncMock()) as unst, \
         patch.object(dt, "robust_fallback", AsyncMock()) as fb:
        with pytest.raises(requests.exceptions.ConnectTimeout):
            await dt.get_content_by_url(URL, 1)
        unst.assert_not_awaited()
        fb.assert_not_awaited()


@pytest.mark.asyncio
async def test_unstructured_connect_error_after_empty_trafilatura():
    with patch.object(dt, "_handle_trafilatura_scrape", AsyncMock(return_value=[])), \
         patch.object(dt, "handle_unstructured_loader", AsyncMock(side_effect=_conn_error())), \
         patch.object(dt, "robust_fallback", AsyncMock()) as fb:
        with pytest.raises(requests.exceptions.ConnectionError):
            await dt.get_content_by_url(URL, 0)
        fb.assert_not_awaited()


@pytest.mark.asyncio
async def test_playwright_connection_timed_out_fast_fails():
    err = ValueError(f"Navigation failed for URL: {URL}. Error: Page.goto: net::ERR_CONNECTION_TIMED_OUT at {URL}")
    params = MagicMock()
    with patch.object(dt, "handle_playwright_scrape", AsyncMock(side_effect=err)), \
         patch.object(dt, "robust_fallback", AsyncMock()) as fb:
        with pytest.raises(ValueError):
            await dt.get_content_by_url(URL, 2, parameters_scrape_type_4=params)
        fb.assert_not_awaited()


@pytest.mark.asyncio
async def test_playwright_goto_timeout_still_falls_back():
    params = MagicMock()
    docs = [MagicMock()]
    with patch.object(dt, "handle_playwright_scrape", AsyncMock(side_effect=Exception("Page.goto: Timeout 60000ms exceeded."))), \
         patch.object(dt, "robust_fallback", AsyncMock(return_value=docs)) as fb:
        assert await dt.get_content_by_url(URL, 2, parameters_scrape_type_4=params) == docs
        fb.assert_awaited_once()


@pytest.mark.asyncio
async def test_captcha_rule_unchanged():
    with patch.object(dt, "_handle_trafilatura_scrape", AsyncMock(side_effect=ValueError("CAPTCHA detected"))), \
         patch.object(dt, "handle_unstructured_loader", AsyncMock(side_effect=ValueError("CAPTCHA detected"))), \
         patch.object(dt, "robust_fallback", AsyncMock()) as fb:
        with pytest.raises(ValueError, match="CAPTCHA"):
            await dt.get_content_by_url(URL, 0)
        fb.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_clears_failure_count():
    ok = [dt.Document(page_content="x" * 200, metadata={"source": URL})]
    uh.record_unreachable(URL)
    with patch.object(dt, "_handle_trafilatura_scrape", AsyncMock(return_value=ok)):
        assert await dt.get_content_by_url(URL, 0)
    assert uh._hosts == {}


@pytest.mark.asyncio
async def test_probe_success_clears_tripped_host(clock):
    ok = [dt.Document(page_content="x" * 200, metadata={"source": URL})]
    _trip()
    clock.t += 61
    with patch.object(dt, "_handle_trafilatura_scrape", AsyncMock(return_value=ok)):
        assert await dt.get_content_by_url(URL, 0)
    assert uh._hosts == {}


@pytest.mark.asyncio
async def test_end_to_end_single_count_trafilatura_path():
    with patch.object(dt, "outbound_requests_get", side_effect=requests.exceptions.ConnectTimeout("t")) as get, \
         patch.object(dt, "robust_fallback", AsyncMock()) as fb:
        for u in (URL, "https://www.dead.example/b"):
            with pytest.raises(requests.exceptions.ConnectTimeout):
                await dt.get_content_by_url(u, 0)
        assert get.call_count == 2
        with pytest.raises(UnreachableHostError):
            await dt.get_content_by_url("https://www.dead.example/c", 0)
        assert get.call_count == 2
        fb.assert_not_awaited()


@pytest.mark.asyncio
async def test_end_to_end_single_count_playwright_path():
    err = ValueError(f"Navigation failed. Error: Page.goto: net::ERR_CONNECTION_TIMED_OUT at {URL}")
    params = MagicMock()
    with patch.object(dt, "scrape_page", AsyncMock(side_effect=err)) as sp, \
         patch.object(dt, "robust_fallback", AsyncMock()) as fb:
        for u in (URL, "https://www.dead.example/b"):
            with pytest.raises(ValueError):
                await dt.get_content_by_url(u, 2, parameters_scrape_type_4=params)
        assert sp.await_count == 2
        with pytest.raises(UnreachableHostError):
            await dt.get_content_by_url("https://www.dead.example/c", 2, parameters_scrape_type_4=params)
        fb.assert_not_awaited()
