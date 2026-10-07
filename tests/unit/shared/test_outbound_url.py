"""
URLs from the request body (file_content, source, *_url, webhook) are fetched
by the service: they must not reach the cloud metadata endpoint, link-local or
loopback addresses — neither directly nor through a redirect from a public URL.
Private networks stay allowed by default (MinIO, internal services) and can be
locked down with OUTBOUND_ALLOW_PRIVATE_NETWORKS=false + OUTBOUND_URL_ALLOWED_HOSTS.
"""
import functools
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from tilellm.shared.outbound_url import (
    BlockedURLError,
    ResponseTooLargeError,
    guard_page,
    outbound_client,
    outbound_requests_get,
    validate_outbound_url,
)


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://[fd00:ec2::254]/",
    "http://169.254.1.1/",
    "http://[fe80::1]/",
    "http://[::ffff:169.254.169.254]/",
    "http://100.100.100.200/",
    "http://127.0.0.1:6379/",
    "http://localhost:8000/api/ask",
    "http://[::1]/",
    "http://0.0.0.0/",
    "file:///etc/hosts",
    "ftp://example.com/x",
    "gopher://127.0.0.1:6379/_FLUSHALL",
    "",
    "http:///nohost",
])
def test_blocked(url):
    with pytest.raises(BlockedURLError):
        validate_outbound_url(url)


@pytest.mark.parametrize("url", ["http://10.0.0.5:9000/bucket/a.pdf", "https://minio:9000/x"])
def test_private_networks_allowed_by_default(url, monkeypatch):
    monkeypatch.delenv("OUTBOUND_ALLOW_PRIVATE_NETWORKS", raising=False)
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.0.0.5", 0))])
    assert validate_outbound_url(url) == url


def test_private_networks_can_be_blocked_and_allowlisted(monkeypatch):
    monkeypatch.setenv("OUTBOUND_ALLOW_PRIVATE_NETWORKS", "false")
    with pytest.raises(BlockedURLError):
        validate_outbound_url("http://10.0.0.5/a.pdf")
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "10.0.0.5, files.internal")
    assert validate_outbound_url("http://10.0.0.5/a.pdf")


def test_hostname_resolving_to_metadata_is_blocked(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("169.254.169.254", 0))])
    with pytest.raises(BlockedURLError):
        validate_outbound_url("http://metadata.google.internal/computeMetadata/v1/")


# ---------------------------------------------------------------------------
# Redirects: a real local server (allowlisted) redirecting to the metadata IP.
# ---------------------------------------------------------------------------

class _Redirect(BaseHTTPRequestHandler):
    def _go(self):
        self.send_response(302)
        self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_POST = _go

    def log_message(self, *a):
        pass


@pytest.fixture
def redirector(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Redirect)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "127.0.0.1")
    yield f"http://127.0.0.1:{server.server_port}/file.pdf"
    server.shutdown()


@pytest.mark.asyncio
async def test_httpx_client_blocks_redirect_to_metadata(redirector):
    async with outbound_client(timeout=5) as client:
        with pytest.raises(BlockedURLError):
            await client.get(redirector, follow_redirects=True)


@pytest.mark.asyncio
async def test_httpx_client_blocks_initial_url():
    async with outbound_client(timeout=5) as client:
        with pytest.raises(BlockedURLError):
            await client.get("http://169.254.169.254/")


def test_requests_get_blocks_redirect_to_metadata(redirector):
    with pytest.raises(BlockedURLError):
        outbound_requests_get(redirector, timeout=5)


def test_requests_get_blocks_initial_url():
    with pytest.raises(BlockedURLError):
        outbound_requests_get("http://127.0.0.1:6379/", timeout=5)


# ---------------------------------------------------------------------------
# Response size: the body must be cut off while it streams in, not measured
# after it is already in memory (a huge or endless response OOMs the pod).
# ---------------------------------------------------------------------------

class _Big(BaseHTTPRequestHandler):
    size = 3000

    def do_GET(self):
        self.send_response(200)
        if "nolength" in self.path:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunk = b"x" * 500
            for _ in range(self.size // 500):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.send_header("Content-Length", str(self.size))
            self.end_headers()
            self.wfile.write(b"x" * self.size)

    def log_message(self, *a):
        pass


@pytest.fixture
def big(monkeypatch):
    _Big.protocol_version = "HTTP/1.1"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Big)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "127.0.0.1")
    monkeypatch.setenv("OUTBOUND_MAX_RESPONSE_BYTES", "1000")
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/declared", "/nolength"])
async def test_httpx_client_stops_oversized_responses(big, path):
    async with outbound_client(timeout=5) as client:
        with pytest.raises(ResponseTooLargeError):
            await client.get(big + path)


@pytest.mark.parametrize("path", ["/declared", "/nolength"])
def test_requests_get_stops_oversized_responses(big, path):
    with pytest.raises(ResponseTooLargeError):
        outbound_requests_get(big + path, timeout=5)


@pytest.mark.asyncio
async def test_responses_under_the_limit_are_untouched(big, monkeypatch):
    monkeypatch.setenv("OUTBOUND_MAX_RESPONSE_BYTES", "5000")
    async with outbound_client(timeout=5) as client:
        assert len((await client.get(big + "/nolength")).content) == 3000
    assert len(outbound_requests_get(big + "/declared", timeout=5).content) == 3000


# ---------------------------------------------------------------------------
# Browser scraping: the page loads iframes/scripts/fetches by itself, so every
# request it makes goes through the same check, not just the first URL.
# ---------------------------------------------------------------------------

class _FakeRoute:
    def __init__(self, url):
        self.request = type("R", (), {"url": url})()
        self.aborted = self.continued = False

    async def abort(self, *a, **k):
        self.aborted = True

    async def continue_(self, *a, **k):
        self.continued = True


class _FakePage:
    async def route(self, pattern, handler):
        self.pattern, self.handler = pattern, handler


@pytest.mark.asyncio
async def test_guard_page_aborts_requests_to_blocked_addresses():
    page = _FakePage()
    await guard_page(page)

    blocked, allowed = _FakeRoute("http://169.254.169.254/latest/"), _FakeRoute("http://93.184.216.34/x.js")
    await page.handler(blocked)
    await page.handler(allowed)

    assert page.pattern == "**/*"
    assert blocked.aborted and not blocked.continued
    assert allowed.continued and not allowed.aborted


def test_every_scraper_page_is_guarded():
    import inspect
    import re

    from tilellm.tools import document_tools

    src = inspect.getsource(document_tools)
    pages = re.findall(r"(\w+) = await (?:browser|context)\.new_page\(", src)
    guarded = set(re.findall(r"await guard_page\((\w+)\)", src))
    assert pages and len(re.findall(r"await guard_page\(", src)) >= len(pages)
    assert set(pages) <= guarded
