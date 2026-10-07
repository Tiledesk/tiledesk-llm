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
    outbound_aiohttp_session,
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


@pytest.mark.asyncio
async def test_aiohttp_session_blocks_redirect_to_metadata(redirector):
    async with outbound_aiohttp_session() as session:
        with pytest.raises(BlockedURLError):
            async with session.get(redirector) as resp:
                await resp.read()


def test_requests_get_blocks_redirect_to_metadata(redirector):
    with pytest.raises(BlockedURLError):
        outbound_requests_get(redirector, timeout=5)


def test_requests_get_blocks_initial_url():
    with pytest.raises(BlockedURLError):
        outbound_requests_get("http://127.0.0.1:6379/", timeout=5)
