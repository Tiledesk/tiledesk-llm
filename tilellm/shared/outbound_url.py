"""
Guard for URLs that come from the request body (file_content, source, *_url,
webhook) and are fetched by the service (SSRF).

Always blocked: non-http(s) schemes, loopback, link-local (cloud metadata
169.254.169.254), multicast/unspecified/reserved, known metadata addresses.
Private networks are allowed by default (MinIO, internal services); set
OUTBOUND_ALLOW_PRIVATE_NETWORKS=false to block them too.
OUTBOUND_URL_ALLOWED_HOSTS (comma separated) bypasses every check for those hosts.

Every redirect hop is re-validated: a public URL answering 302 → metadata is
the classic bypass. Use the helpers below instead of building clients inline.

Response bodies are capped while they stream in (OUTBOUND_MAX_RESPONSE_BYTES,
default 200 MB): measuring len(response.content) afterwards is too late, the
whole body is already in memory. Browser pages: guard_page() applies the URL
check to every request the page makes (iframes, scripts, fetch).

ponytail: the check resolves DNS before the library does (time-of-check /
time-of-use): DNS rebinding can still slip through. Block egress to
169.254.0.0/16 at the network level (k8s NetworkPolicy) for full coverage.
"""
import asyncio
import ipaddress
import os
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import requests

_METADATA_IPS = {
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("fd00:ec2::254"),
    ipaddress.ip_address("100.100.100.200"),
}


class BlockedURLError(ValueError):
    """The URL points somewhere the service must not fetch from."""


class ResponseTooLargeError(ValueError):
    """The response body exceeds OUTBOUND_MAX_RESPONSE_BYTES."""


def _max_response_bytes() -> int:
    return int(os.environ.get("OUTBOUND_MAX_RESPONSE_BYTES", 200 * 1024 * 1024))


def _too_large(url: str, limit: int) -> ResponseTooLargeError:
    return ResponseTooLargeError(f"response larger than {limit} bytes: {str(url)[:120]}")


def _allowed_hosts() -> set:
    raw = os.environ.get("OUTBOUND_URL_ALLOWED_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _allow_private() -> bool:
    return os.environ.get("OUTBOUND_ALLOW_PRIVATE_NETWORKS", "true").strip().lower() not in ("false", "0", "no")


def _check_ip(ip: ipaddress._BaseAddress, url: str) -> None:
    ip = getattr(ip, "ipv4_mapped", None) or ip
    if (ip in _METADATA_IPS or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_unspecified or ip.is_reserved):
        raise BlockedURLError(f"URL not allowed (address {ip}): {url[:120]}")
    if ip.is_private and not _allow_private():
        raise BlockedURLError(f"URL not allowed (private address {ip}): {url[:120]}")


def validate_outbound_url(url: str) -> str:
    parsed = urlparse(url or "")
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise BlockedURLError(f"URL must be http(s) with a host: {str(url)[:120]!r}")
    host = parsed.hostname.lower()
    if host in _allowed_hosts():
        return url
    if host == "localhost" or host.endswith(".localhost"):
        raise BlockedURLError(f"URL not allowed (loopback): {url[:120]}")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            return url  # unresolvable: the fetch itself will fail
        addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
    for ip in addresses:
        _check_ip(ip, url)
    return url


async def avalidate_outbound_url(url: str) -> str:
    return await asyncio.to_thread(validate_outbound_url, url)


async def _httpx_request_hook(request: httpx.Request) -> None:
    await avalidate_outbound_url(str(request.url))


class _LimitedStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, limit: int, url: str):
        self._stream, self._limit, self._url = stream, limit, url

    async def __aiter__(self):
        total = 0
        async for chunk in self._stream:
            total += len(chunk)
            if total > self._limit:
                raise _too_large(self._url, self._limit)
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _LimitedTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await super().handle_async_request(request)
        limit = _max_response_bytes()
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            await response.aclose()
            raise _too_large(str(request.url), limit)
        response.stream = _LimitedStream(response.stream, limit, str(request.url))
        return response


def outbound_client(**kwargs: Any) -> httpx.AsyncClient:
    """httpx.AsyncClient that validates the URL of every request (redirects
    included) and caps the response size.

    ponytail: the cap counts bytes on the wire; Accept-Encoding: identity keeps
    them equal to the decoded size unless a server compresses anyway.
    """
    headers = {"Accept-Encoding": "identity", **(kwargs.pop("headers", None) or {})}
    return httpx.AsyncClient(
        event_hooks={"request": [_httpx_request_hook]},
        transport=_LimitedTransport(),
        headers=headers,
        **kwargs,
    )


def _requests_redirect_hook(response: requests.Response, *args: Any, **kwargs: Any) -> None:
    if response.is_redirect:
        validate_outbound_url(urljoin(response.url, response.headers.get("location", "")))


def outbound_requests_get(url: str, **kwargs: Any) -> requests.Response:
    """requests.get that validates the URL and every redirect target, and caps
    the (decoded) response size."""
    validate_outbound_url(url)
    limit = _max_response_bytes()
    with requests.Session() as session:
        session.hooks["response"].append(_requests_redirect_hook)
        response = session.get(url, stream=True, **kwargs)
        try:
            declared = response.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > limit:
                raise _too_large(url, limit)
            body = bytearray()
            for chunk in response.iter_content(64 * 1024):
                body += chunk
                if len(body) > limit:
                    raise _too_large(url, limit)
        except Exception:
            response.close()
            raise
        response._content = bytes(body)
        return response


async def _route_guard(route: Any) -> None:
    try:
        await avalidate_outbound_url(route.request.url)
    except BlockedURLError:
        await route.abort("blockedbyclient")
        return
    await route.continue_()


async def guard_page(page: Any) -> Any:
    """Apply the URL check to every request a Playwright page makes.

    ponytail: one DNS lookup per request; cache per host if scraping gets slow.
    """
    await page.route("**/*", _route_guard)
    return page
