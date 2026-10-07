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

import aiohttp
import httpx
import requests

_METADATA_IPS = {
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("fd00:ec2::254"),
    ipaddress.ip_address("100.100.100.200"),
}


class BlockedURLError(ValueError):
    """The URL points somewhere the service must not fetch from."""


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


def outbound_client(**kwargs: Any) -> httpx.AsyncClient:
    """httpx.AsyncClient that validates the URL of every request, redirects included."""
    return httpx.AsyncClient(event_hooks={"request": [_httpx_request_hook]}, **kwargs)


async def _aiohttp_on_request_start(session, ctx, params) -> None:
    await avalidate_outbound_url(str(params.url))


async def _aiohttp_on_request_redirect(session, ctx, params) -> None:
    # on_request_start fires once per request, not per hop: check where the redirect goes
    location = params.response.headers.get("Location", "")
    await avalidate_outbound_url(urljoin(str(params.url), location))


def outbound_aiohttp_session(**kwargs: Any) -> aiohttp.ClientSession:
    """aiohttp.ClientSession that validates the URL of every request, redirects included."""
    trace = aiohttp.TraceConfig()
    trace.on_request_start.append(_aiohttp_on_request_start)
    trace.on_request_redirect.append(_aiohttp_on_request_redirect)
    return aiohttp.ClientSession(trace_configs=[trace], **kwargs)


def _requests_redirect_hook(response: requests.Response, *args: Any, **kwargs: Any) -> None:
    if response.is_redirect:
        validate_outbound_url(urljoin(response.url, response.headers.get("location", "")))


def outbound_requests_get(url: str, **kwargs: Any) -> requests.Response:
    """requests.get that validates the URL and every redirect target."""
    validate_outbound_url(url)
    with requests.Session() as session:
        session.hooks["response"].append(_requests_redirect_hook)
        return session.get(url, **kwargs)
