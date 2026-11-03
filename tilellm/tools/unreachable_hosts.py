"""Fail fast on hosts that do not answer at the connection level.

A host that never completes the TCP connect would otherwise cost minutes per URL
(main method + three browser fallbacks), blocking the shared training queue.
`is_unreachable_host_error` classifies the failure; a small per-process circuit
breaker remembers the host for UNREACHABLE_HOST_TTL_SECONDS.
"""
import errno
import logging
import os
import socket
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import requests
import urllib3.exceptions as u3

logger = logging.getLogger(__name__)



def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, default))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


UNREACHABLE_HOST_TTL_SECONDS = _env_int("UNREACHABLE_HOST_TTL_SECONDS", 600)
UNREACHABLE_HOST_THRESHOLD = _env_int("UNREACHABLE_HOST_THRESHOLD", 2)
UNREACHABLE_HOST_WINDOW_SECONDS = _env_int("UNREACHABLE_HOST_WINDOW_SECONDS", 120)
UNREACHABLE_HOST_PROBE_SECONDS = _env_int("UNREACHABLE_HOST_PROBE_SECONDS", 60)
_MAX_HOSTS = 1000

# Chromium messages for a host that does not answer at the connection level. A plain
# `Page.goto: Timeout 60000ms exceeded` is NOT here: the host may just be slow.
_BROWSER_CONNECT_ERRORS = (
    "net::ERR_CONNECTION_TIMED_OUT",
    "net::ERR_CONNECTION_REFUSED",
    "net::ERR_NAME_NOT_RESOLVED",
    "net::ERR_ADDRESS_UNREACHABLE",
)

_CONNECT_TYPES: tuple = (
    requests.exceptions.ConnectTimeout,
    u3.ConnectTimeoutError,
    u3.NewConnectionError,
    socket.gaierror,
    ConnectionRefusedError,
)
_name_resolution = getattr(u3, "NameResolutionError", None)
if _name_resolution is not None:
    _CONNECT_TYPES += (_name_resolution,)


class UnreachableHostError(Exception):
    """Raised without any network call for a host recently found unreachable."""


def _is_connect_level(exc: BaseException) -> bool:
    if isinstance(exc, _CONNECT_TYPES):
        return True
    # [Errno 110] Connection timed out raised while connecting (a read timeout has no errno)
    if isinstance(exc, TimeoutError) and getattr(exc, "errno", None) == errno.ETIMEDOUT:
        return True
    return any(tok in str(exc) for tok in _BROWSER_CONNECT_ERRORS)


def is_unreachable_host_error(exc: BaseException) -> bool:
    """True when `exc` (or anything in its cause/context chain) is a connection-level
    failure. HTTP errors, read timeouts, SSL errors and the like are False."""
    return any(_is_connect_level(e) for e in _chain(exc))


def _chain(exc: BaseException):
    seen: set = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        stack.append(e.__cause__)
        stack.append(e.__context__)
        stack.append(getattr(e, "reason", None))
        stack.extend(a for a in getattr(e, "args", ()) if isinstance(a, BaseException))


def is_temporary_dns_error(exc: BaseException) -> bool:
    """EAI_AGAIN: a resolver hiccup, not evidence that the host is down."""
    return any(
        (isinstance(e, socket.gaierror) and e.errno == socket.EAI_AGAIN)
        or "Temporary failure in name resolution" in str(e)
        for e in _chain(exc)
    )


def _host(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def failed_host(exc: Optional[BaseException], url: str) -> str:
    """Host that actually failed (a redirect may have moved us off `url`'s host)."""
    for e in _chain(exc) if exc is not None else ():
        req_url = getattr(getattr(e, "request", None), "url", None)
        if isinstance(req_url, str) and _host(req_url):
            return _host(req_url)
        pool = getattr(e, "pool", None)
        host = getattr(pool, "host", None)
        if isinstance(host, str) and host:
            port = getattr(pool, "port", None)
            return f"{host}:{port}".lower() if port not in (None, 80, 443) else host.lower()
    return _host(url)


class _State:
    __slots__ = ("failures", "down_since", "last_attempt")

    def __init__(self):
        self.failures: list = []
        self.down_since: Optional[float] = None
        self.last_attempt: float = 0.0


_lock = threading.Lock()
_hosts: dict = {}  # host -> _State, oldest first


def record_unreachable(url: str, exc: Optional[BaseException] = None) -> None:
    """Count a connection-level failure; the host trips after THRESHOLD of them
    within WINDOW seconds. A failure on a tripped host re-arms it."""
    if exc is not None and is_temporary_dns_error(exc):
        return
    host = failed_host(exc, url)
    if not host:
        return
    now = time.monotonic()
    with _lock:
        st = _hosts.pop(host, None) or _State()
        _hosts[host] = st  # most recently touched last
        st.last_attempt = now
        if st.down_since is not None:
            st.down_since = now
        else:
            st.failures = [t for t in st.failures if now - t < UNREACHABLE_HOST_WINDOW_SECONDS] + [now]
            if len(st.failures) >= UNREACHABLE_HOST_THRESHOLD:
                st.down_since = now
                st.failures = []
        while len(_hosts) > _MAX_HOSTS:
            _hosts.pop(next(iter(_hosts)))


def clear_unreachable(url: str) -> None:
    host = _host(url)
    if host and _hosts:
        with _lock:
            _hosts.pop(host, None)


def check_host_reachable(url: str) -> None:
    """Raise UnreachableHostError if the host of `url` is tripped. One caller per
    PROBE seconds is let through (half-open) to find out whether it is back."""
    host = _host(url)
    with _lock:
        st = _hosts.get(host)
        if st is None or st.down_since is None:
            return
        now = time.monotonic()
        age = now - st.down_since
        if age >= UNREACHABLE_HOST_TTL_SECONDS:
            del _hosts[host]
            return
        if now - st.last_attempt >= UNREACHABLE_HOST_PROBE_SECONDS:
            st.last_attempt = now  # probe in flight: concurrent callers keep skipping
            return
    raise UnreachableHostError(f"host {host} unreachable (connection failed {int(age)}s ago), skipped")


def reset() -> None:
    with _lock:
        _hosts.clear()
