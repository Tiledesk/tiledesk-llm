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

UNREACHABLE_HOST_TTL_SECONDS = int(os.environ.get("UNREACHABLE_HOST_TTL_SECONDS", "600"))
_MAX_HOSTS = 1000

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
    return isinstance(exc, TimeoutError) and getattr(exc, "errno", None) == errno.ETIMEDOUT


def is_unreachable_host_error(exc: BaseException) -> bool:
    """True when `exc` (or anything in its cause/context chain) is a connection-level
    failure. HTTP errors, read timeouts, SSL errors and the like are False."""
    seen: set = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if _is_connect_level(e):
            return True
        stack.append(e.__cause__)
        stack.append(e.__context__)
        stack.append(getattr(e, "reason", None))  # urllib3 MaxRetryError
        stack.extend(a for a in getattr(e, "args", ()) if isinstance(a, BaseException))
    return False


_lock = threading.Lock()
_down: dict = {}  # host -> monotonic time of the failure


def _host(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def record_unreachable(url: str) -> None:
    host = _host(url)
    if not host:
        return
    with _lock:
        _down.pop(host, None)
        _down[host] = time.monotonic()
        while len(_down) > _MAX_HOSTS:
            _down.pop(next(iter(_down)))


def clear_unreachable(url: str) -> None:
    host = _host(url)
    if host and _down:
        with _lock:
            _down.pop(host, None)


def check_host_reachable(url: str) -> None:
    """Raise UnreachableHostError if the host of `url` failed recently."""
    host = _host(url)
    with _lock:
        since = _down.get(host)
        if since is None:
            return
        age = time.monotonic() - since
        if age >= UNREACHABLE_HOST_TTL_SECONDS:
            del _down[host]
            return
    raise UnreachableHostError(f"host {host} unreachable (connection failed {int(age)}s ago), skipped")


def reset() -> None:
    with _lock:
        _down.clear()
