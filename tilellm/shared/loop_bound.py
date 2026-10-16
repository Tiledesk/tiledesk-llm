import asyncio
import threading
from typing import Callable, Dict, Generic, Optional, TypeVar

import httpx

T = TypeVar("T")


class LoopBound(Generic[T]):
    """One instance per event loop of an object whose connections are bound to the loop that opened
    them (httpx.AsyncClient, aiohttp sessions). A client cached in one loop and reused from another
    failed with "Event loop is closed" (found against a real laya-serve; any caller running
    asyncio.run more than once — scripts, workers, TestClient — hits it). Several live loops (a sync
    entry point running a coroutine on a temporary loop in another thread) each keep their own."""

    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._values: Dict[asyncio.AbstractEventLoop, T] = {}
        self._lock = threading.Lock()

    def get(self) -> T:
        loop = asyncio.get_running_loop()
        with self._lock:
            value = self._values.get(loop)
            if value is None:
                for closed in [lp for lp in self._values if lp.is_closed()]:
                    # ponytail: dropped, not closed — their loop is gone.
                    del self._values[closed]
                value = self._values[loop] = self._factory()
            return value

    def current(self) -> Optional[T]:
        """The instance of the running loop, if one was created (no creation)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        with self._lock:
            return self._values.get(loop)


class _NoCookies(httpx.Cookies):
    """Cookie jar that never stores anything."""

    def extract_cookies(self, response):
        return None


def cookieless_async_client(**kwargs) -> httpx.AsyncClient:
    """An httpx.AsyncClient meant to outlive a request: a Set-Cookie must not reach the next caller."""
    client = httpx.AsyncClient(**kwargs)
    # ponytail: private attribute — httpx copies any jar passed to the constructor into a plain Cookies.
    client._cookies = _NoCookies()
    return client
