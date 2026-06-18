"""
Crash-proof attempt counter for PDF processing tasks.

SimpleRetryMiddleware counts retries only when a task raises an exception.
When the worker is OOM-killed (SIGKILL) no exception is raised, the stream
message is never XACKed, and the startup reclaim re-publishes it with the
original retry label — producing an infinite poison-pill loop.

This tracker increments a Redis counter BEFORE the dangerous work starts,
so the count survives worker death and is the single source of truth for
"how many times has this document been attempted", regardless of whether
the previous attempt ended with an exception or a SIGKILL.

Fail-open: if Redis is unavailable the counter returns 1 (first attempt).
This is safe because the TaskIQ broker itself requires Redis — if Redis is
down, no task runs at all.
"""

import logging
import os
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_KEY_PREFIX = "pdf_ocr:attempts:"

_ATTEMPTS_TTL = int(os.environ.get("PDF_OCR_ATTEMPTS_TTL", str(24 * 3600)))  # 24h
MAX_ATTEMPTS = int(os.environ.get("PDF_OCR_MAX_ATTEMPTS", "3"))

_client: Optional[aioredis.Redis] = None


def _get_client() -> Optional[aioredis.Redis]:
    global _client
    if _client is None:
        try:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            _client = aioredis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"attempt_tracker: Redis unavailable ({e}), tracking disabled")
    return _client


def _key(namespace: Optional[str], doc_id: str) -> str:
    ns = namespace or "_global_"
    return f"{_KEY_PREFIX}{ns}:{doc_id}"


async def incr_attempt(namespace: Optional[str], doc_id: str) -> int:
    """Increment and return the attempt number for this document (1-based).

    Must be called at task start, BEFORE any memory-heavy work, so the
    increment survives a worker OOM-kill.
    """
    client = _get_client()
    if client is None:
        return 1
    try:
        key = _key(namespace, doc_id)
        attempts = await client.incr(key)
        await client.expire(key, _ATTEMPTS_TTL)
        return int(attempts)
    except Exception as e:
        logger.warning(f"attempt_tracker: incr failed for {doc_id}: {e}")
        return 1


async def clear_attempts(namespace: Optional[str], doc_id: str) -> None:
    """Reset the counter after a successful run (or an explicit terminal failure)."""
    client = _get_client()
    if client is None:
        return
    try:
        await client.delete(_key(namespace, doc_id))
    except Exception as e:
        logger.warning(f"attempt_tracker: clear failed for {doc_id}: {e}")
