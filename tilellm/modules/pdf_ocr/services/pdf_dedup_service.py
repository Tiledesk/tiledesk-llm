"""
Redis-based idempotency guard for PDF OCR tasks.

Prevents re-processing documents that are already queued, processing, or completed.
Uses namespace + doc_id as the composite dedup key.

States:
  queued     — task has been dispatched, worker not yet started
  processing — worker is actively processing the document
  completed  — document successfully indexed (long TTL, prevents re-submission)
  failed     — final processing failure (short TTL, re-submission allowed after cooldown)

Fail-open: if Redis is unavailable, all checks return None and processing continues.
"""

import os
import logging
from typing import Optional, Literal

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

DocState = Literal["queued", "processing", "completed", "failed"]

_KEY_PREFIX = "pdf_ocr:state:"

_COMPLETED_TTL = int(os.environ.get("PDF_OCR_DEDUP_COMPLETED_TTL", str(7 * 86400)))  # 7 days
_ACTIVE_TTL    = int(os.environ.get("PDF_OCR_DEDUP_ACTIVE_TTL",    "7200"))           # 2h safety net
_FAILED_TTL    = int(os.environ.get("PDF_OCR_DEDUP_FAILED_TTL",    "3600"))           # 1h cooldown

_client: Optional[aioredis.Redis] = None


def _get_client() -> Optional[aioredis.Redis]:
    global _client
    if _client is None:
        try:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            _client = aioredis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"pdf_dedup: Redis unavailable ({e}), dedup disabled")
    return _client


def _key(namespace: Optional[str], doc_id: str) -> str:
    ns = namespace or "_global_"
    return f"{_KEY_PREFIX}{ns}:{doc_id}"


async def get_doc_state(namespace: Optional[str], doc_id: str) -> Optional[DocState]:
    """Return current doc state or None if not tracked / Redis unavailable."""
    client = _get_client()
    if client is None:
        return None
    try:
        val = await client.get(_key(namespace, doc_id))
        return val  # type: ignore[return-value]
    except Exception as e:
        logger.warning(f"pdf_dedup: get_doc_state failed for {doc_id}: {e}")
        return None


async def set_queued(namespace: Optional[str], doc_id: str) -> bool:
    """
    Atomically claim the doc_id with state 'queued' (NX — only if not already set).

    Returns True if we claimed it (proceed with dispatch).
    Returns False if another process already claimed it (skip dispatch).
    Fail-open: returns True if Redis is unavailable so the task runs anyway.
    """
    client = _get_client()
    if client is None:
        return True
    try:
        ok = await client.set(_key(namespace, doc_id), "queued", nx=True, ex=_ACTIVE_TTL)
        return bool(ok)
    except Exception as e:
        logger.warning(f"pdf_dedup: set_queued failed for {doc_id}: {e}")
        return True  # fail-open


async def set_processing(namespace: Optional[str], doc_id: str) -> None:
    """Transition to 'processing' and refresh the active TTL."""
    client = _get_client()
    if client is None:
        return
    try:
        await client.set(_key(namespace, doc_id), "processing", ex=_ACTIVE_TTL)
    except Exception as e:
        logger.warning(f"pdf_dedup: set_processing failed for {doc_id}: {e}")


async def set_completed(namespace: Optional[str], doc_id: str) -> None:
    """Mark as completed with long TTL — prevents any re-submission until TTL expires."""
    client = _get_client()
    if client is None:
        return
    try:
        await client.set(_key(namespace, doc_id), "completed", ex=_COMPLETED_TTL)
        logger.info(f"pdf_dedup: doc_id={doc_id} ns={namespace} marked completed (TTL {_COMPLETED_TTL}s)")
    except Exception as e:
        logger.warning(f"pdf_dedup: set_completed failed for {doc_id}: {e}")


async def set_failed(namespace: Optional[str], doc_id: str) -> None:
    """Mark as failed with short TTL — re-submission allowed after cooldown."""
    client = _get_client()
    if client is None:
        return
    try:
        await client.set(_key(namespace, doc_id), "failed", ex=_FAILED_TTL)
        logger.info(f"pdf_dedup: doc_id={doc_id} ns={namespace} marked failed (TTL {_FAILED_TTL}s)")
    except Exception as e:
        logger.warning(f"pdf_dedup: set_failed failed for {doc_id}: {e}")


async def force_reset(namespace: Optional[str], doc_id: str) -> None:
    """Remove dedup state for one doc — allows immediate re-submission regardless of current state."""
    client = _get_client()
    if client is None:
        return
    try:
        await client.delete(_key(namespace, doc_id))
        logger.info(f"pdf_dedup: doc_id={doc_id} ns={namespace} state reset")
    except Exception as e:
        logger.warning(f"pdf_dedup: force_reset failed for {doc_id}: {e}")


async def clear_namespace(namespace: Optional[str]) -> int:
    """
    Remove ALL dedup keys for the given namespace.
    Call this whenever a namespace is deleted from the vector store, otherwise
    the 'completed' keys will block re-ingestion of the same documents.

    Returns the number of keys deleted.
    """
    client = _get_client()
    if client is None:
        return 0
    ns = namespace or "_global_"
    pattern = f"{_KEY_PREFIX}{ns}:*"
    deleted = 0
    try:
        async for key in client.scan_iter(pattern, count=100):
            await client.delete(key)
            deleted += 1
        logger.info(f"pdf_dedup: cleared {deleted} dedup keys for namespace={ns}")
    except Exception as e:
        logger.warning(f"pdf_dedup: clear_namespace({ns}) failed: {e}")
    return deleted
