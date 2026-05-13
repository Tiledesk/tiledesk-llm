"""
Analytics HTTP client for tiledesk-analytics ingest sidecar.

Fire-and-forget: publishes events via HTTP POST to the ingest sidecar.
Never blocks the caller; never raises exceptions.

Usage:
    # At app startup (lifespan hook):
    await analytics.init()

    # At app shutdown (lifespan hook):
    await analytics.shutdown()

    # To emit an event (fire-and-forget):
    analytics.publish_nowait(event_dict)

    # To emit and await (for use inside already-async contexts where
    # you can afford to fire one task and forget it):
    analytics.publish_nowait(event_dict)
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from .config import config

__all__ = ["init", "shutdown", "publish_nowait"]

logger = logging.getLogger(__name__)

# Module-level shared client and semaphore — created at init(), closed at shutdown().
_client: Optional[httpx.AsyncClient] = None
_semaphore: Optional[asyncio.Semaphore] = None


def _build_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["X-Api-Key"] = config.api_key
    return headers


async def init() -> None:
    """
    Create the shared AsyncClient and semaphore.
    Call once from the FastAPI lifespan startup hook.
    No-op when analytics is disabled.
    """
    global _client, _semaphore

    if not config.is_enabled:
        logger.debug("analytics: ANALYTICS_INGEST_URL not set — analytics disabled")
        return

    _semaphore = asyncio.Semaphore(config.max_concurrent)
    _client = httpx.AsyncClient(
        base_url=config.ingest_url,
        headers=_build_headers(),
        timeout=httpx.Timeout(
            connect=config.connect_timeout,
            read=config.read_timeout,
            write=config.read_timeout,
            pool=config.read_timeout,
        ),
        # Connection pool — reuse across requests
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    )
    logger.info("analytics: client initialised (url=%s)", config.ingest_url)


async def shutdown() -> None:
    """
    Close the shared AsyncClient.
    Call once from the FastAPI lifespan shutdown hook.
    """
    global _client, _semaphore

    if _client is not None:
        try:
            await _client.aclose()
            logger.info("analytics: client closed")
        except Exception as e:  # noqa: BLE001
            logger.debug("analytics: error closing client (ignored): %s", e)
        finally:
            _client = None
            _semaphore = None


async def _publish(event: Dict[str, Any]) -> None:
    """
    Internal coroutine: POST the event to the ingest sidecar.
    Acquires the semaphore to bound concurrency.
    All exceptions are caught and logged — never propagated.
    """
    if _client is None or _semaphore is None:
        return

    assert _semaphore is not None  # mypy
    async with _semaphore:
        logger.info(
            "analytics: posting event_type=%s project=%s",
            event.get("event_type"),
            event.get("id_project"),
        )
        try:
            resp = await _client.post("/events", json=event)
            if resp.status_code not in (200, 202):
                logger.warning(
                    "analytics: ingest rejected %s (status=%s): %s",
                    event.get("event_type"),
                    resp.status_code,
                    resp.text[:200],
                )
            else:
                logger.info(
                    "analytics: published %s for project %s",
                    event.get("event_type"),
                    event.get("id_project"),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "analytics: publish error for %s: %s",
                event.get("event_type"),
                exc,
            )


def _build_envelope(
    event_type: str,
    id_project: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct the full analytics event envelope."""
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "id_project": id_project,
        "source_service": config.source_service,
        "event_version": config.event_version,
        "payload": payload,
    }


def publish_nowait(
    event_type: str,
    id_project: Optional[str],
    payload: Dict[str, Any],
) -> None:
    """
    Fire-and-forget analytics event emission.

    - Returns immediately; schedules an asyncio.Task.
    - Silently skips when analytics is disabled or id_project is missing.
    - Never raises; never blocks the LLM response path.

    Args:
        event_type: One of 'ai.token_usage', 'ai.model_call', 'ai.tool_call',
                    'kb.query_executed', 'kb.content_indexed'.
        id_project: Tiledesk project ID. Skipped when None or empty.
        payload:    Event-specific payload dict.
    """
    if not config.is_enabled:
        logger.warning("analytics: disabled (ANALYTICS_INGEST_URL not set) — dropping %s", event_type)
        return

    if not id_project:
        logger.info("analytics: id_project is None/empty — dropping event %s", event_type)
        return

    if _client is None:
        logger.warning("analytics: client not initialized — dropping %s (was analytics.init() called?)", event_type)
        return

    event = _build_envelope(event_type, id_project, payload)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_publish(event))
        else:
            logger.debug("analytics: no running event loop, skipping %s", event_type)
    except RuntimeError:
        pass  # No event loop — skip silently
