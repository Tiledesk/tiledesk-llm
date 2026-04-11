"""
Semantic cache for recurring RAG queries.

Architecture:
  L1 — exact match (Redis GET/SET, key = sha256(namespace + normalized_question))
  L2 — semantic match (cosine similarity on stored embeddings, computed in Python)

Requires: redis>=7.1.0, numpy (already in deps via ML stack)

Redis key schema:
  tiledesk:cache:exact:{namespace}:{sha256}       → JSON payload
  tiledesk:cache:sem:{namespace}:{uuid}            → Hash {question, body, embedding}
  tiledesk:cache:sem_idx:{namespace}               → Set of semantic UUIDs for a namespace
"""

import hashlib
import json
import logging
import os
import struct
import time
import uuid
from typing import Optional

import numpy as np

from tilellm.shared.cache.metrics import (
    CACHE_REQUESTS,
    CACHE_LOOKUP_DURATION,
    CACHE_STORE_DURATION,
    CACHE_INVALIDATIONS,
)

logger = logging.getLogger(__name__)

# Defaults (can be overridden via env vars)
_DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_DEFAULT_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90"))
_DEFAULT_TTL_EXACT = int(os.getenv("CACHE_TTL_EXACT", "86400"))    # 24h
_DEFAULT_TTL_SEMANTIC = int(os.getenv("CACHE_TTL_SEMANTIC", "21600"))  # 6h

_PREFIX_EXACT = "tiledesk:cache:exact"
_PREFIX_SEM = "tiledesk:cache:sem"
_PREFIX_SEM_IDX = "tiledesk:cache:sem_idx"


def _normalize(text: str) -> str:
    return text.strip().lower()


def _exact_key(namespace: str, question: str) -> str:
    h = hashlib.sha256(_normalize(question).encode()).hexdigest()
    return f"{_PREFIX_EXACT}:{namespace}:{h}"


def _sem_key(namespace: str, entry_id: str) -> str:
    return f"{_PREFIX_SEM}:{namespace}:{entry_id}"


def _sem_idx_key(namespace: str) -> str:
    return f"{_PREFIX_SEM_IDX}:{namespace}"


def _vec_to_bytes(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _bytes_to_vec(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticCache:
    """
    Thread-safe, async semantic cache backed by Redis.
    Uses a single shared redis.asyncio.Redis client (lazy init).
    """

    _client = None  # shared async Redis client (lazy)

    @classmethod
    async def _get_client(cls):
        if cls._client is None:
            try:
                import redis.asyncio as aioredis
                cls._client = aioredis.from_url(
                    _DEFAULT_REDIS_URL,
                    decode_responses=False,  # we handle bytes ourselves
                    socket_connect_timeout=2,
                )
                await cls._client.ping()
                logger.info(f"SemanticCache: connected to Redis at {_DEFAULT_REDIS_URL}")
            except Exception as e:
                logger.warning(f"SemanticCache: Redis not available ({e}). Cache disabled.")
                cls._client = None
        return cls._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    async def lookup(
        cls,
        namespace: str,
        question: str,
        embedding: list[float],
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> Optional[dict]:
        """
        Returns cached payload dict {answer, body, ...} on hit, None on miss.
        Tries L1 (exact) first, then L2 (semantic).
        """
        r = await cls._get_client()
        if r is None:
            return None

        t0 = time.perf_counter()
        try:
            # L1 — exact match
            key = _exact_key(namespace, question)
            raw = await r.get(key)
            if raw:
                logger.info(f"SemanticCache L1 hit for namespace={namespace}")
                payload = json.loads(raw)
                payload["_cache_level"] = "exact"
                payload["_cache_similarity"] = 1.0
                CACHE_REQUESTS.labels(level="exact").inc()
                CACHE_LOOKUP_DURATION.observe(time.perf_counter() - t0)
                return payload

            # L2 — semantic match
            idx_key = _sem_idx_key(namespace)
            entry_ids = await r.smembers(idx_key)
            if not entry_ids:
                CACHE_REQUESTS.labels(level="miss").inc()
                CACHE_LOOKUP_DURATION.observe(time.perf_counter() - t0)
                return None

            query_vec = np.array(embedding, dtype=np.float32)
            best_score = -1.0
            best_payload = None

            for entry_id_bytes in entry_ids:
                entry_id = entry_id_bytes.decode()
                sem_key = _sem_key(namespace, entry_id)
                stored = await r.hgetall(sem_key)
                if not stored or b"embedding" not in stored:
                    continue
                stored_vec = _bytes_to_vec(stored[b"embedding"])
                score = _cosine(query_vec, stored_vec)
                if score > best_score:
                    best_score = score
                    best_payload = stored

            if best_score >= threshold and best_payload is not None:
                logger.info(
                    f"SemanticCache L2 hit for namespace={namespace} "
                    f"(cosine={best_score:.4f})"
                )
                body = json.loads(best_payload[b"body"].decode())
                body["_cache_level"] = "semantic"
                body["_cache_similarity"] = best_score
                CACHE_REQUESTS.labels(level="semantic").inc()
                CACHE_LOOKUP_DURATION.observe(time.perf_counter() - t0)
                return body

            CACHE_REQUESTS.labels(level="miss").inc()
            CACHE_LOOKUP_DURATION.observe(time.perf_counter() - t0)
            return None

        except Exception as e:
            logger.warning(f"SemanticCache.lookup error: {e}")
            return None

    @classmethod
    async def store(
        cls,
        namespace: str,
        question: str,
        embedding: list[float],
        body: dict,
        ttl_exact: int = _DEFAULT_TTL_EXACT,
        ttl_semantic: int = _DEFAULT_TTL_SEMANTIC,
    ) -> None:
        """
        Store a query result in both L1 (exact) and L2 (semantic).
        `body` is the full response dict to be returned on hit.
        """
        r = await cls._get_client()
        if r is None:
            return

        t0 = time.perf_counter()
        try:
            serialized = json.dumps(body)

            # L1 — exact
            key = _exact_key(namespace, question)
            await r.set(key, serialized.encode(), ex=ttl_exact)

            # L2 — semantic
            entry_id = str(uuid.uuid4())
            sem_key = _sem_key(namespace, entry_id)
            await r.hset(sem_key, mapping={
                "question": _normalize(question).encode(),
                "body": serialized.encode(),
                "embedding": _vec_to_bytes(embedding),
            })
            await r.expire(sem_key, ttl_semantic)

            # Register in namespace index (also with TTL)
            idx_key = _sem_idx_key(namespace)
            await r.sadd(idx_key, entry_id)
            await r.expire(idx_key, ttl_semantic)

            CACHE_STORE_DURATION.observe(time.perf_counter() - t0)
            logger.debug(f"SemanticCache stored entry for namespace={namespace}")

        except Exception as e:
            logger.warning(f"SemanticCache.store error: {e}")

    @classmethod
    async def invalidate_namespace(cls, namespace: str) -> int:
        """Delete all cache entries for a namespace. Returns count of deleted keys."""
        r = await cls._get_client()
        if r is None:
            return 0

        deleted = 0
        try:
            # Delete all semantic entries
            idx_key = _sem_idx_key(namespace)
            entry_ids = await r.smembers(idx_key)
            for entry_id_bytes in entry_ids:
                entry_id = entry_id_bytes.decode()
                await r.delete(_sem_key(namespace, entry_id))
                deleted += 1
            await r.delete(idx_key)

            # Delete exact-match entries (scan by pattern)
            pattern = f"{_PREFIX_EXACT}:{namespace}:*"
            async for key in r.scan_iter(pattern):
                await r.delete(key)
                deleted += 1

            CACHE_INVALIDATIONS.inc()
            logger.info(f"SemanticCache invalidated {deleted} entries for namespace={namespace}")
        except Exception as e:
            logger.warning(f"SemanticCache.invalidate_namespace error: {e}")

        return deleted

    @classmethod
    async def stats(cls, namespace: Optional[str] = None) -> dict:
        """Return basic stats (entry count per namespace or total)."""
        r = await cls._get_client()
        if r is None:
            return {"available": False}

        try:
            if namespace:
                idx_key = _sem_idx_key(namespace)
                count = await r.scard(idx_key)
                return {"available": True, "namespace": namespace, "semantic_entries": count}
            else:
                # Count all semantic index keys
                total = 0
                namespaces = []
                async for key in r.scan_iter(f"{_PREFIX_SEM_IDX}:*"):
                    ns = key.decode().replace(f"{_PREFIX_SEM_IDX}:", "", 1)
                    count = await r.scard(key)
                    namespaces.append({"namespace": ns, "entries": count})
                    total += count
                return {"available": True, "total_semantic_entries": total, "namespaces": namespaces}
        except Exception as e:
            return {"available": False, "error": str(e)}
