"""
L4 Embedding cache — caches individual text→vector results in Redis.

Key: tiledesk:cache:emb:{model_key}:{sha256(text)}
Value: raw bytes (struct-packed float32 array)
TTL: CACHE_TTL_EMBEDDING (default 3600s = 1h)

Usage:
    from tilellm.shared.cache.embedding_cache import CachedEmbeddings

    base_model = OpenAIEmbeddings(...)
    cached = CachedEmbeddings(base_model, model_key="openai:text-embedding-3-small")
    vec = await cached.aembed_query("What is RAG?")
"""

import hashlib
import logging
import os
import struct
from typing import List

from tilellm.shared.cache.metrics import EMBEDDING_CACHE_REQUESTS

logger = logging.getLogger(__name__)

_TTL = int(os.getenv("CACHE_TTL_EMBEDDING", "3600"))  # 1h
_PREFIX = "tiledesk:cache:emb"


def _emb_key(model_key: str, text: str) -> str:
    h = hashlib.sha256(text.strip().encode()).hexdigest()
    return f"{_PREFIX}:{model_key}:{h}"


def _vec_to_bytes(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _bytes_to_vec(b: bytes) -> List[float]:
    n = len(b) // 4
    return list(struct.unpack(f"{n}f", b))


class CachedEmbeddings:
    """
    Wraps any LangChain Embeddings object and caches embed_query results in Redis.
    Falls back transparently to the base model on Redis errors.
    """

    def __init__(self, base, model_key: str):
        self._base = base
        self._model_key = model_key

    async def _get_redis(self):
        from tilellm.shared.cache.semantic_cache import SemanticCache
        return await SemanticCache._get_client()

    async def aembed_query(self, text: str) -> List[float]:
        r = await self._get_redis()
        if r is not None:
            try:
                key = _emb_key(self._model_key, text)
                raw = await r.get(key)
                if raw:
                    logger.debug(f"EmbeddingCache hit for model={self._model_key}")
                    EMBEDDING_CACHE_REQUESTS.labels(result="hit").inc()
                    return _bytes_to_vec(raw)
            except Exception as e:
                logger.warning(f"EmbeddingCache read error: {e}")

        EMBEDDING_CACHE_REQUESTS.labels(result="miss").inc()
        vec = await self._base.aembed_query(text)

        if r is not None:
            try:
                key = _emb_key(self._model_key, text)
                await r.set(key, _vec_to_bytes(vec), ex=_TTL)
            except Exception as e:
                logger.warning(f"EmbeddingCache write error: {e}")

        return vec

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed — caches each text individually."""
        results = []
        for text in texts:
            results.append(await self.aembed_query(text))
        return results

    def embed_query(self, text: str) -> List[float]:
        """Sync fallback — no caching, delegates to base."""
        return self._base.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._base.embed_documents(texts)
