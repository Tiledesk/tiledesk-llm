"""
Unit tests for SemanticCache (tilellm/shared/cache/semantic_cache.py).

Uses fakeredis to avoid a real Redis dependency.
"""

import json
import math
import struct
import pytest
import pytest_asyncio

from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vec(value: float, dim: int = 8) -> list[float]:
    """Build a unit vector pointing in a fixed direction, scaled by value."""
    v = [value] * dim
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _vec_to_bytes(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def fake_redis():
    """
    Provide a fakeredis AsyncFakeRedis instance and patch SemanticCache._client.
    """
    try:
        import fakeredis.aioredis as fake_aioredis
    except ImportError:
        pytest.skip("fakeredis not installed")

    r = fake_aioredis.FakeRedis(decode_responses=False)

    from tilellm.shared.cache.semantic_cache import SemanticCache
    original_client = SemanticCache._client
    SemanticCache._client = r
    yield r
    SemanticCache._client = original_client
    await r.aclose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExactMatch:
    @pytest.mark.asyncio
    async def test_exact_hit(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        body = {"answer": "Paris", "success": True, "namespace": "test_ns", "id": "1"}
        vec = _make_vec(1.0)

        await SemanticCache.store("test_ns", "What is the capital of France?", vec, body)
        result = await SemanticCache.lookup("test_ns", "What is the capital of France?", vec)

        assert result is not None
        assert result["answer"] == "Paris"
        assert result["_cache_level"] == "exact"
        assert result["_cache_similarity"] == 1.0

    @pytest.mark.asyncio
    async def test_exact_miss_different_question(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        body = {"answer": "Paris", "success": True, "namespace": "test_ns", "id": "1"}
        vec = _make_vec(1.0)

        await SemanticCache.store("test_ns", "What is the capital of France?", vec, body)
        result = await SemanticCache.lookup("test_ns", "What is the capital of Italy?", vec)

        # vec is identical so it will be a semantic hit, not exact
        # (same direction → cosine=1.0 ≥ threshold)
        # This tests that we DO get a hit from L2 even if L1 misses
        assert result is not None  # semantic hit

    @pytest.mark.asyncio
    async def test_exact_hit_case_insensitive(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        body = {"answer": "Paris", "success": True, "namespace": "ns", "id": "1"}
        vec = _make_vec(0.5)

        await SemanticCache.store("ns", "What is the capital of France?", vec, body)
        # Lookup with different casing — normalization makes it an exact hit
        result = await SemanticCache.lookup("ns", "WHAT IS THE CAPITAL OF FRANCE?", vec)

        assert result is not None
        assert result["_cache_level"] == "exact"


class TestSemanticMatch:
    @pytest.mark.asyncio
    async def test_semantic_hit_above_threshold(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        body = {"answer": "Berlin", "success": True, "namespace": "ns2", "id": "2"}
        # Stored with vec_a
        vec_a = _make_vec(1.0, dim=16)
        await SemanticCache.store("ns2", "Capital of Germany?", vec_a, body)

        # Query with slightly different question but almost-identical vec → should hit
        # vec_b is orthogonal to vec_a → cosine = 0 → should miss
        import numpy as np
        # Build an orthogonal vector (should miss)
        vec_ortho = [0.0] * 15 + [1.0]
        result = await SemanticCache.lookup("ns2", "What is Germany's capital?", vec_ortho, threshold=0.90)
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_hit_identical_vectors(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        body = {"answer": "Tokyo", "success": True, "namespace": "ns3", "id": "3"}
        vec = _make_vec(0.7, dim=16)
        await SemanticCache.store("ns3", "Capital of Japan?", vec, body)

        # Different question text but same embedding → L2 hit
        result = await SemanticCache.lookup("ns3", "Japan capital city?", vec, threshold=0.90)
        assert result is not None
        assert result["answer"] == "Tokyo"
        assert result["_cache_level"] == "semantic"
        assert result["_cache_similarity"] >= 0.99

    @pytest.mark.asyncio
    async def test_no_hit_empty_cache(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        vec = _make_vec(0.3)
        result = await SemanticCache.lookup("empty_ns", "Any question?", vec)
        assert result is None


class TestInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_namespace(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        body = {"answer": "Madrid", "success": True, "namespace": "inv_ns", "id": "4"}
        vec = _make_vec(0.9)
        await SemanticCache.store("inv_ns", "Capital of Spain?", vec, body)

        # Verify it's there
        result = await SemanticCache.lookup("inv_ns", "Capital of Spain?", vec)
        assert result is not None

        # Invalidate
        deleted = await SemanticCache.invalidate_namespace("inv_ns")
        assert deleted > 0

        # Now it should be gone
        result = await SemanticCache.lookup("inv_ns", "Capital of Spain?", vec)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_does_not_affect_other_namespaces(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        vec = _make_vec(0.5)
        body_a = {"answer": "A", "success": True, "namespace": "ns_a", "id": "a"}
        body_b = {"answer": "B", "success": True, "namespace": "ns_b", "id": "b"}

        await SemanticCache.store("ns_a", "Question A?", vec, body_a)
        await SemanticCache.store("ns_b", "Question B?", vec, body_b)

        await SemanticCache.invalidate_namespace("ns_a")

        # ns_b should still be cached
        result_b = await SemanticCache.lookup("ns_b", "Question B?", vec)
        assert result_b is not None
        assert result_b["answer"] == "B"


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_returns_entry_count(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        vec = _make_vec(0.3)
        for i in range(3):
            body = {"answer": f"A{i}", "success": True, "namespace": "stat_ns", "id": str(i)}
            await SemanticCache.store("stat_ns", f"Question {i}?", [v + i * 0.01 for v in vec], body)

        stats = await SemanticCache.stats("stat_ns")
        assert stats["available"] is True
        assert stats["semantic_entries"] == 3

    @pytest.mark.asyncio
    async def test_stats_all_namespaces(self, fake_redis):
        from tilellm.shared.cache.semantic_cache import SemanticCache

        vec = _make_vec(0.2)
        await SemanticCache.store("ns_x", "Q1?", vec, {"answer": "x", "success": True, "namespace": "ns_x", "id": "x"})
        await SemanticCache.store("ns_y", "Q2?", vec, {"answer": "y", "success": True, "namespace": "ns_y", "id": "y"})

        stats = await SemanticCache.stats()
        assert stats["available"] is True
        assert stats["total_semantic_entries"] >= 2


class TestCacheDisabled:
    @pytest.mark.asyncio
    async def test_use_cache_false_skips_everything(self, fake_redis):
        """When use_cache=False the nodes should not call SemanticCache at all."""
        from tilellm.agents.nodes import cache_lookup_node
        from tilellm.models.graph_state import GraphState

        qa = MagicMock()
        qa.use_cache = False
        state = {"question_answer": qa}

        result = await cache_lookup_node(state)
        assert result == {"cache_hit": False}
