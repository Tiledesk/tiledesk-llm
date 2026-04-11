"""
Prometheus metrics for the semantic cache.

Exposed via GET /metrics (standard Prometheus scrape endpoint).
Metrics:
  tiledesk_cache_requests_total{level="exact"|"semantic"|"miss"}  — counter
  tiledesk_cache_lookup_duration_seconds                          — histogram
  tiledesk_cache_store_duration_seconds                           — histogram
  tiledesk_cache_invalidations_total                              — counter
  tiledesk_embedding_cache_requests_total{result="hit"|"miss"}    — counter
"""

try:
    from prometheus_client import Counter, Histogram, CollectorRegistry, REGISTRY

    CACHE_REQUESTS = Counter(
        "tiledesk_cache_requests_total",
        "Total semantic cache lookup results",
        ["level"],          # "exact" | "semantic" | "miss"
    )

    CACHE_LOOKUP_DURATION = Histogram(
        "tiledesk_cache_lookup_duration_seconds",
        "Time spent on cache lookup (L1 + L2)",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )

    CACHE_STORE_DURATION = Histogram(
        "tiledesk_cache_store_duration_seconds",
        "Time spent storing a result in cache",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
    )

    CACHE_INVALIDATIONS = Counter(
        "tiledesk_cache_invalidations_total",
        "Total cache invalidation calls (namespace-level)",
    )

    EMBEDDING_CACHE_REQUESTS = Counter(
        "tiledesk_embedding_cache_requests_total",
        "L4 embedding cache hit/miss",
        ["result"],         # "hit" | "miss"
    )

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    _PROMETHEUS_AVAILABLE = False

    class _Noop:
        def labels(self, **_): return self
        def inc(self, *_, **__): pass
        def observe(self, *_, **__): pass
        def time(self): return _NoopCtx()

    class _NoopCtx:
        def __enter__(self): return self
        def __exit__(self, *_): pass

    CACHE_REQUESTS = _Noop()
    CACHE_LOOKUP_DURATION = _Noop()
    CACHE_STORE_DURATION = _Noop()
    CACHE_INVALIDATIONS = _Noop()
    EMBEDDING_CACHE_REQUESTS = _Noop()


def is_available() -> bool:
    return _PROMETHEUS_AVAILABLE
