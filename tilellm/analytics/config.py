"""
Analytics configuration — reads from environment variables.

When ANALYTICS_INGEST_URL is not set, analytics is silently disabled
with zero overhead (all publish calls short-circuit immediately).
"""
import os
from dataclasses import dataclass
from typing import Optional

__all__ = ["config"]


@dataclass(frozen=True)
class AnalyticsConfig:
    """Immutable analytics configuration read once at import time."""

    ingest_url: Optional[str]
    api_key: Optional[str]
    source_service: str = "tiledesk-llm"
    event_version: str = "1.0"

    # Connection-pool & timeout settings
    connect_timeout: float = 2.0
    read_timeout: float = 5.0
    max_concurrent: int = 50  # semaphore bound for fire-and-forget tasks

    @property
    def is_enabled(self) -> bool:
        """True only when the ingest URL is configured and non-empty."""
        return bool(self.ingest_url)


def _load_config() -> AnalyticsConfig:
    return AnalyticsConfig(
        ingest_url=os.environ.get("ANALYTICS_INGEST_URL") or None,
        api_key=os.environ.get("ANALYTICS_INGEST_API_KEY") or None,
    )


# Module-level singleton — safe because the dataclass is frozen.
config: AnalyticsConfig = _load_config()
