"""
tilellm.analytics — fire-and-forget analytics integration.

Public API:
    init()            — call at app startup (lifespan hook)
    shutdown()        — call at app shutdown (lifespan hook)
    publish_nowait()  — emit an event without blocking

Event builder functions are in tilellm.analytics.events.
"""
from .client import init, shutdown, publish_nowait
from . import events

__all__ = ["init", "shutdown", "publish_nowait", "events"]
