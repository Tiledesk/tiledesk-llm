"""
Reusable token-usage tracking for LangChain LLM calls.

Provider-agnostic: reads ``usage_metadata`` from the returned ``AIMessage``
(populated by ChatOpenAI/vLLM, ChatAnthropic, ChatGoogle*, etc.), with a
fallback to ``response_metadata['token_usage']`` for clients that only expose
the raw provider usage. Structured-output calls must be invoked with
``with_structured_output(Schema, include_raw=True)`` so the raw ``AIMessage``
(and therefore the usage) survives — :func:`extract_token_usage` unwraps the
``{"raw", "parsed", "parsing_error"}`` dict transparently.

Design (SOLID):
  - :class:`TokenUsageRecord` — one immutable per-call record (SRP).
  - :func:`extract_token_usage` — pure extraction from a LangChain response (SRP).
  - :class:`TokenUsageCollector` — accumulates records, aggregates, serializes (SRP).
  - :func:`emit_analytics` — always-on fire-and-forget emission to the analytics
    sidecar; depends on the analytics public API, not the collector internals (DIP).

The JSON serialization (:meth:`TokenUsageCollector.to_dict`) is the
``token_usage`` block returned in API responses when ``debug=True``::

    {"total": {"prompt": N, "completion": N, "total": N},
     "calls": [{"op": ..., "model": ..., "prompt": N, "completion": N, "total": N}]}

``thinking`` keys are included only when reasoning tokens are present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

__all__ = [
    "TokenUsageRecord",
    "TokenUsageCollector",
    "extract_token_usage",
    "emit_analytics",
    "model_name_of",
    "aggregate_token_usage",
    "build_ingestion_collector",
]


def model_name_of(model: Any) -> str:
    """Resolve a model-name string from a plain string or an LlmEmbeddingModel-like object."""
    if model is None:
        return ""
    if isinstance(model, str):
        return model
    return getattr(model, "name", str(model))


@dataclass
class TokenUsageRecord:
    """Token usage for a single LLM call."""

    operation: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0

    def to_dict(self) -> dict:
        d = {
            "op": self.operation,
            "model": self.model,
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
            "total": self.total_tokens,
        }
        if self.thinking_tokens:
            d["thinking"] = self.thinking_tokens
        return d


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def extract_token_usage(response: Any, *, operation: str, model: str) -> TokenUsageRecord:
    """
    Build a :class:`TokenUsageRecord` from a LangChain LLM response.

    Accepts:
      - an ``AIMessage`` (reads ``.usage_metadata``);
      - the ``with_structured_output(..., include_raw=True)`` dict
        (``{"raw": AIMessage, ...}`` — the raw message is used);
      - ``None`` / a response without usage → a zero record.

    Never raises: unknown shapes degrade to zeros.
    """
    msg = response
    if isinstance(response, dict) and "raw" in response:
        msg = response.get("raw")

    usage = getattr(msg, "usage_metadata", None)
    if usage:
        prompt = _coerce_int(usage.get("input_tokens"))
        completion = _coerce_int(usage.get("output_tokens"))
        total = _coerce_int(usage.get("total_tokens")) or (prompt + completion)
        details = usage.get("output_token_details") or {}
        thinking = _coerce_int(details.get("reasoning"))
        return TokenUsageRecord(operation, model, prompt, completion, total, thinking)

    # Fallback: provider raw usage stashed in response_metadata (OpenAI-style keys).
    meta = getattr(msg, "response_metadata", None) or {}
    raw_usage = meta.get("token_usage") or meta.get("usage") or {}
    if raw_usage:
        prompt = _coerce_int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens"))
        completion = _coerce_int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens"))
        total = _coerce_int(raw_usage.get("total_tokens")) or (prompt + completion)
        return TokenUsageRecord(operation, model, prompt, completion, total, 0)

    return TokenUsageRecord(operation, model, 0, 0, 0, 0)


class TokenUsageCollector:
    """Accumulates per-call token usage for one API request."""

    def __init__(self) -> None:
        self._calls: List[TokenUsageRecord] = []

    def add(self, record: TokenUsageRecord) -> TokenUsageRecord:
        """Append an already-built record. Returns it for chaining."""
        self._calls.append(record)
        return record

    def record(self, response: Any, *, operation: str, model: str) -> TokenUsageRecord:
        """Extract usage from a LangChain response and append it. Returns the record."""
        return self.add(extract_token_usage(response, operation=operation, model=model))

    @property
    def calls(self) -> List[TokenUsageRecord]:
        return list(self._calls)

    def is_empty(self) -> bool:
        return not self._calls

    def total(self) -> dict:
        total = {
            "prompt": sum(c.prompt_tokens for c in self._calls),
            "completion": sum(c.completion_tokens for c in self._calls),
            "total": sum(c.total_tokens for c in self._calls),
        }
        thinking = sum(c.thinking_tokens for c in self._calls)
        if thinking:
            total["thinking"] = thinking
        return total

    def to_dict(self) -> dict:
        """The ``token_usage`` block for debug API responses."""
        return {"total": self.total(), "calls": [c.to_dict() for c in self._calls]}

    def merge(self, other: "TokenUsageCollector") -> None:
        """Absorb another collector's calls (e.g. aggregate sub-services)."""
        self._calls.extend(other._calls)


def aggregate_token_usage(blocks: Any) -> dict:
    """
    Merge several ``to_dict()`` token_usage blocks into one ``{total, calls}``.

    Used to roll up per-sub-call token usage (e.g. per-operator reports in a bulk
    check) into a single aggregate block. ``None`` / empty blocks are skipped.
    """
    calls: List[dict] = []
    for block in blocks or []:
        if block and block.get("calls"):
            calls.extend(block["calls"])
    total = {
        "prompt": sum(_coerce_int(c.get("prompt")) for c in calls),
        "completion": sum(_coerce_int(c.get("completion")) for c in calls),
        "total": sum(_coerce_int(c.get("total")) for c in calls),
    }
    thinking = sum(_coerce_int(c.get("thinking")) for c in calls)
    if thinking:
        total["thinking"] = thinking
    return {"total": total, "calls": calls}


def build_ingestion_collector(
    *,
    embedding_model: str,
    embedding_tokens: int = 0,
    sc_model: str = "",
    sc_input_tokens: int = 0,
    sc_output_tokens: int = 0,
    sc_total_tokens: int = 0,
) -> TokenUsageCollector:
    """
    Build a collector from already-counted ingestion token usage.

    Embeddings only consume input tokens (``embedding`` op). When situated-context
    enrichment ran, its LLM usage is recorded as a separate ``situated_context`` op.
    Zero-token components are skipped.
    """
    collector = TokenUsageCollector()
    if embedding_tokens:
        collector.add(TokenUsageRecord(
            operation="embedding", model=embedding_model,
            prompt_tokens=embedding_tokens, total_tokens=embedding_tokens,
        ))
    if sc_total_tokens:
        collector.add(TokenUsageRecord(
            operation="situated_context", model=sc_model,
            prompt_tokens=sc_input_tokens, completion_tokens=sc_output_tokens,
            total_tokens=sc_total_tokens,
        ))
    return collector


def emit_analytics(
    collector: TokenUsageCollector,
    *,
    id_project: Any,
    source: str,
    provider: Any = None,
    request_id: Any = None,
    agent_id: Any = None,
) -> None:
    """
    Emit one ``ai.token_usage`` analytics event per recorded call.

    Always attempted regardless of any ``debug`` flag — fire-and-forget, never
    raises, no-op when the collector is empty or analytics is disabled (the
    underlying ``publish_nowait`` already guards the disabled/no-project cases).
    """
    if collector.is_empty():
        return

    import tilellm.analytics as analytics

    for rec in collector.calls:
        event_type, payload = analytics.events.token_usage(
            model=rec.model,
            prompt_tokens=rec.prompt_tokens,
            completion_tokens=rec.completion_tokens,
            total_tokens=rec.total_tokens,
            operation=rec.operation,
            source=source,
            thinking_tokens=rec.thinking_tokens,
            request_id=request_id,
            agent_id=agent_id,
        )
        analytics.publish_nowait(event_type, id_project, payload)
