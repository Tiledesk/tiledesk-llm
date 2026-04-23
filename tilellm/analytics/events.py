"""
Type-safe analytics event builder functions.

Each function returns a (event_type, payload) tuple ready for
passing to analytics.publish_nowait(event_type, id_project, payload).

All fields that may be null are typed Optional — callers should pass
None when a value is not available, rather than omitting the key entirely.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

__all__ = [
    "token_usage",
    "model_call",
    "kb_query",
    "content_indexed",
    "tool_call",
    "get_reranker_model",
    "get_embedding_model_name",
    "get_engine_value",
]


def token_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    operation: str,
    source: str,
    thinking_tokens: int = 0,
    request_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Build an ai.token_usage event payload.

    Args:
        model:             Model name string (e.g. 'gpt-4o').
        prompt_tokens:     Input token count.
        completion_tokens: Output token count.
        total_tokens:      Total token count.
        operation:         'qa' | 'ask' | 'thinking'.
        source:            'kb' | 'chat'.
        thinking_tokens:   Reasoning tokens (GPT-5, Claude 4+, Gemini 2.5+, DeepSeek).
        request_id:        Tiledesk conversation/request ID.
        agent_id:          Tiledesk agent/bot ID.
    """
    return "ai.token_usage", {
        "model":             model,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
        "thinking_tokens":   thinking_tokens,
        "operation":         operation,
        "source":            source,
        "request_id":        request_id,
        "agent_id":          agent_id,
    }


def model_call(
    model: str,
    provider: str,
    operation: str,
    latency_ms: int,
    success: bool,
    error_type: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Build an ai.model_call event payload.

    Args:
        model:      Model name string.
        provider:   LLM provider (e.g. 'openai', 'anthropic').
        operation:  'qa' | 'ask' | 'thinking'.
        latency_ms: Wall-clock ms for the ainvoke/astream call only.
        success:    True unless exception was raised.
        error_type: type(e).__name__ on exception, else None.
        request_id: Tiledesk conversation/request ID.
    """
    return "ai.model_call", {
        "model":      model,
        "provider":   provider,
        "operation":  operation,
        "latency_ms": latency_ms,
        "success":    success,
        "error_type": error_type,
        "request_id": request_id,
    }


def kb_query(
    kb_id: str,
    kb_name: str,
    query_text: str,
    chunks_retrieved: int,
    reranking_applied: bool,
    latency_ms: int,
    reranker_model: Optional[str] = None,
    reranker_latency_ms: Optional[int] = None,
    request_id: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Build a kb.query_executed event payload.

    Args:
        kb_id:               Namespace / KB ID.
        kb_name:             Display name (same as kb_id when no alias).
        query_text:          The raw user query sent to the vector store.
        chunks_retrieved:    Number of chunks returned.
        reranking_applied:   True when a reranker was used.
        latency_ms:          Total retrieval + generation latency.
        reranker_model:      Reranker model name (null when not applied).
        reranker_latency_ms: Reranker-only latency (omitted when not applied).
        request_id:          Tiledesk conversation/request ID.
    """
    payload: dict = {
        "kb_id":             kb_id,
        "kb_name":           kb_name,
        "query_text":        query_text,
        "chunks_retrieved":  chunks_retrieved,
        "reranking_applied": reranking_applied,
        "reranker_model":    reranker_model,
        "latency_ms":        latency_ms,
        "request_id":        request_id,
    }
    if reranker_latency_ms is not None:
        payload["reranker_latency_ms"] = reranker_latency_ms
    return "kb.query_executed", payload


def content_indexed(
    kb_id: str,
    kb_name: str,
    embedding_model: str,
    engine: str,
    duration_ms: int,
    success: bool,
    source_url: Optional[str] = None,
    source_type: Optional[str] = None,
    chunks_indexed: int = 0,
    error_message: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Build a kb.content_indexed event payload.

    Args:
        kb_id:          Namespace / KB ID.
        kb_name:        Display name.
        embedding_model: Embedding model name string.
        engine:         Vector engine value string (e.g. 'pinecone', 'qdrant').
        duration_ms:    Wall-clock ms for the full indexing call.
        success:        True on normal return, False on exception.
        source_url:     Source URL (may be null for text content).
        source_type:    Content type ('url', 'text', 'pdf', 'docx', etc.).
        chunks_indexed: Number of chunks successfully indexed.
        error_message:  str(e) on exception, else null.
        request_id:     Tiledesk conversation/request ID.
    """
    return "kb.content_indexed", {
        "kb_id":           kb_id,
        "kb_name":         kb_name,
        "source_url":      source_url,
        "source_type":     source_type,
        "embedding_model": embedding_model,
        "engine":          engine,
        "chunks_indexed":  chunks_indexed,
        "duration_ms":     duration_ms,
        "success":         success,
        "error_message":   error_message,
        "request_id":      request_id,
    }


def tool_call(
    tool_name: str,
    tool_provider: str,
    model: str,
    latency_ms: int,
    success: bool,
    operation: Optional[str] = None,
    error_type: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Build an ai.tool_call event payload.

    Args:
        tool_name:     MCP/internal tool name.
        tool_provider: MCP server name or 'internal'.
        model:         Model name that invoked the tool.
        latency_ms:    Wall-clock ms for the tool invocation.
        success:       True unless tool raised.
        operation:     Null (reserved for future use).
        error_type:    type(e).__name__ on exception, else None.
        request_id:    Tiledesk conversation/request ID.
    """
    return "ai.tool_call", {
        "tool_name":     tool_name,
        "tool_provider": tool_provider,
        "operation":     operation,
        "model":         model,
        "latency_ms":    latency_ms,
        "success":       success,
        "error_type":    error_type,
        "request_id":    request_id,
    }


# ---------------------------------------------------------------------------
# Derivation helpers (shared between controller.py and __main__.py)
# ---------------------------------------------------------------------------

def get_reranker_model(question_answer: Any) -> Optional[str]:
    """
    Derive the reranker model name string from a QuestionAnswer object.
    Returns None when reranking is disabled.
    """
    r = getattr(question_answer, "reranking", False)
    if r is False or r is None:
        return None
    if r is True:
        return getattr(question_answer, "reranker_model", None)
    if hasattr(r, "name"):          # TEIConfig or PineconeRerankerConfig
        return r.name
    return str(r)


def get_embedding_model_name(embedding: Any) -> str:
    """Return embedding model name as string from str or LlmEmbeddingModel."""
    if isinstance(embedding, str):
        return embedding
    return getattr(embedding, "name", str(embedding))


def get_engine_value(engine: Any) -> str:
    """Return engine value as string from Engine enum or plain string."""
    if hasattr(engine, "value"):
        return engine.value
    return str(engine)
