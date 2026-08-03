"""
Temporal Digest — business logic entry points.
DI decorators inject LLM + repo; service layer does the actual work.

Analytics (docs/MIGLIORIE_DA_FARE.md P1#14): kb.content_indexed on generate
(digests are indexed into the vector store), kb.query_executed on query/
agent_query. Wired at this outer layer rather than inside digest_service.py:
DigestService.generate/query/agent_query each make multiple internal LLM
calls (act_type classifier batch, judge/synthesis, rollup) — per-call
ai.token_usage/ai.model_call would require instrumenting each call site
individually, deferred as a documented ceiling (same scope decision as
knowledge_graph_falkor's QA endpoints).
"""
import time
from typing import Optional

import tilellm.analytics as analytics
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async
from tilellm.modules.temporal_digest.models.schemas import (
    DigestAgentRequest,
    DigestAgentResponse,
    DigestGenerationRequest,
    DigestGenerationResponse,
    DigestQueryRequest,
    DigestQueryResponse,
)
from tilellm.modules.temporal_digest.services.digest_service import DigestService

_service = DigestService()


@inject_llm_chat_async
@inject_repo_async
async def generate_digest(
    request: DigestGenerationRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    embedding_config_key=None,
    **kwargs,
) -> DigestGenerationResponse:
    """Public DI-wired entry point; the core stays plain/testable without a
    live LLM/repo connection (mirrors ingest_md/qa_lgraph_hybrid)."""
    return await _generate_digest_core(request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)


async def _generate_digest_core(request: DigestGenerationRequest, repo, llm, llm_embeddings) -> DigestGenerationResponse:
    t0 = time.monotonic()
    error_msg: Optional[str] = None
    result: Optional[DigestGenerationResponse] = None
    try:
        result = await _service.generate(request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)
        return result
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        duration_ms = int((time.monotonic() - t0) * 1000)
        event_type, payload = analytics.events.content_indexed(
            kb_id=request.namespace,
            kb_name=request.namespace,
            embedding_model=analytics.events.get_embedding_model_name(request.embedding),
            engine=analytics.events.get_engine_value(request.engine),
            duration_ms=duration_ms,
            success=error_msg is None,
            source_type="temporal_digest_generate",
            chunks_indexed=result.total_chunks_processed if result else 0,
            error_message=error_msg,
            request_id=request.request_id,
        )
        analytics.publish_nowait(event_type, request.id_project, payload)


@inject_llm_chat_async
@inject_repo_async
async def query_digest(
    request: DigestQueryRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    embedding_config_key=None,
    **kwargs,
) -> DigestQueryResponse:
    """Public DI-wired entry point; the core stays plain/testable without a
    live LLM/repo connection (mirrors ingest_md/qa_lgraph_hybrid)."""
    return await _query_digest_core(request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)


async def _query_digest_core(request: DigestQueryRequest, repo, llm, llm_embeddings) -> DigestQueryResponse:
    t0 = time.monotonic()
    error_msg: Optional[str] = None
    result: Optional[DigestQueryResponse] = None
    try:
        result = await _service.query(request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)
        return result
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        latency_ms = int((time.monotonic() - t0) * 1000)
        event_type, payload = analytics.events.kb_query(
            kb_id=request.namespace,
            kb_name=request.namespace,
            query_text=request.question,
            chunks_retrieved=result.chunk_count if result else 0,
            reranking_applied=bool(request.reranking),
            latency_ms=latency_ms,
            request_id=request.request_id,
            success=error_msg is None,
        )
        analytics.publish_nowait(event_type, request.id_project, payload)


@inject_llm_chat_async
@inject_repo_async
async def agent_query_digest(
    request: DigestAgentRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    embedding_config_key=None,
    **kwargs,
) -> DigestAgentResponse:
    """Public DI-wired entry point; the core stays plain/testable without a
    live LLM/repo connection (mirrors ingest_md/qa_lgraph_hybrid)."""
    return await _agent_query_digest_core(request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)


async def _agent_query_digest_core(request: DigestAgentRequest, repo, llm, llm_embeddings) -> DigestAgentResponse:
    t0 = time.monotonic()
    error_msg: Optional[str] = None
    result: Optional[DigestAgentResponse] = None
    try:
        result = await _service.agent_query(request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)
        return result
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        latency_ms = int((time.monotonic() - t0) * 1000)
        event_type, payload = analytics.events.kb_query(
            kb_id=request.namespace,
            kb_name=request.namespace,
            query_text=request.question,
            chunks_retrieved=result.chunk_count if result else 0,
            reranking_applied=bool(request.reranking),
            latency_ms=latency_ms,
            request_id=request.request_id,
            success=error_msg is None,
        )
        analytics.publish_nowait(event_type, request.id_project, payload)
