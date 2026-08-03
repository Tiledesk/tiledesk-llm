import json
import logging
import time
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from starlette.responses import JSONResponse as StarletteJSONResponse

import tilellm.analytics as analytics
from tilellm.agents.workflow import app, simple_app
from tilellm.models import ItemSingle, QuestionAnswer
from tilellm.models.schemas import IndexingResult, RetrievalResult
from tilellm.modules.api_v2.dependencies import get_redis_client
from tilellm.modules.api_v2.models import QASimpleRequest
from tilellm.modules.api_v2.services.ingestion_v2_service import ingest_v2
from tilellm.modules.api_v2.services.scrape_single_service import ScrapeSingleService
from tilellm.modules.api_v2.services.scrape_status_service import ScrapeStatusService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2",
    tags=["Agentic API v2"],
)


def _emit_api_v2_kb_query(payload, response: JSONResponse, latency_ms: int) -> None:
    """kb.query_executed for /query and /qa — mirrors /api/qa's own emission
    in __main__.py. token_usage/model_call are NOT duplicated here: they
    already fire inside rag_node -> ask_with_memory/ask_hybrid_with_memory
    (controller.py), which both graphs route through."""
    try:
        body = json.loads(response.body.decode())
    except Exception:
        body = {}
    question_text = payload.question if isinstance(payload.question, str) else str(payload.question)
    event_type, evt_payload = analytics.events.kb_query(
        kb_id=payload.namespace,
        kb_name=payload.namespace,
        query_text=question_text,
        chunks_retrieved=len(body.get("content_chunks") or []),
        reranking_applied=bool(getattr(payload, "reranking", False)),
        latency_ms=latency_ms,
        request_id=payload.request_id,
        success=body.get("success") if response.status_code < 400 else False,
    )
    analytics.publish_nowait(event_type, payload.id_project, evt_payload)


def _build_response(final_state: dict) -> JSONResponse:
    """Shared response builder for both /query and /qa."""
    if final_state.get("is_on_topic") == "no":
        return JSONResponse(
            status_code=400,
            content={"detail": "Domanda non pertinente o non sicura."},
        )

    retrieval = final_state.get("retrieval_result")
    if retrieval is None:
        return JSONResponse(status_code=500, content={"detail": "RAG returned no result"})

    if isinstance(retrieval, StarletteJSONResponse):
        response_body = json.loads(retrieval.body)
    else:
        response_body = retrieval.model_dump()

    compliance_report = final_state.get("compliance_report")
    if compliance_report:
        response_body["compliance_report"] = compliance_report

    return JSONResponse(content=response_body)


@router.post("/query")
async def ask_question_full(payload: QuestionAnswer):
    """
    Full agentic Q&A workflow.

    Pipeline: input_guard → intent_router → cache_lookup → HyDE →
              RAPTOR | RAG → hallucination_grader (retry loop) → cache_store

    Routes automatically to compliance check when the question contains RTM
    requirements; otherwise runs standard RAG with self-correction.
    """
    initial_state = {
        "question_answer": payload,
        "retry_count": 0,
        "max_retries": 3,
        "metadata": {
            "search_type": payload.search_type,
            "trace": [],
        },
    }
    t0 = time.monotonic()
    final_state = await app.ainvoke(initial_state)
    response = _build_response(final_state)
    _emit_api_v2_kb_query(payload, response, int((time.monotonic() - t0) * 1000))
    return response


@router.post("/scrape/single", response_model=IndexingResult, tags=["Scrape v2"])
async def scrape_single_v2(
    item: ItemSingle,
    redis_client: Redis = Depends(get_redis_client),
):
    """
    Index a single document into the vector store.

    Mirrors /api/scrape/single with SOLID-compliant service decomposition:
    ``ScrapeStatusService`` owns Redis lifecycle writes; ``ScrapeSingleService``
    owns indexing orchestration and analytics.
    """
    status_svc = ScrapeStatusService(redis_client)
    await status_svc.set_started(item.namespace, item.id)
    try:
        result = await ScrapeSingleService().run(item)
        await status_svc.set_finished(item.namespace, item.id)
        return JSONResponse(content=result.model_dump(exclude_none=True))
    except Exception as e:
        await status_svc.set_error(item.namespace, item.id)
        traceback.print_exc()
        logger.error(e)
        return JSONResponse(status_code=400, content=e.args[0] if e.args else str(e))


@router.post("/ingestion", tags=["Ingestion v2"])
async def unified_ingestion_v2(item: ItemSingle):
    """
    Unified ingestion — canonical MD+frontmatter form for every document type.

    Same request contract as the legacy ``/api/ingestion`` (ItemSingle): auto
    type detection, ``hybrid``/``sparse_encoder`` for dense-only vs dense+sparse
    indexing, ``situated_context`` for Contextual Retrieval. What changes is
    the default path: instead of the legacy per-backend ``add_item``, content
    goes through the same converters as ``/api/export/md`` and is written with
    ``ingest.service.write_extracted_document`` — every chunk carries the same
    baseline provenance metadata (document identity + page, when applicable)
    regardless of source file type. ``use_ocr=True`` (pdf/docx) still routes to
    the existing Docling/LLM-enrichment pipeline unchanged, for callers who
    want generated image captions/table descriptions.

    ``/api/ingestion`` (legacy, in production) is untouched by this endpoint.
    """
    return await ingest_v2(item)


@router.post("/qa")
async def ask_question_simple(payload: QASimpleRequest):
    """
    Simplified agentic Q&A endpoint with optional guard and hallucination grader.

    Pipeline: [guard] → intent_router → compliance | RAG → [hallucination_grader]

    Flags:
      - ``use_guard`` (default True): set False to skip the input safety guard.
      - ``use_hallucination_grader`` (default True): set False to return the RAG
        answer without grounding verification.

    Routes to compliance check when the question contains RTM requirements;
    otherwise runs standard RAG.
    """
    initial_state = {
        "question_answer": payload,
        "retry_count": 0,
        "max_retries": 3,
        "metadata": {
            "search_type": payload.search_type,
            "trace": [],
        },
    }
    t0 = time.monotonic()
    final_state = await simple_app.ainvoke(initial_state)
    response = _build_response(final_state)
    _emit_api_v2_kb_query(payload, response, int((time.monotonic() - t0) * 1000))
    return response
