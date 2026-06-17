import json
import logging
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from starlette.responses import JSONResponse as StarletteJSONResponse

from tilellm.agents.workflow import app, simple_app
from tilellm.models import ItemSingle, QuestionAnswer
from tilellm.models.schemas import IndexingResult, RetrievalResult
from tilellm.modules.api_v2.dependencies import get_redis_client
from tilellm.modules.api_v2.models import QASimpleRequest
from tilellm.modules.api_v2.services.scrape_single_service import ScrapeSingleService
from tilellm.modules.api_v2.services.scrape_status_service import ScrapeStatusService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2",
    tags=["Agentic API v2"],
)


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
    final_state = await app.ainvoke(initial_state)
    return _build_response(final_state)


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
    final_state = await simple_app.ainvoke(initial_state)
    return _build_response(final_state)
