import json
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.responses import JSONResponse as StarletteJSONResponse

from tilellm.agents.workflow import app, simple_app
from tilellm.models import QuestionAnswer
from tilellm.models.schemas import RetrievalResult
from tilellm.modules.api_v2.models import QASimpleRequest

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
