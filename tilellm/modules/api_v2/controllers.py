import json
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.responses import JSONResponse as StarletteJSONResponse

from tilellm.agents.workflow import app
from tilellm.models import QuestionAnswer
from tilellm.models.schemas import RetrievalResult

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2",
    tags=["Agentic API v2"],
)


@router.post("/qa")
async def ask_question(payload: QuestionAnswer):
    """
    Agentic Q&A endpoint.

    Routes automatically between:
    - Compliance check (when the question contains RTM requirements)
    - Standard RAG Q&A with hallucination grading and self-correction

    Returns RetrievalResult JSON; compliance responses include an extra
    ``compliance_report`` key with the full structured report + RTM CSV.
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

    # Attach compliance report when present (intent == "compliance")
    compliance_report = final_state.get("compliance_report")
    if compliance_report:
        response_body["compliance_report"] = compliance_report

    return JSONResponse(content=response_body)