"""
Temporal Digest — FastAPI controllers.

Endpoints:
  POST /api/digest/generate         Generate digests for a date range (manual trigger / nightly batch)
  GET  /api/digest/{namespace}/{date} Retrieve existing digest for a namespace + specific date
  POST /api/digest/query             Query with automatic temporal/semantic routing
"""
import json
import logging
import os
from datetime import date
from typing import Union

from fastapi import APIRouter, HTTPException, Query

from tilellm.modules.temporal_digest.logic import agent_query_digest, generate_digest, query_digest
from tilellm.modules.temporal_digest.models.schemas import (
    DigestAgentRequest,
    DigestAgentResponse,
    DigestGenerationRequest,
    DigestGenerationResponse,
    DigestQueryRequest,
    DigestQueryResponse,
)
from tilellm.models import Engine

logger = logging.getLogger(__name__)

ENABLE_TASKIQ = os.environ.get("ENABLE_TASKIQ", "false").lower() == "true"

try:
    from tilellm.modules.task_executor.broker import broker as _broker  # noqa: F401
    _TASKIQ_AVAILABLE = True
except Exception:
    _TASKIQ_AVAILABLE = False

router = APIRouter(prefix="/api/digest", tags=["Temporal Digest"])


@router.post("/generate", response_model=Union[DigestGenerationResponse, dict])
async def generate_digest_endpoint(request: DigestGenerationRequest):
    """
    Generate temporal digests for all time windows in the requested date range.

    When **TaskIQ is enabled** (``ENABLE_TASKIQ=true``), the job is queued and the
    endpoint returns immediately with ``{"task_id": "...", "status": "queued"}``.
    Use the task_id to poll ``/api/task/{task_id}`` and the optional ``webhook_url``
    field to receive a notification on completion.

    When TaskIQ is not available the endpoint runs synchronously and returns the
    full ``DigestGenerationResponse``.

    - Set **force_regenerate=true** to re-generate even if a digest already exists.
    - Set **domain** to a pre-built prompt key (e.g. ``pa_italiana``) for domain-specific summaries.
    - **top_k** controls how many source chunks are retrieved per window (default 1000).
    """
    if ENABLE_TASKIQ and _TASKIQ_AVAILABLE:
        try:
            from tilellm.modules.task_executor.tasks import task_digest_generate
            from tilellm.shared.llm_config import serialize_with_secrets

            request_dict = serialize_with_secrets(request.model_dump(mode="python"))
            task = await task_digest_generate.kiq(request_dict)
            logger.info(f"Digest generation queued: task_id={task.task_id} namespace={request.namespace}")
            return {
                "task_id": task.task_id,
                "namespace": request.namespace,
                "status": "queued",
                "message": (
                    f"Digest generation queued for namespace='{request.namespace}' "
                    f"from {request.date_from} granularity={request.granularity}. "
                    "Use task_id to poll status."
                ),
            }
        except Exception as e:
            logger.error(f"Failed to queue digest generation task: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # Synchronous fallback
    try:
        return await generate_digest(request)
    except Exception as e:
        logger.error(f"Digest generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=DigestQueryResponse)
async def query_digest_endpoint(request: DigestQueryRequest):
    """
    Answer a question by routing to the most appropriate retrieval strategy.

    - **query_mode=auto** (default): classifies the question automatically.
      - Temporal/aggregative questions ("cosa hanno fatto oggi?", "riassunto della settimana")
        → retrieves pre-generated digest vectors.
      - Specific semantic questions ("hanno acquistato antibiotici?")
        → standard vector search on raw chunks.
    - **query_mode=temporal**: force digest retrieval.
    - **query_mode=semantic**: force raw chunk vector search.

    Optionally restrict retrieval to a date range with **date_from** / **date_to**.
    """
    try:
        return await query_digest(request)
    except Exception as e:
        logger.error(f"Digest query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa", response_model=DigestAgentResponse)
async def agent_query_digest_endpoint(request: DigestAgentRequest):
    """
    Agentic digest query — no need to specify dates or query_mode manually.

    The LLM analyzes the free-form question and conversation history to extract:
    - **date_from / date_to**: temporal references ("la settimana scorsa", "ad aprile", "ieri")
    - **query_mode**: temporal (aggregate/digest retrieval) vs semantic (specific fact search)

    The response includes the extracted parameters in `extracted_date_from`,
    `extracted_date_to`, `extracted_query_mode`, and `agent_reasoning` so the caller
    can inspect what the agent decided.

    Optionally pass **chat_history_dict** to carry conversation context across turns.
    """
    try:
        return await agent_query_digest(request)
    except Exception as e:
        logger.error(f"Agent digest query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{namespace}/{digest_date}", response_model=DigestQueryResponse)
async def get_digest_for_date(
    namespace: str,
    digest_date: date,
    engine_json: str = Query(..., description="JSON-serialized Engine configuration."),
    gptkey: str = Query(..., description="LLM API key."),
    embedding: str = Query(default="text-embedding-3-small"),
    model: str = Query(default="gpt-4o-mini"),
    llm: str = Query(default="openai"),
    date_metadata_field: str = Query(default="date"),
):
    """
    Convenience endpoint: retrieve the digest for a specific namespace + date.

    Passes ``query_mode='temporal'`` with ``date_from == date_to == digest_date``
    so only the pre-generated digest for that exact day is returned.

    **Note**: Requires that a digest has already been generated for this date.
    Use ``POST /api/digest/generate`` to create it if missing.

    The ``engine_json`` query parameter must be a JSON-encoded Engine object, e.g.:
    ``{"index_name":"tiledesk","engine_type":"qdrant_local","qdrant_url":"http://localhost:6333","text_key":"text","metric":"cosine","vector_size":1536}``
    """
    try:
        engine_data = json.loads(engine_json)
        engine = Engine(**engine_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid engine_json: {e}")

    request = DigestQueryRequest(
        question=f"Fornisci il digest degli atti amministrativi del {digest_date.isoformat()}.",
        namespace=namespace,
        date_from=digest_date,
        date_to=digest_date,
        engine=engine,
        embedding=embedding,
        gptkey=gptkey,
        model=model,
        llm=llm,
        query_mode="temporal",
        date_metadata_field=date_metadata_field,
    )

    try:
        return await query_digest(request)
    except Exception as e:
        logger.error(f"GET digest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
