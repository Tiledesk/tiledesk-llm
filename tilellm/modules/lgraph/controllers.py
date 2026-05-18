"""
FastAPI controllers for the Light GraphRAG (lgraph) module.

Endpoints
---------
POST   /api/lgraph/build              Build light graph (async TaskIQ)
GET    /api/lgraph/tasks/{id}         Poll async task status
POST   /api/lgraph/search             PPR-based retrieval (sync)
POST   /api/lgraph/qa                 PPR + LLM answer
POST   /api/lgraph/leiden             Leiden community detection (async TaskIQ)
GET    /api/lgraph/network            Visualization data
DELETE /api/lgraph/{namespace}        Delete graph for a namespace/index
GET    /api/lgraph/health             Health check
"""

import logging
import os

from fastapi import APIRouter, HTTPException, Query, status

from .models.schemas import (
    LGraphAsyncTaskResponse,
    LGraphBuildRequest,
    LGraphDeleteResponse,
    LGraphLeidenAsyncTaskResponse,
    LGraphLeidenRequest,
    LGraphLeidenResponse,
    LGraphNetworkResponse,
    LGraphQARequest,
    LGraphQAResponse,
    LGraphSearchRequest,
    LGraphSearchResponse,
    LGraphTaskPollResponse,
)
from tilellm.shared.llm_config import serialize_with_secrets

logger = logging.getLogger(__name__)

from . import logic as lgraph_logic

# ---- TaskIQ wiring (optional — gracefully disabled if unavailable) --------
try:
    from tilellm.modules.task_executor.tasks import task_lgraph_build, task_lgraph_leiden
    from tilellm.modules.task_executor.broker import broker
    TASKIQ_AVAILABLE = True
except Exception as e:
    logger.warning(f"[lgraph] TaskIQ not available: {e}")
    TASKIQ_AVAILABLE = False
    task_lgraph_build = None
    task_lgraph_leiden = None
    broker = None

ENABLE_TASKIQ = os.environ.get("ENABLE_TASKIQ", "false").lower() == "true" and TASKIQ_AVAILABLE

# ---- Router ------------------------------------------------------------------
router = APIRouter(prefix="/api/lgraph", tags=["Light GraphRAG (LLM-free)"])


# ---- Health ------------------------------------------------------------------

@router.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """Check FalkorDB connectivity used by the lgraph module."""
    try:
        repo = lgraph_logic._get_falkor_repo()
        ok = await repo.verify_connection()
        if not ok:
            raise HTTPException(status_code=503, detail="FalkorDB connection failed")
        return {"status": "healthy"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---- Task polling -----------------------------------------------------------

@router.get("/tasks/{task_id}", response_model=LGraphTaskPollResponse)
async def get_task_status(task_id: str):
    """Poll the status of an async build or leiden task."""
    if not ENABLE_TASKIQ:
        raise HTTPException(status_code=501, detail="Async tasks not enabled (ENABLE_TASKIQ=false)")
    try:
        result_backend = broker.result_backend
        is_ready = await result_backend.is_result_ready(task_id)
        if not is_ready:
            return LGraphTaskPollResponse(task_id=task_id, status="in_progress")
        result = await result_backend.get_result(task_id)
        if result.is_err:
            return LGraphTaskPollResponse(
                task_id=task_id,
                status="failed",
                error=str(result.error) if hasattr(result, "error") else str(result.return_value),
            )
        return LGraphTaskPollResponse(task_id=task_id, status="success", result=result.return_value)
    except Exception as e:
        logger.error(f"[lgraph] task status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Build ------------------------------------------------------------------

@router.post("/build", response_model=LGraphAsyncTaskResponse)
async def build_graph(request: LGraphBuildRequest):
    """Build the light graph over an existing vector store namespace (async via TaskIQ)."""
    if not ENABLE_TASKIQ:
        raise HTTPException(
            status_code=501,
            detail="TaskIQ is required for graph build (set ENABLE_TASKIQ=true and start a worker).",
        )
    try:
        payload = serialize_with_secrets(request.model_dump(mode="python"))
        task = await task_lgraph_build.kiq(payload)
        logger.info(f"[lgraph] build task queued task_id={task.task_id}")
        return LGraphAsyncTaskResponse(task_id=task.task_id)
    except Exception as e:
        logger.error(f"[lgraph] build dispatch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Search (PPR only) ------------------------------------------------------

@router.post("/search", response_model=LGraphSearchResponse)
async def search_graph(request: LGraphSearchRequest):
    """Query the light graph using Personalized PageRank (no LLM)."""
    try:
        return await lgraph_logic.search_lgraph(request)
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"[lgraph] search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- QA (PPR + LLM) ---------------------------------------------------------

@router.post("/qa", response_model=LGraphQAResponse)
async def qa_graph(request: LGraphQARequest):
    """
    PPR retrieval + LLM answer.

    Extracts entities from the question, uses them as PPR seeds to find the most
    relevant chunks, then calls the LLM to produce an exhaustive answer.
    Supports optional date_from/date_to for temporal filtering on DATE_IT entities.
    """
    try:
        return await lgraph_logic.qa_lgraph(request)
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"[lgraph] qa error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---- Leiden clustering ------------------------------------------------------

@router.post("/leiden", response_model=LGraphLeidenAsyncTaskResponse)
async def leiden_cluster(request: LGraphLeidenRequest):
    """
    Run Leiden community detection on the LEntity CO_OCCURS graph (async via TaskIQ).

    Assigns a community_id to each LEntity node. Communities can be used for
    community-aware PPR or thematic digest generation.
    """
    if not ENABLE_TASKIQ:
        raise HTTPException(
            status_code=501,
            detail="TaskIQ is required for Leiden clustering (set ENABLE_TASKIQ=true and start a worker).",
        )
    try:
        payload = serialize_with_secrets(request.model_dump(mode="python"))
        task = await task_lgraph_leiden.kiq(payload)
        logger.info(f"[lgraph] leiden task queued task_id={task.task_id}")
        return LGraphLeidenAsyncTaskResponse(task_id=task.task_id)
    except Exception as e:
        logger.error(f"[lgraph] leiden dispatch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Network (visualization) ------------------------------------------------

@router.get("/network", response_model=LGraphNetworkResponse)
async def get_network(
    namespace: str = Query(..., description="Tenant namespace"),
    index_name: str = Query(..., description="Vector store index / collection name"),
    node_limit: int = Query(500, ge=1, le=5000),
    edge_limit: int = Query(2000, ge=1, le=20000),
):
    """Return the lgraph nodes and edges for visualization."""
    try:
        return await lgraph_logic.get_lgraph_network(
            namespace=namespace,
            index_name=index_name,
            node_limit=node_limit,
            edge_limit=edge_limit,
        )
    except Exception as e:
        logger.error(f"[lgraph] network error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Delete -----------------------------------------------------------------

@router.delete("/{namespace}", response_model=LGraphDeleteResponse)
async def delete_graph(
    namespace: str,
    index_name: str = Query(..., description="Vector store index / collection name"),
):
    """Delete the light graph for a namespace+index pair."""
    try:
        return await lgraph_logic.delete_lgraph(namespace=namespace, index_name=index_name)
    except Exception as e:
        logger.error(f"[lgraph] delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
