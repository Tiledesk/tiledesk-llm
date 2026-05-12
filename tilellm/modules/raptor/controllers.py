"""
RAPTOR API Controllers.

Provides REST endpoints for:
- Building RAPTOR trees
- Retrieving with RAPTOR strategies
- Managing summary trees
"""

import asyncio
import logging
import os
import time
from typing import Optional, Any, Union

from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document

from tilellm.models.schemas.general_schemas import AsyncTaskResponse
from tilellm.modules.raptor.models.models import (
    RaptorRequest,
    RaptorResponse,
    RaptorRetrievalRequest,
    RaptorRetrievalResult,
    RaptorSummaryRequest,
    RaptorSummaryResponse,
    RaptorQARequest,
    RaptorQAResponse,
)
from tilellm.modules.raptor.services.raptor_service import RaptorService
from tilellm.modules.raptor.services.retrieval_strategies import RaptorRetriever
from tilellm.modules.raptor.repository import RaptorRepository
from tilellm.modules.raptor.config_loader import (
    is_raptor_enabled,
    get_raptor_config_from_env,
    should_use_raptor_for_document,
)
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TaskIQ — optional async dispatch
# ---------------------------------------------------------------------------
try:
    from tilellm.modules.task_executor.tasks import task_raptor_build
    from tilellm.modules.task_executor.broker import broker
    TASKIQ_AVAILABLE = True
except Exception as _e:
    logger.debug("TaskIQ not available for RAPTOR: %s", _e)
    TASKIQ_AVAILABLE = False
    broker = None
    task_raptor_build = None

ENABLE_TASKIQ = os.environ.get("ENABLE_TASKIQ", "false").lower() == "true"
ENABLE_TASKIQ = ENABLE_TASKIQ and TASKIQ_AVAILABLE

# Create router
router = APIRouter(prefix="/api/raptor", tags=["RAPTOR"])

# Module-level singleton for RAPTOR repository (to avoid connection leaks)
_raptor_repo: Optional[RaptorRepository] = None


# ---------------------------------------------------------------------------
# Helper: Get RAPTOR repository
# ---------------------------------------------------------------------------

async def get_raptor_repo() -> RaptorRepository:
    """
    Get RaptorRepository instance backed by Redis.

    Uses a module-level singleton to avoid creating new Redis connections
    on every request (which would cause a connection leak).
    """
    global _raptor_repo

    if _raptor_repo is None:
        import redis.asyncio as redis
        from tilellm.shared.utility import get_service_config

        config = get_service_config()
        redis_config = config.get('redis', {})
        redis_url = (
            f"redis://{redis_config.get('host', 'localhost')}:"
            f"{redis_config.get('port', 6379)}/"
            f"{redis_config.get('db', 0)}"
        )
        redis_client = redis.from_url(redis_url)
        _raptor_repo = RaptorRepository(redis_client=redis_client)

    return _raptor_repo


# ---------------------------------------------------------------------------
# Business logic functions with DI decorators
# ---------------------------------------------------------------------------

@inject_llm_chat_async
@inject_repo_async
async def _build_raptor_tree_logic(
    request: RaptorRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs,
) -> RaptorResponse:
    """Core logic for building RAPTOR tree, with injected LLM and repo."""
    config = request.config or get_raptor_config_from_env()

    if not should_use_raptor_for_document(
        doc_type=request.doc_type,
        page_count=request.page_count
    ):
        logger.info(
            f"RAPTOR not activated for doc {request.doc_id} "
            f"(type={request.doc_type}, pages={request.page_count})"
        )
        return RaptorResponse(
            success=False,
            error="Document does not meet RAPTOR activation criteria",
        )

    # Retrieve chunks from vector store
    chunks = await _retrieve_document_chunks(
        namespace=request.namespace,
        doc_id=request.doc_id,
        chunk_ids=request.chunk_ids,
        vector_repo=repo,
        engine=request.engine,
    )
    
    # Ensure chunks is a List[Document] (defensive programming)
    if not isinstance(chunks, list):
        chunks = _to_documents(chunks)

    if not chunks:
        return RaptorResponse(
            success=False,
            error=f"No chunks found for document {request.doc_id}",
            total_chunks=0,
        )

    raptor_repo = await get_raptor_repo()
    service = RaptorService(repo=raptor_repo)
    return await service.build_raptor_tree(
        chunks=chunks,
        namespace=request.namespace,
        doc_id=request.doc_id,
        llm=llm,
        embeddings=llm_embeddings,
        vector_repo=repo,
        engine=request.engine,
        config=config,
        sparse_encoder=request.sparse_encoder,
    )


@inject_llm_chat_async
@inject_repo_async
async def _retrieve_raptor_logic(
    request: RaptorRetrievalRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs,
) -> RaptorRetrievalResult:
    """Core logic for RAPTOR retrieval, with injected LLM and repo."""
    raptor_repo = await get_raptor_repo()
    retriever = RaptorRetriever(repo=raptor_repo)
    return await retriever.retrieve(
        request=request,
        llm=llm,
        embeddings=llm_embeddings,
        vector_repo=repo,
    )


@inject_llm_chat_async
@inject_repo_async
async def _summarize_logic(
    request: RaptorSummaryRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs,
) -> RaptorSummaryResponse:
    """Core logic for chunk summarization, with injected LLM and repo."""
    start_time = time.time()

    chunks = await _retrieve_chunks_by_ids(
        namespace=request.namespace,
        chunk_ids=request.chunk_ids,
        vector_repo=repo,
        engine=request.engine,
        doc_id=request.doc_id,
    )

    if not chunks:
        return RaptorSummaryResponse(
            success=False,
            error="No chunks found for specified IDs",
        )

    if llm is None:
        logger.error("LLM is not available in _summarize_logic")
        return RaptorSummaryResponse(
            success=False,
            error="LLM service is not available",
        )

    groups = [
        chunks[i:i + request.cluster_size]
        for i in range(0, len(chunks), request.cluster_size)
    ]

    from tilellm.modules.raptor.prompts import RAPTOR_SUMMARY_PROMPT

    async def _summarize_group(group_idx: int, group: list) -> dict:
        context = "\n\n".join([c.page_content for c in group])
        prompt = RAPTOR_SUMMARY_PROMPT.format(context=context)
        response = await llm.ainvoke(prompt)
        return {
            "group_id": group_idx,
            "chunk_ids": [c.metadata.get("id") for c in group],
            "summary": response.content.strip(),
            "num_chunks": len(group),
        }

    summaries = list(await asyncio.gather(*[
        _summarize_group(i, g) for i, g in enumerate(groups)
    ]))

    return RaptorSummaryResponse(
        success=True,
        summaries=summaries,
        total_groups=len(groups),
        processing_time_seconds=time.time() - start_time,
    )


@inject_llm_chat_async
@inject_repo_async
async def _raptor_qa_logic(
    request: RaptorQARequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs,
) -> RaptorQAResponse:
    """Core logic for RAPTOR-based Q&A: retrieve + answer generation."""
    start_time = time.time()

    if llm is None:
        return RaptorQAResponse(
            success=False,
            answer="",
            error="LLM service is not available",
        )

    # Step 1: Retrieve context using RAPTOR
    retrieval_start = time.time()
    retriever = RaptorRetriever(repo=await get_raptor_repo())
    retrieval = await retriever.retrieve(
        request=request,
        llm=llm,
        embeddings=llm_embeddings,
        vector_repo=repo,
    )
    retrieval_time = time.time() - retrieval_start

    if not retrieval.success or not retrieval.results:
        return RaptorQAResponse(
            success=False,
            answer="",
            error=retrieval.error or "No relevant context found",
            strategy_used=retrieval.strategy_used,
        )

    # Step 2: Extract top-k chunks for answer generation
    top_chunks = retrieval.results[:request.top_k]
    context_text = "\n\n".join([
        f"[Level {r.get('level', '?')}] {r.get('content', '')}"
        for r in top_chunks
    ])

    # Step 3: Generate answer using LLM
    answer_start = time.time()
    from tilellm.modules.raptor.prompts import RAPTOR_QA_PROMPT

    prompt = RAPTOR_QA_PROMPT.format(
        context=context_text,
        question=request.question
    )

    try:
        response = await llm.ainvoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        answer = f"Could not generate answer: {str(e)}"

    answer_time = time.time() - answer_start

    return RaptorQAResponse(
        success=True,
        answer=answer,
        retrieved_chunks=top_chunks,
        strategy_used=retrieval.strategy_used,
        levels_searched=retrieval.levels_searched,
        total_chunks_retrieved=len(retrieval.results),
        processing_time_seconds=time.time() - start_time,
        retrieval_time_seconds=retrieval_time,
        answer_time_seconds=answer_time,
        traversal_path=retrieval.traversal_path,
    )


# ---------------------------------------------------------------------------
# Endpoint: Build RAPTOR tree
# ---------------------------------------------------------------------------

@router.post("/build", response_model=Union[AsyncTaskResponse, RaptorResponse])
async def build_raptor_tree(request: RaptorRequest):
    """
    Build RAPTOR hierarchical summary tree for a document.

    **Example:**
    ```bash
    curl -X POST http://localhost:8000/api/raptor/build \\
      -H "Content-Type: application/json" \\
      -d '{
        "namespace": "my-documents",
        "doc_id": "doc123",
        "engine": {"name": "qdrant", "index_name": "myindex", "deployment": "local"},
        "config": {"cluster_size": 5, "max_levels": 3}
      }'
    """
    if not is_raptor_enabled():
        raise HTTPException(status_code=400, detail="RAPTOR module is not enabled")
    try:
        # Use TaskIQ if enabled and available
        if ENABLE_TASKIQ and TASKIQ_AVAILABLE and task_raptor_build:
            payload = request.model_dump(mode='python')
            task = await task_raptor_build.kiq(payload)
            return AsyncTaskResponse(task_id=task.task_id)
        
        # Otherwise, execute synchronously
        return await _build_raptor_tree_logic(request)
    except Exception as e:
        logger.error(f"Error building RAPTOR tree: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoint: RAPTOR retrieval
# ---------------------------------------------------------------------------

@router.post("/retrieve", response_model=RaptorRetrievalResult)
async def retrieve_with_raptor(request: RaptorRetrievalRequest):
    """
    Retrieve from RAPTOR tree using specified strategy.

    **Strategies:**
    1. **collapsed_tree**: All nodes in same vector space (default, faster)
    2. **tree_traversal**: Agent decides which levels to search

    **Example:**
    ```bash
    curl -X POST http://localhost:8000/api/raptor/retrieve \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "What are the main themes?",
        "namespace": "my-documents",
        "engine": {"name": "qdrant", "index_name": "myindex", "deployment": "local"},
        "strategy": "collapsed_tree",
        "top_k": 5
      }'
    ```
    """
    if not is_raptor_enabled():
        raise HTTPException(status_code=400, detail="RAPTOR module is not enabled")
    try:
        return await _retrieve_raptor_logic(request)
    except Exception as e:
        logger.error(f"RAPTOR retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoint: RAPTOR Q&A (retrieve + answer generation)
# ---------------------------------------------------------------------------

@router.post("/qa", response_model=RaptorQAResponse)
async def raptor_qa(request: RaptorQARequest):
    """
    Complete RAPTOR Q&A pipeline: retrieve context from tree + generate answer.

    Combines document retrieval with LLM answer generation for comprehensive QA.

    **Strategies:**
    1. **collapsed_tree**: Fast retrieval across all levels
    2. **tree_traversal**: Smart level selection by agent

    **Example:**
    ```bash
    curl -X POST http://localhost:8000/api/raptor/qa \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "What are the main conclusions?",
        "namespace": "my-documents",
        "engine": {"name": "qdrant", "index_name": "myindex", "deployment": "local"},
        "strategy": "collapsed_tree",
        "top_k": 5,
        "llm": "openai",
        "model": "gpt-4o-mini",
        "gptkey": "sk-..."
      }'
    ```
    """
    if not is_raptor_enabled():
        raise HTTPException(status_code=400, detail="RAPTOR module is not enabled")
    try:
        return await _raptor_qa_logic(request)
    except Exception as e:
        logger.error(f"RAPTOR Q&A failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoint: Generate summaries only
# ---------------------------------------------------------------------------

@router.post("/summarize", response_model=RaptorSummaryResponse)
async def generate_summaries(request: RaptorSummaryRequest):
    """
    Generate summaries for a group of chunks (without building full tree).

    **Example:**
    ```bash
    curl -X POST http://localhost:8000/api/raptor/summarize \\
      -H "Content-Type: application/json" \\
      -d '{
        "namespace": "my-documents",
        "chunk_ids": ["chunk1", "chunk2", "chunk3"],
        "engine": {"name": "qdrant", "index_name": "myindex", "deployment": "local"},
        "cluster_size": 3
      }'
    ```
    """
    if not is_raptor_enabled():
        raise HTTPException(status_code=400, detail="RAPTOR module is not enabled")
    try:
        return await _summarize_logic(request)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoint: Get tree info
# ---------------------------------------------------------------------------

@router.get("/tree/{tree_id}")
async def get_tree_info(tree_id: str):
    """Get RAPTOR tree structure and statistics."""
    if not is_raptor_enabled():
        raise HTTPException(status_code=400, detail="RAPTOR module is not enabled")
    try:
        raptor_repo = await get_raptor_repo()
        tree = await raptor_repo.get_tree(tree_id)
        if not tree:
            raise HTTPException(status_code=404, detail=f"Tree {tree_id} not found")
        return {
            "tree_id": tree.tree_id,
            "namespace": tree.namespace,
            "doc_id": tree.doc_id,
            "total_nodes": tree.total_nodes,
            "levels": {str(k): v for k, v in tree.levels.items()},
            "root_ids": tree.root_ids,
            "leaf_ids": tree.leaf_ids,
            "config": tree.config.model_dump(),
            "created_at": tree.created_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tree info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoint: Delete tree
# ---------------------------------------------------------------------------

@router.delete("/tree/{tree_id}")
async def delete_tree(tree_id: str):
    """Delete RAPTOR tree and all associated summaries."""
    if not is_raptor_enabled():
        raise HTTPException(status_code=400, detail="RAPTOR module is not enabled")
    try:
        raptor_repo = await get_raptor_repo()
        success = await raptor_repo.delete_tree(tree_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Tree {tree_id} not found")
        return {"success": True, "message": f"Tree {tree_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting tree: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoint: List trees
# ---------------------------------------------------------------------------

@router.get("/trees/{namespace}")
async def list_trees(namespace: str):
    """List all RAPTOR trees in a namespace."""
    if not is_raptor_enabled():
        raise HTTPException(status_code=400, detail="RAPTOR module is not enabled")
    try:
        raptor_repo = await get_raptor_repo()
        tree_ids = await raptor_repo.list_trees(namespace)
        return {
            "namespace": namespace,
            "tree_count": len(tree_ids),
            "tree_ids": tree_ids,
        }
    except Exception as e:
        logger.error(f"Error listing trees: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_documents(result: Any) -> list:
    """Convert a RepositoryItems (or any list) to List[Document].

    ``get_by_doc_id`` returns a ``RepositoryItems`` Pydantic model whose
    ``matches`` field is a list of ``RepositoryQueryResult``.  RAPTOR
    service expects plain ``langchain_core.documents.Document`` objects.
    """
    if result is None:
        return []
    # Already a plain list (e.g. returned by some other repo method)
    if isinstance(result, list):
        return result
    # RepositoryItems wrapper
    matches = getattr(result, "matches", None)
    if matches is None:
        return []
    docs = []
    for match in matches:
        docs.append(Document(
            page_content=match.text or "",
            metadata={
                "id": match.metadata_id or "",
                "metadata_id": match.metadata_id or "",
                "source": match.metadata_source or "",
                "type": match.metadata_type or "",
            },
        ))
    return docs


async def _retrieve_document_chunks(
    namespace: str,
    doc_id: str,
    chunk_ids: Optional[list] = None,
    vector_repo: Any = None,
    engine: Any = None,
) -> list:
    """Retrieve all chunks for a document from vector store."""
    try:
        if chunk_ids:
            return await _retrieve_chunks_by_ids(
                namespace=namespace,
                chunk_ids=chunk_ids,
                vector_repo=vector_repo,
                doc_id=doc_id,
                engine=engine
            )
        else:
            if hasattr(vector_repo, 'get_by_doc_id'):
                raw = await vector_repo.get_by_doc_id(
                    engine=engine,
                    namespace=namespace,
                    doc_id=doc_id,
                )
                return _to_documents(raw)
            else:
                logger.warning(
                    f"vector_repo has no get_by_doc_id; cannot retrieve chunks for {doc_id}"
                )
                return []
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []


async def _retrieve_chunks_by_ids(
    namespace: str,
    chunk_ids: list,
    vector_repo: Any,
    doc_id: Optional[str] = None,
    engine: Any = None,
) -> list:
    """Retrieve specific chunks by ID using get_by_doc_id filter."""
    try:
        # If doc_id not provided but we have chunk_ids, try to extract it
        if not doc_id and chunk_ids:
            # Extract doc_id from the first chunk_id (format: doc_id#uuid)
            first_chunk_id = chunk_ids[0]
            if '#' in first_chunk_id:
                doc_id = first_chunk_id.split('#')[0]
            else:
                doc_id = first_chunk_id

        # If we have a doc_id and the vector repo supports get_by_doc_id, use it
        if doc_id and hasattr(vector_repo, 'get_by_doc_id'):
            logger.info(f"Retrieving chunks for doc_id={doc_id} via get_by_doc_id for efficient filtering")
            raw = await vector_repo.get_by_doc_id(
                engine=engine,
                namespace=namespace,
                doc_id=doc_id
            )
            all_doc_chunks = _to_documents(raw)
            # Map for efficient lookup
            chunk_map = {doc.metadata.get("id", ""): doc for doc in all_doc_chunks}
            # Return chunks in requested order; fall through if none matched
            matched = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
            if matched:
                return matched

        # Fallback to individual get_by_id if doc_id resolution fails or method not available
        logger.debug(f"Falling back to individual get_by_id for {len(chunk_ids)} chunks")
        chunks = []
        for chunk_id in chunk_ids:
            if hasattr(vector_repo, 'get_by_id'):
                chunk = await vector_repo.get_by_id(
                    namespace=namespace,
                    id=chunk_id,
                )
                if chunk:
                    chunks.append(chunk)
        return chunks
    except Exception as e:
        logger.error(f"Error retrieving chunks by IDs: {e}")
        return []


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """Check if RAPTOR module is enabled and healthy."""
    return {
        "enabled": is_raptor_enabled(),
        "status": "healthy" if is_raptor_enabled() else "disabled",
    }
