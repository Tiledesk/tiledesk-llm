"""
RAPTOR Retrieval Strategies.

Implements two retrieval approaches:
1. Collapsed Tree: All nodes in same vector space (simpler, faster)
2. Tree Traversal: Agent decides which levels to search (dynamic, intelligent)
"""

import logging
import time
from typing import List, Dict, Any

from tilellm.modules.raptor.models.models import (
    RaptorRetrievalStrategy,
    RaptorRetrievalRequest,
    RaptorRetrievalResult,
    RaptorTraversalDecision,
)
from tilellm.modules.raptor.repository import RaptorRepository
from tilellm.modules.raptor.prompts import (
    RAPTOR_TRAVERSAL_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def _build_raptor_question_answer(request: RaptorRetrievalRequest, raptor_namespace: str):
    """
    Build a QuestionAnswer object for RAPTOR namespace retrieval.

    Uses the engine/credentials from the request, but with the RAPTOR namespace
    so get_chunks_from_repo retrieves from {namespace}-raptor.
    """
    from tilellm.models import QuestionAnswer
    return QuestionAnswer(
        question=request.question,
        namespace=raptor_namespace,
        engine=request.engine,
        gptkey=request.gptkey,
        model=request.model,
        llm=request.llm,
        embedding=request.embedding,
        sparse_encoder=request.sparse_encoder,
        temperature=request.temperature,
        top_k=request.top_k,
        debug=request.debug,
    )


class RaptorRetriever:
    """
    Retriever for RAPTOR hierarchical trees.

    Supports both collapsed tree and tree traversal strategies.
    """

    def __init__(self, repo: RaptorRepository):
        self.repo = repo

    async def retrieve(
        self,
        request: RaptorRetrievalRequest,
        llm: Any,
        embeddings: Any,
        vector_repo: Any,
    ) -> RaptorRetrievalResult:
        """Retrieve from RAPTOR tree using specified strategy."""
        start_time = time.time()

        try:
            if request.strategy == RaptorRetrievalStrategy.COLLAPSED_TREE:
                return await self._retrieve_collapsed_tree(
                    request=request,
                    embeddings=embeddings,
                    vector_repo=vector_repo,
                )
            elif request.strategy == RaptorRetrievalStrategy.TREE_TRAVERSAL:
                return await self._retrieve_tree_traversal(
                    request=request,
                    llm=llm,
                    embeddings=embeddings,
                    vector_repo=vector_repo,
                )
            else:
                return RaptorRetrievalResult(
                    success=False,
                    error=f"Unknown strategy: {request.strategy}",
                    strategy_used=str(request.strategy),
                )

        except Exception as e:
            logger.error(f"RAPTOR retrieval failed: {e}", exc_info=True)
            return RaptorRetrievalResult(
                success=False,
                error=str(e),
                strategy_used=str(request.strategy),
                processing_time_seconds=time.time() - start_time,
            )

    async def _retrieve_collapsed_tree(
        self,
        request: RaptorRetrievalRequest,
        embeddings: Any,
        vector_repo: Any,
    ) -> RaptorRetrievalResult:
        """
        Retrieve from collapsed tree (all levels in same vector space).

        Uses SAME vector store as application with namespace: {namespace}-raptor
        """
        start_time = time.time()
        try:
            raptor_namespace = self.repo.get_raptor_namespace(request.namespace)

            # Build a QuestionAnswer for the RAPTOR namespace and call get_chunks_from_repo
            qa = _build_raptor_question_answer(request, raptor_namespace)
            # Note: decorator @inject_embedding_qa_async_optimized() injects embeddings automatically
            retrieval = await vector_repo.get_chunks_from_repo(qa)

            # Format results with rank-based scores
            results = []
            n = len(retrieval.chunks)
            for i, (content, metadata) in enumerate(
                zip(retrieval.chunks, retrieval.metadata)
            ):
                # Skip if doc_id filter is set and doesn't match
                if request.doc_id and metadata.get("doc_id") != request.doc_id:
                    continue
                results.append({
                    "content": content,
                    "score": float(n - i) / n,  # rank-based score
                    "metadata": metadata,
                    "level": metadata.get("level", 0),
                    "is_summary": metadata.get("is_summary", False),
                    "node_id": metadata.get("node_id"),
                })

            levels_searched = sorted(set(r["level"] for r in results))

            return RaptorRetrievalResult(
                success=True,
                results=results[:request.top_k],
                strategy_used="collapsed_tree",
                levels_searched=levels_searched,
                total_results=len(results),
                processing_time_seconds=time.time() - start_time,
            )

        except ValueError as e:
            # get_chunks_from_repo raises ValueError when no chunks found
            logger.info(f"No RAPTOR nodes found: {e}")
            return RaptorRetrievalResult(
                success=True,
                results=[],
                strategy_used="collapsed_tree",
                levels_searched=[],
                total_results=0,
                processing_time_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Collapsed tree retrieval failed: {e}", exc_info=True)
            return RaptorRetrievalResult(
                success=False,
                error=str(e),
                strategy_used="collapsed_tree",
                processing_time_seconds=time.time() - start_time,
            )

    async def _retrieve_tree_traversal(
        self,
        request: RaptorRetrievalRequest,
        llm: Any,
        embeddings: Any,
        vector_repo: Any,
    ) -> RaptorRetrievalResult:
        """
        Retrieve using tree traversal (agent decides which levels to search).
        """
        start_time = time.time()
        try:
            # Get tree for document (if specified)
            tree = None
            if request.doc_id:
                tree = await self.repo.get_tree_by_doc_id(
                    doc_id=request.doc_id,
                    namespace=request.namespace,
                )

            available_levels = [0, 1, 2, 3]
            if tree:
                available_levels = list(tree.levels.keys())

            traversal_path = []
            all_results = []
            current_level = 2  # Start at higher level for overview
            iteration = 0
            # Cap max iterations to avoid excessive LLM calls (each iteration calls LLM)
            # Default to 6, but never exceed top_k
            max_iterations = min(request.top_k, 6)

            structured_llm = llm.with_structured_output(RaptorTraversalDecision)

            while iteration < max_iterations:
                prompt = RAPTOR_TRAVERSAL_SYSTEM_PROMPT.format(
                    question=request.question,
                    current_level=current_level,
                    available_levels=available_levels,
                )

                decision: RaptorTraversalDecision = await structured_llm.ainvoke(prompt)

                traversal_path.append({
                    "iteration": iteration,
                    "level": current_level,
                    "action": decision.action,
                    "reasoning": decision.reasoning,
                })

                if decision.action == "stop":
                    break
                elif decision.action == "search_this_level":
                    level_results = await self._search_level(
                        level=current_level,
                        request=request,
                        embeddings=embeddings,
                        vector_repo=vector_repo,
                    )
                    all_results.extend(level_results)
                elif decision.action == "go_deeper":
                    next_level = current_level - 1
                    if next_level >= 0 and next_level in available_levels:
                        current_level = next_level
                    else:
                        break
                elif decision.action == "go_higher":
                    next_level = current_level + 1
                    if next_level <= max(available_levels) and next_level in available_levels:
                        current_level = next_level
                    else:
                        break

                iteration += 1
                if len(all_results) >= request.top_k:
                    break

            unique_results = self._deduplicate_results(all_results)
            unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            levels_searched = sorted(set(r.get("level", 0) for r in unique_results))

            return RaptorRetrievalResult(
                success=True,
                results=unique_results[:request.top_k],
                strategy_used="tree_traversal",
                levels_searched=levels_searched,
                traversal_path=traversal_path,
                total_results=len(unique_results),
                processing_time_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Tree traversal retrieval failed: {e}", exc_info=True)
            return RaptorRetrievalResult(
                success=False,
                error=str(e),
                strategy_used="tree_traversal",
                processing_time_seconds=time.time() - start_time,
            )

    async def _search_level(
        self,
        level: int,
        request: RaptorRetrievalRequest,
        embeddings: Any,
        vector_repo: Any,
    ) -> List[Dict[str, Any]]:
        """
        Search at a specific level by retrieving from the RAPTOR namespace
        and post-filtering by level metadata.
        """
        try:
            raptor_namespace = self.repo.get_raptor_namespace(request.namespace)
            # Over-fetch so we have enough after post-filtering by level
            over_fetch_request = request.model_copy(
                update={"top_k": request.top_k_per_level * 4}
            )
            qa = _build_raptor_question_answer(over_fetch_request, raptor_namespace)

            # Note: decorator @inject_embedding_qa_async_optimized() injects embeddings automatically
            retrieval = await vector_repo.get_chunks_from_repo(qa)

            results = []
            n = len(retrieval.chunks)
            for i, (content, metadata) in enumerate(
                zip(retrieval.chunks, retrieval.metadata)
            ):
                if metadata.get("level", 0) != level:
                    continue
                if request.doc_id and metadata.get("doc_id") != request.doc_id:
                    continue
                results.append({
                    "content": content,
                    "score": float(n - i) / n,
                    "metadata": metadata,
                    "level": level,
                    "is_summary": metadata.get("is_summary", False),
                    "node_id": metadata.get("node_id"),
                })

            return results[:request.top_k_per_level]

        except Exception as e:
            logger.error(f"Error searching level {level}: {e}")
            return []

    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Remove duplicate results based on node_id."""
        seen_ids = set()
        unique_results = []
        for result in results:
            node_id = result.get("node_id")
            if node_id and node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_results.append(result)
        return unique_results
