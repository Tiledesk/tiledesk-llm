"""
RAPTOR integration with the main orchestrator.

This module provides the summary_agent node that can be optionally
added to the main RAG workflow for documents that benefit from
hierarchical summarization.
"""

import logging
from typing import Optional, Dict, Any

from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async
from tilellm.modules.raptor.config_loader import (
    is_raptor_enabled,
    should_use_raptor_for_document,
    get_raptor_config_from_env,
)
from tilellm.modules.raptor.services.raptor_service import RaptorService
from tilellm.modules.raptor.services.retrieval_strategies import RaptorRetriever
from tilellm.modules.raptor.repository import RaptorRepository
from tilellm.modules.raptor.models.models import (
    RaptorRetrievalRequest,
    RaptorRetrievalStrategy,
)

logger = logging.getLogger(__name__)


@inject_llm_chat_async
@inject_repo_async
async def raptor_summary_node(
    state: Dict[str, Any],
    llm=None,
    llm_embeddings=None,
    repo=None,
) -> Dict[str, Any]:
    """
    RAPTOR summary agent node for the main orchestrator workflow.
    
    This node can be conditionally added to the RAG workflow when:
    - Document type is "accademico", "tecnico", "legale", or "scientifico"
    - Document has > N pages (default: 10)
    - RAPTOR module is enabled
    
    The node:
    1. Checks if RAPTOR should be used for this document
    2. If tree exists, uses it for retrieval
    3. If tree doesn't exist but should be created, builds it
    4. Returns enhanced retrieval results
    
    Args:
        state: Graph state containing question_answer and other context
        llm: Injected LLM
        llm_embeddings: Injected embeddings
        repo: Injected vector store repository
        
    Returns:
        Updated state with RAPTOR retrieval results
    """
    if not is_raptor_enabled():
        logger.debug("RAPTOR module is disabled, skipping")
        return {}
    
    try:
        question_answer = state.get("question_answer")
        if not question_answer:
            logger.warning("No question_answer in state, skipping RAPTOR")
            return {}
        
        # Extract document metadata
        metadata = state.get("metadata", {})
        doc_type = metadata.get("doc_type")
        page_count = metadata.get("page_count")
        doc_id = metadata.get("doc_id") or getattr(question_answer, 'namespace', None)
        namespace = getattr(question_answer, 'namespace', 'default')
        
        # Check if RAPTOR should be used
        if not should_use_raptor_for_document(doc_type, page_count):
            logger.info(
                f"RAPTOR not activated for doc {doc_id} "
                f"(type={doc_type}, pages={page_count})"
            )
            return {}
        
        logger.info(f"RAPTOR activated for document {doc_id}")
        
        # Get RAPTOR repository
        raptor_repo = RaptorRepository()
        
        # Check if tree already exists
        tree = await raptor_repo.get_tree_by_doc_id(doc_id, namespace)
        
        if not tree:
            # Build RAPTOR tree
            logger.info(f"Building RAPTOR tree for document {doc_id}")
            
            # Retrieve chunks
            chunks = await _retrieve_document_chunks(
                namespace=namespace,
                doc_id=doc_id,
                vector_repo=repo,
                engine=question_answer.engine,
            )
            
            if chunks:
                # Get configuration
                config = get_raptor_config_from_env()
                
                # Build tree
                service = RaptorService(repo=raptor_repo)
                build_response = await service.build_raptor_tree(
                    chunks=chunks,
                    namespace=namespace,
                    doc_id=doc_id,
                    llm=llm,
                    embeddings=llm_embeddings,
                    vector_repo=repo,
                    engine=question_answer.engine,
                    config=config,
                )
                
                if build_response.success:
                    logger.info(
                        f"RAPTOR tree built: {build_response.tree_id}, "
                        f"{build_response.total_summaries} summaries"
                    )
                else:
                    logger.warning(
                        f"RAPTOR tree build failed: {build_response.error}"
                    )
                    return {}
            else:
                logger.warning(f"No chunks found for document {doc_id}")
                return {}
        
        # Retrieve using RAPTOR
        retrieval_strategy = RaptorRetrievalStrategy.COLLAPSED_TREE  # Default
        
        retrieval_request = RaptorRetrievalRequest(
            question=getattr(question_answer, 'question', ''),
            namespace=namespace,
            doc_id=doc_id,
            strategy=retrieval_strategy,
            top_k=getattr(question_answer, 'top_k', 5),
        )
        
        retriever = RaptorRetriever(repo=raptor_repo)
        retrieval_result = await retriever.retrieve(
            request=retrieval_request,
            llm=llm,
            embeddings=llm_embeddings,
            vector_repo=repo,
        )
        
        if retrieval_result.success:
            logger.info(
                f"RAPTOR retrieval successful: {retrieval_result.total_results} results"
            )
            
            # Add RAPTOR results to state
            # These can be used to enhance the standard RAG retrieval
            return {
                "raptor_results": retrieval_result.results,
                "raptor_strategy": retrieval_result.strategy_used,
                "metadata": {
                    **metadata,
                    "raptor_used": True,
                    "raptor_levels_searched": retrieval_result.levels_searched,
                }
            }
        else:
            logger.warning(f"RAPTOR retrieval failed: {retrieval_result.error}")
            return {}
        
    except Exception as e:
        logger.error(f"RAPTOR summary node failed: {e}", exc_info=True)
        return {}


async def _retrieve_document_chunks(
    namespace: str,
    doc_id: str,
    vector_repo: Any,
    engine: Any = None,
) -> list:
    """
    Retrieve all chunks for a document.
    
    Args:
        namespace: Namespace
        doc_id: Document ID
        vector_repo: Vector store repository
        engine: Vector store engine
        
    Returns:
        List of chunk documents
    """
    try:
        # Try to get chunks by filtering on doc_id
        if hasattr(vector_repo, 'get_by_doc_id'):
            return await vector_repo.get_by_doc_id(
                engine=engine,
                namespace=namespace,
                doc_id=doc_id,
            )
        else:
            logger.warning(
                f"vector_repo has no get_by_doc_id; cannot retrieve chunks for {doc_id}"
            )
            return []
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []


def should_activate_raptor(
    doc_type: Optional[str],
    page_count: Optional[int],
) -> bool:
    """
    Determine if RAPTOR should be activated for a document.
    
    Args:
        doc_type: Document type
        page_count: Number of pages
        
    Returns:
        True if RAPTOR should be activated
    """
    return should_use_raptor_for_document(doc_type, page_count)
