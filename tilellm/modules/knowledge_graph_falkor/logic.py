"""
Business Logic for Knowledge Graph Module.
Handles service initialization, dependency injection, and core operations.
Separated from controllers.py to keep routes clean.
"""

import logging
from typing import List, Dict, Any, Optional

from cleo.ui import question

from tilellm.models.schemas import RepositoryEngine
from tilellm.models.schemas.multimodal_content import TextContent
from tilellm.shared.utility import inject_repo_async, inject_llm_chat_async, inject_llm_async, get_service_config
from .models import Node, NodeUpdate, Relationship, RelationshipUpdate
from .models.schemas import (
    GraphQARequest, GraphCreateRequest, GraphQAAdvancedRequest, GraphClusterRequest, AddDocumentRequest
)
from .services import GraphService, GraphRAGService
from .repository import AsyncFalkorGraphRepository  # Use ASYNC repository

logger = logging.getLogger(__name__)


# Try to import Community Graph Service
try:
    from .services.community_graph_service import CommunityGraphService
    COMMUNITY_GRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Community Graph service not available: {e}")
    COMMUNITY_GRAPH_AVAILABLE = False
    CommunityGraphService = None

# New Services
from .services.multimodal_search import MultimodalPDFSearch
from .services.community_analyzer import DocumentCommunityAnalyzer
from .services.agentic_qa_service import AgenticQAService

# Global Service Instances (ASYNC)
repository: Optional[AsyncFalkorGraphRepository] = None
graph_service: Optional[GraphService] = None
graph_rag_service: Optional[GraphRAGService] = None
_initialization_attempted: bool = False
_initialization_lock = None  # Will be asyncio.Lock()

async def _get_lock():
    """Get or create the initialization lock"""
    global _initialization_lock
    if _initialization_lock is None:
        import asyncio
        _initialization_lock = asyncio.Lock()
    return _initialization_lock

async def initialize_services():
    """Initializes Graph services and connections (ASYNC)."""
    global repository, graph_service, graph_rag_service

    # Check if FalkorDB graphrag service is enabled in configuration
    try:
        service_config = get_service_config()
        graphrag_falkor_enabled = service_config.get("services", {}).get("graphrag_falkor", False) if service_config.get("services") else False
        if not graphrag_falkor_enabled:
            logger.info("FalkorDB Knowledge Graph service is disabled in configuration (graphrag_falkor: false).")
            repository = None
            graph_service = GraphService()
            graph_rag_service = GraphRAGService(graph_service=graph_service)
            logger.warning("FalkorDB Knowledge Graph services are DISABLED. API endpoints will fail if used.")
            return
    except Exception as e:
        logger.warning(f"Failed to read service configuration: {e}. Assuming FalkorDB graphrag is enabled.")

    try:
        repository = AsyncFalkorGraphRepository()
        # Use await for async verify_connection
        is_connected = await repository.verify_connection()
        if not is_connected:
            raise ConnectionError("Failed to verify connection to FalkorDB database.")

        graph_service = GraphService(repository=repository)
        graph_rag_service = GraphRAGService(graph_service=graph_service)
        logger.info("Successfully initialized and verified FalkorDB async connection and services.")

    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize FalkorDB connection: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        # Create dummy services that allow app to start but will fail on use
        repository = None
        graph_service = GraphService()
        graph_rag_service = GraphRAGService(graph_service=graph_service)
        logger.warning("Knowledge Graph services are NOT operational due to DB connection failure.")

async def ensure_initialized():
    """
    Lazy initialization - chiamato prima di ogni operazione.
    Inizializza FalkorDB solo se necessario e non ancora fatto.
    Thread-safe con lock.
    """
    global repository, graph_service, graph_rag_service, _initialization_attempted

    # Se già inizializzato, ritorna subito
    if repository is not None:
        return

    # Se già tentato e fallito, non ritentare continuamente
    if _initialization_attempted:
        if repository is None:
            raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")
        return

    # Lock per evitare inizializzazioni multiple concorrenti
    lock = await _get_lock()
    async with lock:
        # Double-check dopo aver acquisito il lock
        if repository is not None:
            return

        if _initialization_attempted:
            if repository is None:
                raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")
            return

        # Tenta l'inizializzazione
        await initialize_services()
        _initialization_attempted = True

        # Verifica se è riuscita
        if repository is None:
            raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")

# Note: Lazy initialization - initialize_services() verrà chiamato on-demand
# Non inizializzare qui per supportare app modulari

# ==================== HEALTH & STATS LOGIC ====================

def check_health():
    """Verify FalkorDB connection."""
    is_connected = graph_service.verify_connection()
    if is_connected:
        return {"status": "healthy", "database": "connected"}
    else:
        raise Exception("FalkorDB database connection failed")

def get_stats():
    """Get database statistics."""
    return graph_service.get_database_stats()

async def get_graph_network(
    namespace: Optional[str] = None,
    index_name: Optional[str] = None,
    node_limit: int = 1000,
    relationship_limit: int = 5000,
    node_labels: Optional[List[str]] = None,
    community: bool = False
) -> Dict[str, Any]:
    """
    Get graph network (nodes + relationships) for visualization.

    Args:
        namespace: Optional namespace to filter
        index_name: Optional index_name to filter
        node_limit: Maximum nodes to return
        relationship_limit: Maximum relationships to return
        node_labels: Optional list of node labels to filter
        community: If True, returns community BELONGS_TO relationships

    Returns:
        Dictionary with nodes, relationships, and stats
    """
    return await graph_service.get_graph_network(
        namespace=namespace,
        index_name=index_name,
        node_limit=node_limit,
        relationship_limit=relationship_limit,
        node_labels=node_labels,
        community=community
    )

# ==================== NODE & RELATIONSHIP LOGIC ====================

async def create_node(node: Node) -> Node:
    return await graph_service.create_node(node)

async def get_node(node_id: str) -> Optional[Node]:
    return await graph_service.get_node(node_id)

async def get_nodes_by_label(label: str, limit: int) -> List[Node]:
    return await graph_service.get_nodes_by_label(label, limit)

async def search_nodes(label: str, property_key: str, property_value: str, limit: int) -> List[Node]:
    return await graph_service.search_nodes(label, property_key, property_value, limit)

async def update_node(node_id: str, node_update: NodeUpdate) -> Optional[Node]:
    return await graph_service.update_node(node_id, node_update)

async def delete_node(node_id: str, detach: bool) -> bool:
    return await graph_service.delete_node(node_id, detach)

# ==================== RELATIONSHIP ====================

async def create_relationship(relationship: Relationship) -> Relationship:
    return await graph_service.create_relationship(relationship)

async def get_relationship(relationship_id: str) -> Optional[Relationship]:
    return await graph_service.get_relationship(relationship_id)

async def get_node_relationships(node_id: str, direction: str) -> List[Relationship]:
    return await graph_service.get_node_relationships(node_id, direction)

async def update_relationship(relationship_id: str, relationship_update: RelationshipUpdate) -> Optional[Relationship]:
    return await graph_service.update_relationship(relationship_id, relationship_update)

async def delete_relationship(relationship_id: str) -> bool:
    return await graph_service.delete_relationship(relationship_id)

# ==================== ADVANCED LOGIC (DI WRAPPERS) ====================

@inject_llm_chat_async
@inject_repo_async
async def add_document_to_graph(
    request: AddDocumentRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Add all chunks of a document to the knowledge graph by retrieving them from vector store.

    This enables incremental graph updates when a new document is added to the knowledge base,
    without regenerating the entire namespace graph.
    """
    # Lazy initialization
    await ensure_initialized()

    if llm is None:
        raise ValueError("LLM configuration is required for entity extraction.")

    if repo is None:
        raise RuntimeError("Vector store repository not injected")

    if request.engine is None:
        raise ValueError("Engine configuration is required for vector store access")

    # Call GraphRAG service to add document
    result = await graph_rag_service.add_document_to_graph(
        metadata_id=request.metadata_id,
        namespace=request.namespace,
        engine=request.engine,
        vector_store_repo=repo,
        llm=llm,
        deduplicate_entities=request.deduplicate_entities if request.deduplicate_entities is not None else True
    )
    
    # If document added successfully, update community reports
    if (result.get("status") == "success" and 
        COMMUNITY_GRAPH_AVAILABLE and 
        CommunityGraphService is not None and
        repository is not None):
        try:
            community_service = CommunityGraphService(
                graph_service=graph_service,
                graph_rag_service=graph_rag_service
            )
            # Update community reports (overwrite existing)
            report_stats = await community_service.generate_hierarchical_reports(
                namespace=request.namespace,
                index_name=request.engine.index_name if hasattr(request.engine, 'index_name') else None,
                sparse_encoder=request.sparse_encoder,
                llm=llm,
                vector_store_repo=repo,
                llm_embeddings=llm_embeddings,  # Could be injected separately
                engine=request.engine,
                overwrite=True
            )
            result["community_reports_updated"] = True
            result["report_stats"] = report_stats
        except Exception as e:
            logger.warning(f"Failed to update community reports after adding document: {e}")
            result["community_reports_updated"] = False
            result["report_error"] = str(e)
    
    return result


# ==================== COMMUNITY GRAPH LOGIC ====================

@inject_llm_chat_async
@inject_repo_async
async def create_graph(
    request: GraphCreateRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    """Create community graph using existing GraphRAG extraction and clustering."""
    # Lazy initialization
    await ensure_initialized()

    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")

    if repository is None:
        raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")
    
    if repo is None:
        raise RuntimeError("Vector store repository not injected")
    
    if request.engine is None:
        raise ValueError("Engine configuration is required for vector store access")
    
    community_service = CommunityGraphService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service
    )
    
    return await community_service.create_community_graph(
        namespace=request.namespace,
        engine=request.engine,
        creation_prompt=request.creation_prompt,
        vector_store_repo=repo,
        llm=llm,
        llm_embeddings=llm_embeddings,
        sparse_encoder=request.sparse_encoder,
        limit=request.limit or 100,
        index_name=request.engine.index_name,
        overwrite=request.overwrite or False,
        import_to_graph=True
    )

@inject_llm_async
@inject_repo_async
async def cluster_graph_louvain(
    request: GraphClusterRequest,
    repo=None,
    chat_model=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    """Cluster graph and generate reports (Microsoft GraphRAG style)."""
    await ensure_initialized()
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
    
    if repository is None:
        raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")
        
    if chat_model is None:
        raise ValueError("LLM configuration is required for community report generation.")
    
    if not hasattr(request, 'engine') or request.engine is None:
         raise ValueError("Engine configuration is required on the request for report indexing.")
        
    community_service = CommunityGraphService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service
    )
    
    return await community_service.generate_community_reports(
        namespace=request.namespace,
        index_name=request.engine.index_name if hasattr(request.engine, "index_name") else None,
        llm=chat_model,
        vector_store_repo=repo,
        llm_embeddings=llm_embeddings,
        engine=request.engine,
        overwrite=request.overwrite if request.overwrite is not None else True
    )

@inject_llm_async
@inject_repo_async
async def cluster_graph_leiden(
    request: GraphClusterRequest,
    repo=None,
    chat_model=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    await ensure_initialized()
    """Cluster graph using Leiden algorithm."""
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
    
    if repository is None:
        raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")
        
    if chat_model is None:
        raise ValueError("LLM configuration is required for community report generation.")

    if not hasattr(request, 'engine') or request.engine is None:
         raise ValueError("Engine configuration is required on the request for report indexing.")
        
    community_service = CommunityGraphService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service
    )
    
    return await community_service.generate_community_reports_leiden(
        namespace=request.namespace,
        index_name=request.engine.index_name if hasattr(request.engine, "index_name") else None,
        llm=chat_model,
        vector_store_repo=repo,
        llm_embeddings=llm_embeddings,
        engine=request.engine,
        overwrite=request.overwrite if request.overwrite is not None else True
    )

@inject_llm_chat_async
@inject_repo_async
async def cluster_graph_hierarchical(
    request: GraphClusterRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    await ensure_initialized()
    """Cluster graph hierarchically (Levels 0, 1, 2)."""
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
    
    if repository is None:
        raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")
        
    if llm is None:
        raise ValueError("LLM configuration is required for community report generation.")
    
    if not hasattr(request, 'engine') or request.engine is None:
         raise ValueError("Engine configuration is required on the request for report indexing.")

    community_service = CommunityGraphService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service
    )
    
    return await community_service.generate_hierarchical_reports(
        namespace=request.namespace,
        index_name=request.engine.index_name if hasattr(request.engine, 'index_name') else None,
        sparse_encoder=request.sparse_encoder,
        llm=llm,
        vector_store_repo=repo,
        llm_embeddings=llm_embeddings,
        engine=request.engine,
        overwrite=request.overwrite if request.overwrite is not None else True
    )

@inject_llm_chat_async
@inject_repo_async
async def query_graph(
    request: GraphQARequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    embedding_dimension=None,
    **kwargs
) -> Dict[str, Any]:
    await ensure_initialized()
    """Query community graph using global search."""
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
    
    if repository is None:
        raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")
    
    if not hasattr(request, 'engine') or request.engine is None:
        raise ValueError("Engine configuration is required for semantic report search.")

    community_service = CommunityGraphService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service
    )
    
    question_text = ""
    if isinstance(request.question, str):
        question_text = request.question
    elif isinstance(request.question, list):
        for content in request.question:
            if isinstance(content, TextContent):
                question_text = content.text
                break
    
    if not question_text:
        raise ValueError("No valid question text provided.")
    
    return await community_service.query_with_global_search(
        question=question_text,
        namespace=request.namespace,
        sparse_encoder=request.sparse_encoder,
        llm=llm,
        vector_store_repo=repo,
        llm_embeddings=llm_embeddings,
        engine=request.engine,
        chat_history_dict=request.chat_history_dict,
        reranking_config=request.reranker_config,  # Pass reranking config
        use_reranking=True,  # Enable reranking by default
        top_k_initial=20,
        top_k_reranked=5
    )

@inject_llm_chat_async
@inject_repo_async
async def context_fusion_graph_search(
    request: GraphQAAdvancedRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    embedding_dimension=None,
    **kwargs
) -> Dict[str, Any]:
    await ensure_initialized()
    """Hybrid Search with Context Fusion (Local + Global)."""
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
    
    if repository is None:
        raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")
    
    if request.engine is None:
         raise ValueError("Engine configuration is required for vector store access")
          
    community_service = CommunityGraphService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service
    )
    
    question_text = request.question if isinstance(request.question, str) else request.question[0].text

    #if isinstance(request.reranking, bool):  # True
    #    reranking_model_config = request.reranker_model
    #else:
    #    reranking_model_config = request.reranking
    return await community_service.context_fusion_search(
        question=question_text,
        namespace=request.namespace,
        search_type=request.search_type,
        sparse_encoder_injected=request.sparse_encoder,
        reranking_injected=request.reranker_config,
        engine=request.engine,
        vector_store_repo=repo,
        max_results=request.top_k if request.top_k else 15,
        llm=llm,
        llm_embeddings=llm_embeddings,
        query_type=request.query_type,
        chat_history_dict=request.chat_history_dict
    )

# ==================== MULTIMODAL & ANALYSIS LOGIC ====================

@inject_llm_chat_async
@inject_repo_async
async def multimodal_search(
    request: GraphQAAdvancedRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    await ensure_initialized()
    """
    Perform multimodal search (Text + Table + Image).
    """
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
    
    if repository is None:
        raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")

    if request.engine is None:
         raise ValueError("Engine configuration is required")

    community_service = CommunityGraphService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service
    )
    
    search_service = MultimodalPDFSearch(community_service=community_service, llm=llm)
    
    question_text = request.question if isinstance(request.question, str) else request.question[0].text
    
    # Determine search types based on request or default
    search_types = ["text"] # Default
    # You might want to add a field to GraphQAAdvancedRequest for search_types
    
    return await search_service.search(
        query=question_text,
        namespace=request.namespace,
        engine=request.engine,
        vector_store_repo=repo,
        llm=llm,
        llm_embeddings=llm_embeddings,
        search_types=search_types
    )

@inject_repo_async
async def analyze_community(
    request: GraphClusterRequest,
    repo=None,
    **kwargs
) -> Dict[str, Any]:
    await ensure_initialized()
    """
    Analyze document collection to find communities.
    """
    if not COMMUNITY_GRAPH_AVAILABLE:
        raise RuntimeError("Community Graph service not available")
        
    # We need a ClusterService instance
    from .services.clustering import ClusterService
    
    # Access the graph repository from the global graph_service
    graph_repo = graph_service._get_repository() if graph_service else None
    
    # We need an LLM for the ClusterService (it uses it for summaries)
    # We can inject it or get it from GraphRAGService
    llm = graph_rag_service.llm if graph_rag_service else None
    
    if not graph_repo or not llm:
         raise RuntimeError("Graph dependencies not ready")

    cluster_service = ClusterService(repository=graph_repo, llm=llm)
    analyzer = DocumentCommunityAnalyzer(cluster_service=cluster_service)
    
    return await analyzer.analyze_collection(
        namespace=request.namespace,
        engine=request.engine
    )

# ==================== ADVANCED QA LOGIC ====================

@inject_llm_chat_async
@inject_repo_async
async def advanced_qa_search(
    request: GraphQAAdvancedRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    await ensure_initialized()
    """
    Advanced QA Search with intent classification and domain-specific Cypher queries.
    Specialized for debt collection (recupero crediti) domain.

    Pipeline:
    1. Intent Classifier → identifies query type (timeline, exposure, guarantees, etc.)
    2. Parameter Extractor → extracts entities (debtor_name, loan_id, etc.)
    3. Find seed nodes from vector store
    4. Get relevant community reports (max 3)
    5. Template Cypher Engine → generates safe parameterized queries
    6. Graph Execution → runs queries on FalkorDB
    7. LLM Response Enricher → transforms results into readable response
    """
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")

    if repository is None:
        raise RuntimeError("Graph repository not initialized. FalkorDB service may be disabled or failed to connect.")

    if request.engine is None:
        raise ValueError("Engine configuration is required for vector store access")

    if llm is None:
        raise ValueError("LLM is required for advanced QA processing")

    # Import Advanced QA Service
    from .services.advanced_qa_service import AdvancedQAService

    # Initialize services
    community_service = CommunityGraphService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service
    )

    advanced_qa_service = AdvancedQAService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service,
        community_graph_service=community_service,
        llm=llm
    )

    # Extract question text
    question_text = request.question if isinstance(request.question, str) else request.question[0].text

    # Process query through advanced pipeline
    return await advanced_qa_service.process_query(
        question=question_text,
        namespace=request.namespace,
        engine=request.engine,
        vector_store_repo=repo,
        llm_embeddings=llm_embeddings,
        search_type=request.search_type if hasattr(request, 'search_type') else "hybrid",
        sparse_encoder=request.sparse_encoder,
        chat_history_dict=request.chat_history_dict,
        max_community_reports=3,  # Always max 3 reports as requested
        top_k=request.top_k if request.top_k else 10
    )


@inject_llm_chat_async
@inject_repo_async
async def agentic_qa_search(
    request: GraphQAAdvancedRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    await ensure_initialized()
    """
    Agentic QA Search using "The Graph Specialist".
    """
    if AgenticQAService is None:
        raise RuntimeError("Agentic QA service not available")

    if repository is None:
        raise RuntimeError("Graph repository not initialized")

    if llm is None:
        raise ValueError("LLM is required for agentic QA")

    agent_service = AgenticQAService(
        graph_service=graph_service,
        llm=llm
    )

    question_text = request.question if isinstance(request.question, str) else request.question[0].text

    return await agent_service.process_query(
        question=question_text,
        namespace=request.namespace,
        chat_history_dict=request.chat_history_dict,
        creation_prompt=getattr(request, 'creation_prompt', None)
    )
