"""
Business Logic for Knowledge Graph Module.
Handles service initialization, dependency injection, and core operations.
Separated from controllers.py to keep routes clean.
"""

import logging
from typing import List, Dict, Any, Optional

from tilellm.models.schemas import RepositoryEngine
from tilellm.models.schemas.multimodal_content import TextContent
from tilellm.shared.utility import inject_repo_async, inject_llm_chat_async, inject_llm_async, get_service_config
from .models import Node, NodeUpdate, Relationship, RelationshipUpdate
from .models.schemas import (
    GraphQARequest, GraphCreateRequest, GraphQAAdvancedRequest, GraphClusterRequest, AddDocumentRequest
)
from .services import GraphService, GraphRAGService
from tilellm.modules.knowledge_graph.repository.repository import GraphRepository

logger = logging.getLogger(__name__)


# Try to import Community Graph Service
try:
    from .services.community_graph_service import CommunityGraphService
    COMMUNITY_GRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Community Graph service not available: {e}")
    COMMUNITY_GRAPH_AVAILABLE = False
    CommunityGraphService = None

# Global Service Instances
repository: Optional[GraphRepository] = None
graph_service: Optional[GraphService] = None
graph_rag_service: Optional[GraphRAGService] = None

def initialize_services():
    """Initializes Graph services and connections."""
    global repository, graph_service, graph_rag_service
    
    # Check if graphrag service is enabled in configuration
    try:
        service_config = get_service_config()
        graphrag_enabled = service_config.get("services", {}).get("graphrag", False)
        if not graphrag_enabled:
            logger.info("Knowledge Graph service is disabled in configuration (graphrag: false).")
            repository = None
            graph_service = GraphService()
            graph_rag_service = GraphRAGService(graph_service=graph_service)
            logger.warning("Knowledge Graph services are DISABLED. API endpoints will fail if used.")
            return
    except Exception as e:
        logger.warning(f"Failed to read service configuration: {e}. Assuming graphrag is enabled.")
    
    try:
        repository = GraphRepository()
        if not repository.verify_connection():
            raise ConnectionError("Failed to verify connection to Neo4j database.")
        
        graph_service = GraphService(repository=repository)
        graph_rag_service = GraphRAGService(graph_service=graph_service)
        logger.info("Successfully initialized and verified Neo4j connection and services.")

    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize Neo4j connection: {e}")
        # Create dummy services that allow app to start but will fail on use
        repository = None
        graph_service = GraphService()
        graph_rag_service = GraphRAGService(graph_service=graph_service)
        logger.warning("Knowledge Graph services are NOT operational due to DB connection failure.")

# Initialize on module load
initialize_services()

# ==================== HEALTH & STATS LOGIC ====================

def check_health():
    """Verify Neo4j connection."""
    is_connected = graph_service.verify_connection()
    if is_connected:
        return {"status": "healthy", "database": "connected"}
    else:
        raise Exception("Neo4j database connection failed")

def get_stats():
    """Get database statistics."""
    return graph_service.get_database_stats()

def get_graph_network(
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
    return graph_service.get_graph_network(
        namespace=namespace,
        index_name=index_name,
        node_limit=node_limit,
        relationship_limit=relationship_limit,
        node_labels=node_labels,
        community=community
    )

# ==================== NODE & RELATIONSHIP LOGIC ====================

def create_node(node: Node) -> Node:
    return graph_service.create_node(node)

def get_node(node_id: str) -> Optional[Node]:
    return graph_service.get_node(node_id)

def get_nodes_by_label(label: str, limit: int) -> List[Node]:
    return graph_service.get_nodes_by_label(label, limit)

def search_nodes(label: str, property_key: str, property_value: str, limit: int) -> List[Node]:
    return graph_service.search_nodes(label, property_key, property_value, limit)

def update_node(node_id: str, node_update: NodeUpdate) -> Optional[Node]:
    return graph_service.update_node(node_id, node_update)

def delete_node(node_id: str, detach: bool) -> bool:
    return graph_service.delete_node(node_id, detach)

def create_relationship(relationship: Relationship) -> Relationship:
    return graph_service.create_relationship(relationship)

def get_relationship(relationship_id: str) -> Optional[Relationship]:
    return graph_service.get_relationship(relationship_id)

def get_node_relationships(node_id: str, direction: str) -> List[Relationship]:
    return graph_service.get_node_relationships(node_id, direction)

def update_relationship(relationship_id: str, relationship_update: RelationshipUpdate) -> Optional[Relationship]:
    return graph_service.update_relationship(relationship_id, relationship_update)

def delete_relationship(relationship_id: str) -> bool:
    return graph_service.delete_relationship(relationship_id)

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
    if result.get("status") == "success" and COMMUNITY_GRAPH_AVAILABLE and CommunityGraphService is not None:
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
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
    
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
        vector_store_repo=repo,
        llm=llm,
        llm_embeddings=llm_embeddings,
        sparse_encoder=request.sparse_encoder,
        limit=request.limit or 100,
        index_name=request.index_name,
        overwrite=request.overwrite or False,
        import_to_neo4j=True
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
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
        
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
    """Cluster graph using Leiden algorithm."""
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
        
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
    """Cluster graph hierarchically (Levels 0, 1, 2)."""
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
        
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
    """Query community graph using global search."""
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
    
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
        reranking_config=getattr(request, 'reranking', None),  # Pass reranking config
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
    """Hybrid Search with Context Fusion (Local + Global)."""
    if not COMMUNITY_GRAPH_AVAILABLE or CommunityGraphService is None:
        raise RuntimeError("Community Graph service not available")
    
    if request.engine is None:
         raise ValueError("Engine configuration is required for vector store access")
         
    community_service = CommunityGraphService(
        graph_service=graph_service,
        graph_rag_service=graph_rag_service
    )
    
    question_text = request.question if isinstance(request.question, str) else request.question[0].text
    
    return await community_service.context_fusion_search(
        question=question_text,
        namespace=request.namespace,
        search_type=request.search_type,
        sparse_encoder_injected=request.sparse_encoder,
        reranking_injected=request.reranking,
        engine=request.engine,
        vector_store_repo=repo,
        llm=llm,
        llm_embeddings=llm_embeddings,
        query_type=request.query_type,
        chat_history_dict=request.chat_history_dict
    )
