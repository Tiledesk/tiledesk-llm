"""
FastAPI Controllers for Knowledge Graph API endpoints.
Provides RESTful API for managing nodes and relationships in Neo4j.
"""

import logging
from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Dict, Any
from tilellm.models.schemas import RepositoryEngine
from tilellm.models.schemas.multimodal_content import TextContent
from tilellm.shared.utility import inject_repo_async, inject_llm_chat_async, inject_llm_async
from .models import Node, NodeUpdate, Relationship, RelationshipUpdate
from .models.schemas import GraphQARequest, GraphQAResponse, GraphCreateRequest, GraphCreateResponse, GraphQAAdvancedRequest, GraphQAAdvancedResponse, GraphClusterRequest, GraphClusterResponse
from .services import GraphService, GraphRAGService
from .repository import GraphRepository

logger = logging.getLogger(__name__)


"""
# ==================== HELPER FUNCTIONS ====================

@inject_repo_async
async def import_documents_from_vector_store(
    repository_engine: RepositoryEngine, 
    namespace: str, 
    limit: int = 100,
    repo=None
) -> Dict[str, Any]:
    '''
    Import documents from vector store namespace to knowledge graph.
    
    Args:
        repository_engine: Repository engine configuration
        namespace: Namespace/collection name
        limit: Maximum chunks to process
        repo: Injected vector store repository
        
    Returns:
        Dictionary with import statistics
    '''
    try:
        # Get all chunks from the namespace
        logger.info(f"Retrieving chunks from namespace: {namespace}, limit: {limit}")
        chunks_result = await repo.get_all_obj_namespace(
            engine=repository_engine.engine,
            namespace=namespace
        )
        
        # Process chunks (for now just count them)
        chunks = chunks_result.matches if hasattr(chunks_result, 'matches') else []
        chunk_contents = []
        for chunk in chunks[:limit]:
            # Extract text content from chunk
            # Assuming chunk has 'metadata' with 'text' or 'content' field
            metadata = getattr(chunk, 'metadata', {})
            text = metadata.get('text') or metadata.get('content') or getattr(chunk, 'text', '')
            if text:
                chunk_contents.append(text)
        
        logger.info(f"Retrieved {len(chunk_contents)} chunks with text content")
        
        # TODO: Implement GraphRAG extraction using self.llm
        # For now, create stub nodes and relationships
        
        nodes_created = 0
        relationships_created = 0
        
        # Example: create a document node for each chunk
        for i, text in enumerate(chunk_contents[:10]):  # Limit to 10 for stub
            try:
                # Create a Document node
                node = Node(
                    label="Document",
                    properties={
                        "title": f"Document from chunk {i}",
                        "content": text[:200] + "..." if len(text) > 200 else text,  # Truncate
                        "source": namespace,
                        "chunk_index": i
                    }
                )
                created_node = graph_service.create_node(node)
                nodes_created += 1
                
                # Create relationships between consecutive documents
                if i > 0:
                    rel = Relationship(
                        source_id=prev_node_id,
                        target_id=created_node.id,
                        type="NEXT_CHUNK",
                        properties={"weight": 1.0}
                    )
                    graph_service.create_relationship(rel)
                    relationships_created += 1
                prev_node_id = created_node.id
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                continue
        
        return {
            "namespace": namespace,
            "chunks_processed": len(chunk_contents),
            "nodes_created": nodes_created,
            "relationships_created": relationships_created,
            "status": "partial" if chunk_contents else "empty"
        }
        
    except Exception as e:
        logger.error(f"Error importing documents from vector store: {e}")
        raise

"""
# ==================== NEO4J CONFIGURATION ====================
# TODO: Move these to environment variables or configuration file
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "n3o4j_Mammata"
NEO4J_POOL_SIZE = 50

# ==================== ROUTER SETUP ====================
router = APIRouter(
    prefix="/api/kg",
    tags=["Knowledge Graph"]
)

# Initialize repository and service, and ensure connection is verified
try:
    repository = GraphRepository(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        max_connection_pool_size=NEO4J_POOL_SIZE
    )
    if not repository.verify_connection():
        raise ConnectionError("Failed to verify connection to Neo4j database.")
    
    graph_service = GraphService(repository=repository)
    graph_rag_service = GraphRAGService(graph_service=graph_service)
    logger.info("Successfully initialized and verified Neo4j connection and services.")

except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize Neo4j connection: {e}")
    # Create dummy services that will raise errors if used, but allow app to start
    # This might be useful for environments where API should be up even if DB is down
    repository = None
    graph_service = GraphService()
    graph_rag_service = GraphRAGService(graph_service=graph_service)
    logger.warning("Knowledge Graph services are NOT operational due to DB connection failure.")


# ==================== IMPORT HELPER FUNCTION ====================

@inject_llm_async
@inject_repo_async
async def import_documents_from_vector_store_v2(
    request: GraphCreateRequest,
    repo=None,
    chat_model=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Import documents from vector store namespace to knowledge graph.
    
    Args:
        request: GraphCreateRequest with LLM configuration and vector store engine
        repo: Injected vector store repository
        chat_model: Injected LLM instance for GraphRAG extraction
        
    Returns:
        Dictionary with import statistics
    """
    try:
        if repo is None:
            raise RuntimeError("Vector store repository not injected")
        
        # Check if engine is provided
        if request.engine is None:
            raise ValueError("Engine configuration is required for vector store access")
        
        # Log LLM configuration for debugging
        logger.info(f"LLM injection debug - llm: {chat_model}")
        
        # Check if LLM is available for GraphRAG extraction
        if chat_model is None:
            logger.error("LLM not injected for GraphRAG extraction. Check LLM configuration in request.")
            raise ValueError("LLM configuration is required for GraphRAG extraction.")
        
        # Create RepositoryEngine object
        repository_engine = RepositoryEngine(engine=request.engine)
        
        result = await graph_rag_service.import_from_vector_store(
            namespace=request.namespace,
            vector_store_repo=repo,
            engine=repository_engine.engine,
            limit=request.limit or 100,
            llm=chat_model,
            index_name=request.index_name
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in import_documents_from_vector_store_v2: {e}")
        raise


# ==================== GRAPH CLUSTER HELPER ====================

@inject_llm_async
async def cluster_graph_v2(
    request: GraphClusterRequest,
    chat_model=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Cluster the graph and generate community reports.
    """
    try:
        if chat_model is None:
            raise ValueError("LLM configuration is required for community report generation.")
            
        result = await graph_rag_service.cluster_graph(
            llm=chat_model,
            level=request.level or 0,
            namespace=request.namespace,
            index_name=request.index_name
        )
        return result
    except Exception as e:
        logger.error(f"Error in cluster_graph_v2: {e}")
        raise


# ==================== GRAPH QA HELPER ====================

@inject_llm_chat_async
async def answer_with_graph_v2(
    request: GraphQARequest, 
    llm=None, 
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Answer a question using knowledge graph with LLM injection.
    """
    try:
        # Extract text from question which can be string or list of MultimodalContent
        question_text = ""
        if isinstance(request.question, str):
            question_text = request.question
        elif isinstance(request.question, list):
            # Extract text from multimodal content
            for content in request.question:
                if isinstance(content, TextContent):
                    question_text = content.text
                    break
        
        if not question_text:
            raise ValueError("No valid question text provided. For GraphRAG QA, provide a text question.")
        
        result = await graph_rag_service.answer_with_graph(
            question=question_text,
            namespace=request.namespace,
            max_results=request.max_results or 10,
            similarity_threshold=request.similarity_threshold or 0.3,
            llm=llm,
            llm_embeddings=llm_embeddings,
            index_name=request.index_name
        )
        return result
    except Exception as e:
        logger.error(f"Error in answer_with_graph_v2: {e}")
        raise


@inject_llm_chat_async
@inject_repo_async
async def answer_with_graph_advanced(
    request: GraphQAAdvancedRequest, 
    repo=None, 
    llm=None, 
    llm_embeddings=None,
    embedding_dimension=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Advanced GraphRAG QA with hybrid retrieval and graph expansion.
    """
    try:
        # Extract question text
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
        
        logger.info(f"Advanced Graph QA for question: {question_text}, namespace: {request.namespace}")
        
        # Check if engine is provided for vector store access
        if request.engine is None:
            raise ValueError("Engine configuration is required for vector store access in advanced Graph QA.")
        
        if llm_embeddings is None:
            logger.warning("Embeddings not injected! This will cause vector search errors.")
            
        # Pass engine, repo, llm, and embeddings to the service
        result = await graph_rag_service.answer_with_graph_advanced(
            question=question_text,
            namespace=request.namespace,
            max_results=request.max_results or 20,
            similarity_threshold=request.similarity_threshold or 0.3,
            vector_weight=request.vector_weight if request.vector_weight is not None else 1.0,
            keyword_weight=request.keyword_weight if request.keyword_weight is not None else 1.0,
            graph_weight=request.graph_weight if request.graph_weight is not None else 1.0,
            query_type=request.query_type,
            use_reranking=request.use_reranking if request.use_reranking is not None else True,
            max_expansion_nodes=request.max_expansion_nodes if request.max_expansion_nodes is not None else 20,
            engine=request.engine,
            vector_store_repo=repo,
            llm=llm,
            llm_embeddings=llm_embeddings,
            embedding_dimension=embedding_dimension,
            chat_history_dict=request.chat_history_dict
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in answer_with_graph_advanced: {e}")
        raise


# ==================== UTILITY ENDPOINTS ====================

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Check if the Neo4j connection is working.
    """
    try:
        is_connected = graph_service.verify_connection()
        if is_connected:
            return {"status": "healthy", "database": "connected"}
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j database connection failed"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/stats")
def get_stats():
    """
    Get database statistics (node count, relationship count, etc.).
    """
    try:
        stats = graph_service.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")


# ==================== NODE ENDPOINTS ====================

@router.post("/nodes", status_code=status.HTTP_201_CREATED, response_model=Node)
def create_node(node: Node):
    """
    Create a new node in the knowledge graph.

    - **label**: The type/category of the node (e.g., 'Document', 'Person')
    - **properties**: Key-value pairs of node attributes

    Returns the created node with its generated ID.
    """
    try:
        return graph_service.create_node(node)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create node: {str(e)}")


@router.get("/nodes/{node_id}", response_model=Node)
def get_node(node_id: str):
    """
    Retrieve a node by its ID.
    """
    try:
        node = graph_service.get_node(node_id)
        if not node:
            raise HTTPException(
                status_code=404,
                detail=f"Node with ID '{node_id}' not found"
            )
        return node
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve node: {str(e)}")


@router.get("/nodes", response_model=List[Node])
def get_nodes_by_label(
    label: str = Query(..., description="Node label to filter by"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of nodes to return")
):
    """
    Get all nodes with a specific label.
    """
    try:
        nodes = graph_service.get_nodes_by_label(label, limit)
        return nodes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve nodes: {str(e)}")


@router.get("/nodes/search", response_model=List[Node])
def search_nodes(
    label: str = Query(..., description="Node label"),
    property_key: str = Query(..., description="Property key to search"),
    property_value: str = Query(..., description="Property value to match"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of nodes to return")
):
    """
    Search for nodes by property value.
    """
    try:
        nodes = graph_service.search_nodes(label, property_key, property_value, limit)
        return nodes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.put("/nodes/{node_id}", response_model=Node)
def update_node(node_id: str, node_update: NodeUpdate):
    """
    Update a node's label and/or properties.

    Only provide the fields you want to update.
    """
    try:
        updated_node = graph_service.update_node(node_id, node_update)
        if not updated_node:
            raise HTTPException(
                status_code=404,
                detail=f"Node with ID '{node_id}' not found"
            )
        return updated_node
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update node: {str(e)}")


@router.patch("/nodes/{node_id}", response_model=Node)
def patch_node(node_id: str, node_update: NodeUpdate):
    """
    Partially update a node's properties (alias for PUT).
    """
    return update_node(node_id, node_update)


@router.delete("/nodes/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_node(
    node_id: str,
    detach: bool = Query(True, description="If true, also delete all relationships")
):
    """
    Delete a node from the graph.

    - **detach**: If true, all relationships connected to this node will be deleted first
    """
    try:
        deleted = graph_service.delete_node(node_id, detach)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Node with ID '{node_id}' not found"
            )
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete node: {str(e)}")


# ==================== RELATIONSHIP ENDPOINTS ====================

@router.post("/relationships", status_code=status.HTTP_201_CREATED, response_model=Relationship)
def create_relationship(relationship: Relationship):
    """
    Create a relationship between two nodes.

    - **source_id**: ID of the source node
    - **target_id**: ID of the target node
    - **type**: Relationship type (e.g., 'RELATES_TO', 'CONTAINS')
    - **properties**: Optional key-value pairs for relationship attributes

    Returns the created relationship with its generated ID.
    """
    try:
        return graph_service.create_relationship(relationship)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create relationship: {str(e)}")


@router.get("/relationships/{relationship_id}", response_model=Relationship)
def get_relationship(relationship_id: str):
    """
    Retrieve a relationship by its ID.
    """
    try:
        relationship = graph_service.get_relationship(relationship_id)
        if not relationship:
            raise HTTPException(
                status_code=404,
                detail=f"Relationship with ID '{relationship_id}' not found"
            )
        return relationship
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve relationship: {str(e)}")


@router.get("/nodes/{node_id}/relationships", response_model=List[Relationship])
def get_node_relationships(
    node_id: str,
    direction: str = Query("both", regex="^(incoming|outgoing|both)$", description="Relationship direction")
):
    """
    Get all relationships connected to a node.

    - **direction**: Filter by direction - 'incoming', 'outgoing', or 'both'
    """
    try:
        relationships = graph_service.get_node_relationships(node_id, direction)
        return relationships
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve relationships: {str(e)}")


@router.put("/relationships/{relationship_id}", response_model=Relationship)
def update_relationship(relationship_id: str, relationship_update: RelationshipUpdate):
    """
    Update a relationship's type and/or properties.

    Note: Changing the relationship type requires recreating it in Neo4j.
    """
    try:
        updated_rel = graph_service.update_relationship(relationship_id, relationship_update)
        if not updated_rel:
            raise HTTPException(
                status_code=404,
                detail=f"Relationship with ID '{relationship_id}' not found"
            )
        return updated_rel
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update relationship: {str(e)}")


@router.patch("/relationships/{relationship_id}", response_model=Relationship)
def patch_relationship(relationship_id: str, relationship_update: RelationshipUpdate):
    """
    Partially update a relationship's properties (alias for PUT).
    """
    return update_relationship(relationship_id, relationship_update)


@router.delete("/relationships/{relationship_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_relationship(relationship_id: str):
    """
    Delete a relationship from the graph.
    """
    try:
        deleted = graph_service.delete_relationship(relationship_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Relationship with ID '{relationship_id}' not found"
            )
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete relationship: {str(e)}")


# ==================== GRAPH QA ENDPOINT ====================

@router.post("/qa", response_model=GraphQAResponse)
async def graph_qa(request: GraphQARequest):
    """
    Question Answering using Knowledge Graph structure (GraphRAG).
    
    Uses Neo4j to retrieve relevant entities and relationships,
    then generates an answer using LLM.
    """
    try:
        result = await answer_with_graph_v2(request)
        
        return GraphQAResponse(
            answer=result["answer"],
            entities=result["entities"],
            relationships=result["relationships"],
            query_used=result["query_used"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph QA failed: {str(e)}")


@router.post("/cluster", response_model=GraphClusterResponse)
async def graph_cluster(request: GraphClusterRequest):
    """
    Detect communities in the knowledge graph and generate reports for each.
    
    This implements the hierarchical clustering (Leiden/Louvain) and 
    summarization phase of GraphRAG.
    """
    try:
        result = await cluster_graph_v2(request)
        return GraphClusterResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph clustering failed: {str(e)}")


# ==================== GRAPH QA ADVANCED ENDPOINT ====================

@router.post("/graphqa", response_model=GraphQAAdvancedResponse)
async def graph_qa_advanced(request: GraphQAAdvancedRequest):
    """
    Advanced Question Answering using GraphRAG with hybrid retrieval.
    
    Implements the full GraphRAG pipeline:
    1. Global search (community-based if enabled)
    2. Parallel retrieval (vector + keyword)
    3. Reciprocal Rank Fusion (RRF)
    4. Graph expansion from seed nodes
    5. Cross-encoder reranking (if enabled)
    
    Uses weights to balance vector, keyword, and graph retrieval.
    """
    try:
        result = await answer_with_graph_advanced(request)
        
        return GraphQAAdvancedResponse(
            answer=result["answer"],
            entities=result["entities"],
            relationships=result["relationships"],
            query_used=result["query_used"],
            retrieval_strategy=result["retrieval_strategy"],
            scores=result["scores"],
            expanded_nodes=result["expanded_nodes"],
            expanded_relationships=result["expanded_relationships"]
        )
    except Exception as e:
        logger.error(f"Advanced Graph QA failed: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced Graph QA failed: {str(e)}")


# ==================== GRAPH CREATE ENDPOINT ====================

@router.post("/create", response_model=GraphCreateResponse)
async def graph_create(request: GraphCreateRequest):
    """
    Create/import a knowledge graph from documents in a vector store namespace.
    
    Extracts entities and relationships from documents and stores them in Neo4j.
    """
    try:
        logger.info(f"Graph create request for namespace: {request.namespace}, limit: {request.limit}, overwrite: {request.overwrite}")
        
        # Check if engine is provided
        if request.engine is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Engine configuration is required for vector store access"
            )
        
        # Import documents using the helper function (LLM and repo injected)
        result = await import_documents_from_vector_store_v2(request)
        
        return GraphCreateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph creation failed: {str(e)}")

