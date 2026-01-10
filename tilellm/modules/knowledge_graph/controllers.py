"""
FastAPI Controllers for Knowledge Graph API endpoints.
Provides RESTful API for managing nodes and relationships in Neo4j.
"""

import logging
from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Union
from pydantic import SecretStr

from .models import Node, NodeUpdate, Relationship, RelationshipUpdate
from .models.schemas import (
    GraphQARequest, GraphQAResponse,
    GraphCreateRequest, GraphCreateResponse,
    GraphQAAdvancedRequest, GraphQAAdvancedResponse,
    GraphClusterRequest, GraphClusterResponse, CommunityQAResponse,
    AddDocumentRequest, AddDocumentResponse,
    GraphNetworkResponse, AsyncTaskResponse, TaskPollResponse, MultimodalSearchResponse
)

# Import business logic
import tilellm.modules.knowledge_graph.logic as kg_logic

import os

from ...shared.llm_config import serialize_with_secrets

from tilellm.shared.utility import get_service_config

# Import Taskiq tasks if available
try:
    from tilellm.modules.task_executor.tasks import (
        task_graph_create, task_add_document,
        task_louvain_cluster, task_leiden_cluster,
        task_hierarchical_cluster, task_community_analysis
    )
    from tilellm.modules.task_executor.broker import broker
    TASKIQ_AVAILABLE = True
except ImportError:
    TASKIQ_AVAILABLE = False

ENABLE_TASKIQ = os.environ.get("ENABLE_TASKIQ", "false").lower() == "true"
ENABLE_TASKIQ = ENABLE_TASKIQ and TASKIQ_AVAILABLE

logger = logging.getLogger(__name__)



# ==================== ROUTER SETUP ====================
router = APIRouter(
    prefix="/api/kg",
    tags=["Knowledge Graph"]
)

# ==================== UTILITY ENDPOINTS ====================

@router.get("/tasks/{task_id}", response_model=TaskPollResponse)
async def get_task_status(task_id: str):
    """
    Check the status of an asynchronous task.
    """
    if not (ENABLE_TASKIQ and TASKIQ_AVAILABLE):
        raise HTTPException(status_code=501, detail="Async tasks not enabled")
    
    try:
        # Check if task is ready
        result_backend = broker.result_backend
        is_ready = await result_backend.is_result_ready(task_id)
        
        if not is_ready:
            return TaskPollResponse(task_id=task_id, status="in_progress")
            
        result = await result_backend.get_result(task_id)
        
        if result.is_err:
             # Extract error message properly
             return TaskPollResponse(
                 task_id=task_id, 
                 status="failed", 
                 error=str(result.error) if hasattr(result, 'error') else str(result.return_value)
             )
             
        return TaskPollResponse(
            task_id=task_id, 
            status="success", 
            result=result.return_value
        )
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve task status: {str(e)}")

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Check if the Neo4j connection is working."""
    try:
        return kg_logic.check_health()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/stats")
def get_stats():
    """Get database statistics (node count, relationship count, etc.)."""
    try:
        return kg_logic.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")

# ==================== NODE ENDPOINTS ====================

@router.post("/nodes", status_code=status.HTTP_201_CREATED, response_model=Node)
def create_node(node: Node):
    """Create a new node in the knowledge graph."""
    try:
        return kg_logic.create_node(node)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create node: {str(e)}")

@router.get("/nodes/{node_id}", response_model=Node)
def get_node(node_id: str):
    """Retrieve a node by its ID."""
    try:
        node = kg_logic.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found")
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
    """Get all nodes with a specific label."""
    try:
        return kg_logic.get_nodes_by_label(label, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve nodes: {str(e)}")

@router.get("/nodes/search", response_model=List[Node])
def search_nodes(
    label: str = Query(..., description="Node label"),
    property_key: str = Query(..., description="Property key to search"),
    property_value: str = Query(..., description="Property value to match"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of nodes to return")
):
    """Search for nodes by property value."""
    try:
        return kg_logic.search_nodes(label, property_key, property_value, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.put("/nodes/{node_id}", response_model=Node)
def update_node(node_id: str, node_update: NodeUpdate):
    """Update a node's label and/or properties."""
    try:
        updated_node = kg_logic.update_node(node_id, node_update)
        if not updated_node:
            raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found")
        return updated_node
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update node: {str(e)}")

@router.patch("/nodes/{node_id}", response_model=Node)
def patch_node(node_id: str, node_update: NodeUpdate):
    """Partially update a node's properties (alias for PUT)."""
    return update_node(node_id, node_update)

@router.delete("/nodes/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_node(
    node_id: str,
    detach: bool = Query(True, description="If true, also delete all relationships")
):
    """Delete a node from the graph."""
    try:
        deleted = kg_logic.delete_node(node_id, detach)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found")
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete node: {str(e)}")


# ==================== RELATIONSHIP ENDPOINTS ====================

@router.post("/relationships", status_code=status.HTTP_201_CREATED, response_model=Relationship)
def create_relationship(relationship: Relationship):
    """Create a relationship between two nodes."""
    try:
        return kg_logic.create_relationship(relationship)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create relationship: {str(e)}")

@router.get("/relationships/{relationship_id}", response_model=Relationship)
def get_relationship(relationship_id: str):
    """Retrieve a relationship by its ID."""
    try:
        relationship = kg_logic.get_relationship(relationship_id)
        if not relationship:
            raise HTTPException(status_code=404, detail=f"Relationship with ID '{relationship_id}' not found")
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
    """Get all relationships connected to a node."""
    try:
        return kg_logic.get_node_relationships(node_id, direction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve relationships: {str(e)}")

@router.put("/relationships/{relationship_id}", response_model=Relationship)
def update_relationship(relationship_id: str, relationship_update: RelationshipUpdate):
    """Update a relationship's type and/or properties."""
    try:
        updated_rel = kg_logic.update_relationship(relationship_id, relationship_update)
        if not updated_rel:
            raise HTTPException(status_code=404, detail=f"Relationship with ID '{relationship_id}' not found")
        return updated_rel
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update relationship: {str(e)}")

@router.patch("/relationships/{relationship_id}", response_model=Relationship)
def patch_relationship(relationship_id: str, relationship_update: RelationshipUpdate):
    """Partially update a relationship's properties (alias for PUT)."""
    return update_relationship(relationship_id, relationship_update)

@router.delete("/relationships/{relationship_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_relationship(relationship_id: str):
    """Delete a relationship from the graph."""
    try:
        deleted = kg_logic.delete_relationship(relationship_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Relationship with ID '{relationship_id}' not found")
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete relationship: {str(e)}")

# ==================== GRAPH CREATE ENDPOINT ====================

@router.post("/create", response_model=Union[AsyncTaskResponse, GraphCreateResponse])
async def graph_create(request: GraphCreateRequest):
    """Create/import a community graph using existing GraphRAG extraction and clustering."""
    try:
        if ENABLE_TASKIQ and TASKIQ_AVAILABLE:
            payload = serialize_with_secrets(request.model_dump(mode='python'))
            task = await task_graph_create.kiq(payload)
            return AsyncTaskResponse(task_id=task.task_id)

        result = await kg_logic.create_graph(request)
        return GraphCreateResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Add document to graph failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document to graph: {str(e)}")



@router.post("/add-document", response_model=Union[AsyncTaskResponse, AddDocumentResponse])
async def add_document(request: AddDocumentRequest):
    """
    Add a document to the knowledge graph by retrieving chunks from vector store.

    This endpoint enables incremental graph updates when a new document is added
    to the knowledge base, without regenerating the entire namespace graph. It:

    1. Retrieves all chunks of the document from vector store using metadata_id
    2. Extracts entities and relationships from each chunk using GraphRAG
    3. Optionally deduplicates entities (reuses existing nodes with same name/type)
    4. Creates new nodes for new entities
    5. Creates relationships between entities

    **Use Cases:**
    - Add newly uploaded documents to existing knowledge graph incrementally
    - Update graph after document ingestion without full rebuild
    - Maintain graph consistency with vector store

    **Parameters:**
    - `metadata_id`: Document identifier in vector store (e.g., file UUID)
    - `namespace`: Knowledge base namespace
    - `engine`: Engine configuration with name, index_name, and type/deployment
        - For Pinecone: include `type` (pod/serverless)
        - For Qdrant: include `deployment` (local/cloud)
    - `deduplicate_entities`: Reuse existing entity nodes (default: true)

    **Note:** After adding multiple documents, consider regenerating community reports
    using the `/clusterms` or `/cluster-leiden` endpoints to update graph summaries.

    **Example Request:**
    ```json
    {
        "metadata_id": "doc_12345_uuid",
        "namespace": "economia",
        "engine": {
            "name": "qdrant",
            "index_name": "economia_kb",
            "deployment": "local",
            "host": "localhost",
            "port": 6333
        },
        "deduplicate_entities": true,
        "llm_key": "my-llm-key",
        "model": "gpt-4"
    }
    ```
    """
    try:
        logger.info(f"Add document to graph: metadata_id={request.metadata_id}, namespace={request.namespace}")
        if ENABLE_TASKIQ and TASKIQ_AVAILABLE:
            payload = serialize_with_secrets(request.model_dump(mode='python'))
            task = await task_add_document.kiq(payload)
            return AsyncTaskResponse(task_id=task.task_id)

        result = await kg_logic.add_document_to_graph(request)
        return AddDocumentResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Add document to graph failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document to graph: {str(e)}")

@router.post("/louvain-cluster", response_model=Union[AsyncTaskResponse, GraphClusterResponse])
async def graph_cluster_louvain(request: GraphClusterRequest):
    """Perform Louvain clustering and generate community reports (Parquet/MinIO)."""
    try:
        if ENABLE_TASKIQ and TASKIQ_AVAILABLE:
            payload = serialize_with_secrets(request.model_dump(mode='python'))
            task = await task_louvain_cluster.kiq(payload)
            return AsyncTaskResponse(task_id=task.task_id)

        result = await kg_logic.cluster_graph_louvain(request)
        return GraphClusterResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Cluster MS failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cluster MS failed: {str(e)}")


@router.post("/leiden-cluster", response_model=Union[AsyncTaskResponse, GraphClusterResponse])
async def graph_cluster_leiden(request: GraphClusterRequest):
    """Perform Leiden clustering (via igraph) and generate community reports."""
    try:
        if ENABLE_TASKIQ and TASKIQ_AVAILABLE:
            payload = serialize_with_secrets(request.model_dump(mode='python'))
            task = await task_leiden_cluster.kiq(payload)
            return AsyncTaskResponse(task_id=task.task_id)

        result = await kg_logic.cluster_graph_leiden(request)
        return GraphClusterResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Leiden MS failed: {e}")
        raise HTTPException(status_code=500, detail=f"Leiden MS failed: {str(e)}")


@router.post("/hierarchical", response_model=Union[AsyncTaskResponse, GraphClusterResponse])
async def graph_cluster_hierarchical(request: GraphClusterRequest):
    """Perform Hierarchical Clustering (Levels 0, 1, 2) using Leiden."""
    try:
        if ENABLE_TASKIQ and TASKIQ_AVAILABLE:
            payload = serialize_with_secrets(request.model_dump(mode='python'))
            task = await task_hierarchical_cluster.kiq(payload)
            return AsyncTaskResponse(task_id=task.task_id)

        result = await kg_logic.cluster_graph_hierarchical(request)
        return GraphClusterResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Hierarchical MS failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hierarchical MS failed: {str(e)}")

# ==================== GRAPH QA ENDPOINT ====================

@router.post("/qa", tags=["Search Methods"], response_model=CommunityQAResponse)
async def graph_qa_community(request: GraphQARequest):
    """
    **METHOD 1: Community/Global Search**
    Performs global search ONLY on community reports.
    """
    try:
        result = await kg_logic.query_graph(request)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Community graph QA failed: {e}")
        raise HTTPException(status_code=500, detail=f"Community graph QA failed: {str(e)}")



@router.post("/hybrid", response_model=GraphQAAdvancedResponse, tags=["Search Methods"])
async def graph_qa_hybrid(request: GraphQAAdvancedRequest):
    """
    **METHOD 2: Integrated Hybrid Search**
    Unified pipeline: Global + Parallel Retrieval + RRF + Expansion + Reranking.
    """
    try:
        result = await kg_logic.context_fusion_graph_search(request)
        
        return GraphQAAdvancedResponse(
            answer=result.get("answer", ""),
            entities=result.get("entities", []),
            relationships=result.get("relationships", []),
            query_used=request.question if isinstance(request.question, str) else "",
            retrieval_strategy=result.get("retrieval_strategy", "integrated_hybrid"),
            scores=result.get("scores", {}),
            expanded_nodes=result.get("expanded_nodes", []),
            expanded_relationships=result.get("expanded_relationships", []),
            chat_history_dict=result.get("chat_history_dict")
        )
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@router.get("/network", response_model=GraphNetworkResponse)
def get_graph_network(
    namespace: str = Query(None, description="Filter by namespace"),
    index_name: str = Query(None, description="Filter by index_name"),
    node_limit: int = Query(1000, ge=1, le=10000, description="Maximum number of nodes to return"),
    relationship_limit: int = Query(5000, ge=1, le=50000, description="Maximum number of relationships to return"),
    node_labels: List[str] = Query(None, description="Filter nodes by labels (e.g., PERSON, ORGANIZATION)"),
    community: bool = Query(False, description="If true, returns only BELONGS_TO_COMMUNITY relationships")

):
    """
    Get the graph network (nodes + relationships) for visualization.

    Returns nodes and relationships that can be visualized using graph visualization libraries
    like D3.js, Cytoscape, or other graph rendering tools.

    **Use Cases:**
    - Visualize knowledge graph in a frontend application
    - Export graph data for analysis
    - Understand entity relationships in a namespace

    **Response Format:**
    ```json
    {
        "nodes": [
            {"id": "node_id", "label": "PERSON", "properties": {"name": "Mario Draghi", ...}},
            ...
        ],
        "relationships": [
            {"id": "rel_id", "source_id": "node1", "target_id": "node2", "type": "RELATED_TO", "properties": {...}},
            ...
        ],
        "stats": {
            "node_count": 150,
            "relationship_count": 300,
            "filtered_by": {"namespace": "bancaitalia", ...}
        }
    }
    ```

    **Example Usage:**
    - Get all entities: `GET /api/kg/network?namespace=bancaitalia&node_limit=500`
    - Get only people and organizations: `GET /api/kg/network?namespace=economia&node_labels=PERSON&node_labels=ORGANIZATION`
    - Get community graph: `GET /api/kg/network?namespace=economia&index_name=my_index&community=true`
    """
    try:
        result = kg_logic.get_graph_network(
            namespace=namespace,
            index_name=index_name,
            node_limit=node_limit,
            relationship_limit=relationship_limit,
            node_labels=node_labels,
            community=community
        )
        return GraphNetworkResponse(**result)
    except Exception as e:
        logger.error(f"Failed to retrieve graph network: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve graph network: {str(e)}")


@router.post("/multimodal-search", response_model=MultimodalSearchResponse, tags=["Search Methods"])
async def multimodal_search(request: GraphQAAdvancedRequest):
    """
    **METHOD 3: Multimodal Search**
    Combines Text (Vector/Graph), Tables (Semantic), and Images (Semantic).
    """
    try:
        result = await kg_logic.multimodal_search(request)
        return MultimodalSearchResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", {}),
            query_used=request.question if isinstance(request.question, str) else request.question[0].text
        )
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Multimodal search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multimodal search failed: {str(e)}")


@router.post("/community-analysis", response_model=Union[AsyncTaskResponse, GraphClusterResponse])
async def community_analysis(request: GraphClusterRequest):
    """
    **Analysis: Document Community Detection**
    Analyzes collection to find communities. Runs as async task.
    """
    try:
        if ENABLE_TASKIQ and TASKIQ_AVAILABLE:
            payload = serialize_with_secrets(request.model_dump(mode='python'))
            task = await task_community_analysis.kiq(payload)
            return AsyncTaskResponse(task_id=task.task_id)

        result = await kg_logic.analyze_community(request)
        # Map result to response model
        return GraphClusterResponse(
            status=result.get("status", "success"),
            communities_detected=result.get("communities", 0),
            reports_created=result.get("reports", 0),
            message="Analysis completed"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Community analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Community analysis failed: {str(e)}")
