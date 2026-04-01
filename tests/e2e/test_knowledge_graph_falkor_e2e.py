"""
End-to-End tests for Knowledge Graph Falkor API endpoints.
Tests full API flow with mocked backend services.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from tilellm.modules.knowledge_graph_falkor.models import Node, Relationship


@pytest.fixture
def mock_kg_logic():
    """Mock knowledge graph logic module."""
    with patch('tilellm.modules.knowledge_graph_falkor.controllers.kg_logic') as mock:
        mock.check_health = Mock(return_value={"status": "healthy", "database": "connected"})
        mock.get_stats = Mock(return_value={
            "node_count": 100,
            "relationship_count": 50,
            "graph_name": "test_graph"
        })
        mock.create_node = AsyncMock()
        mock.get_node = AsyncMock()
        mock.get_nodes_by_label = AsyncMock()
        mock.search_nodes = AsyncMock()
        mock.update_node = AsyncMock()
        mock.delete_node = AsyncMock()
        mock.create_relationship = AsyncMock()
        mock.get_relationship = AsyncMock()
        mock.get_node_relationships = AsyncMock()
        mock.update_relationship = AsyncMock()
        mock.delete_relationship = AsyncMock()
        mock.get_graph_network = AsyncMock()
        mock.query_graph = AsyncMock()
        mock.context_fusion_graph_search = AsyncMock()
        mock.multimodal_search = AsyncMock()
        mock.add_document_to_graph = AsyncMock()
        mock.create_graph = AsyncMock()
        mock.cluster_graph_louvain = AsyncMock()
        mock.cluster_graph_leiden = AsyncMock()
        mock.cluster_graph_hierarchical = AsyncMock()
        mock.analyze_community = AsyncMock()
        yield mock


@pytest.fixture
def client(mock_kg_logic):
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI
    from tilellm.modules.knowledge_graph_falkor.controllers import router
    
    app = FastAPI()
    app.include_router(router)
    
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check_success(self, client, mock_kg_logic):
        """Test successful health check."""
        response = client.get("/api/kg-falkor/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
    
    def test_health_check_failure(self, client, mock_kg_logic):
        """Test health check when database fails."""
        mock_kg_logic.check_health.side_effect = Exception("Connection failed")
        
        response = client.get("/api/kg-falkor/health")
        
        assert response.status_code == 503
        assert "Health check failed" in response.json()["detail"]
    
    def test_get_stats(self, client, mock_kg_logic):
        """Test getting database statistics."""
        response = client.get("/api/kg-falkor/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["node_count"] == 100
        assert data["relationship_count"] == 50


class TestNodeEndpoints:
    """Test node CRUD endpoints."""
    
    def test_create_node(self, client, mock_kg_logic):
        """Test creating a node."""
        mock_kg_logic.create_node.return_value = Node(
            id="123",
            label="Person",
            properties={"name": "John", "age": 30}
        )
        
        response = client.post("/api/kg-falkor/nodes", json={
            "label": "Person",
            "properties": {"name": "John", "age": 30}
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "123"
        assert data["label"] == "Person"
        assert data["properties"]["name"] == "John"
    
    def test_create_node_validation_error(self, client, mock_kg_logic):
        """Test node creation with validation error."""
        mock_kg_logic.create_node.side_effect = ValueError("Node must have a label")
        
        response = client.post("/api/kg-falkor/nodes", json={
            "label": "",
            "properties": {"name": "John"}
        })
        
        assert response.status_code == 400
        assert "Node must have a label" in response.json()["detail"]
    
    def test_get_node(self, client, mock_kg_logic):
        """Test getting a node by ID."""
        mock_kg_logic.get_node.return_value = Node(
            id="456",
            label="Organization",
            properties={"name": "Acme Corp"}
        )
        
        response = client.get("/api/kg-falkor/nodes/456")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "456"
        assert data["label"] == "Organization"
    
    def test_get_node_not_found(self, client, mock_kg_logic):
        """Test getting a non-existent node."""
        mock_kg_logic.get_node.return_value = None
        
        response = client.get("/api/kg-falkor/nodes/999")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_nodes_by_label(self, client, mock_kg_logic):
        """Test getting nodes by label."""
        mock_kg_logic.get_nodes_by_label.return_value = [
            Node(id="1", label="Person", properties={"name": "Alice"}),
            Node(id="2", label="Person", properties={"name": "Bob"})
        ]
        
        response = client.get("/api/kg-falkor/nodes?label=Person&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(n["label"] == "Person" for n in data)
    
    def test_search_nodes(self, client, mock_kg_logic):
        """Test searching nodes by property."""
        mock_kg_logic.search_nodes.return_value = [
            Node(id="1", label="Person", properties={"name": "Alice"})
        ]
        
        response = client.get(
            "/api/kg-falkor/nodes/search?label=Person&property_key=name&property_value=Alice"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
    
    def test_update_node(self, client, mock_kg_logic):
        """Test updating a node."""
        mock_kg_logic.update_node.return_value = Node(
            id="1",
            label="Person",
            properties={"name": "John", "age": 30}
        )
        
        response = client.put("/api/kg-falkor/nodes/1", json={
            "properties": {"age": 30}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["properties"]["age"] == 30
    
    def test_delete_node(self, client, mock_kg_logic):
        """Test deleting a node."""
        mock_kg_logic.delete_node.return_value = True
        
        response = client.delete("/api/kg-falkor/nodes/1?detach=true")
        
        assert response.status_code == 204


class TestRelationshipEndpoints:
    """Test relationship CRUD endpoints."""
    
    def test_create_relationship(self, client, mock_kg_logic):
        """Test creating a relationship."""
        mock_kg_logic.create_relationship.return_value = Relationship(
            id="100",
            source_id="1",
            target_id="2",
            type="KNOWS",
            properties={"since": "2020"}
        )
        
        response = client.post("/api/kg-falkor/relationships", json={
            "source_id": "1",
            "target_id": "2",
            "type": "KNOWS",
            "properties": {"since": "2020"}
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "100"
        assert data["type"] == "KNOWS"
    
    def test_create_relationship_missing_nodes(self, client, mock_kg_logic):
        """Test creating relationship with missing nodes."""
        mock_kg_logic.create_relationship.side_effect = RuntimeError("Source node does not exist")
        
        response = client.post("/api/kg-falkor/relationships", json={
            "source_id": "999",
            "target_id": "2",
            "type": "KNOWS",
            "properties": {}
        })
        
        assert response.status_code == 400
        assert "does not exist" in response.json()["detail"]
    
    def test_get_relationship(self, client, mock_kg_logic):
        """Test getting a relationship."""
        mock_kg_logic.get_relationship.return_value = Relationship(
            id="5",
            source_id="10",
            target_id="20",
            type="WORKS_WITH",
            properties={}
        )
        
        response = client.get("/api/kg-falkor/relationships/5")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "5"
    
    def test_get_node_relationships(self, client, mock_kg_logic):
        """Test getting node relationships."""
        mock_kg_logic.get_node_relationships.return_value = [
            Relationship(id="1", source_id="10", target_id="20", type="KNOWS", properties={})
        ]
        
        response = client.get("/api/kg-falkor/nodes/10/relationships?direction=both")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
    
    def test_update_relationship(self, client, mock_kg_logic):
        """Test updating a relationship."""
        mock_kg_logic.update_relationship.return_value = Relationship(
            id="1",
            source_id="10",
            target_id="20",
            type="FRIENDS_WITH",
            properties={"strength": 0.9}
        )
        
        response = client.put("/api/kg-falkor/relationships/1", json={
            "type": "FRIENDS_WITH",
            "properties": {"strength": 0.9}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FRIENDS_WITH"
    
    def test_delete_relationship(self, client, mock_kg_logic):
        """Test deleting a relationship."""
        mock_kg_logic.delete_relationship.return_value = True
        
        response = client.delete("/api/kg-falkor/relationships/1")
        
        assert response.status_code == 204


class TestGraphNetworkEndpoint:
    """Test graph network endpoint."""
    
    def test_get_graph_network(self, client, mock_kg_logic):
        """Test getting graph network for visualization."""
        mock_kg_logic.get_graph_network.return_value = {
            "nodes": [
                {"id": "1", "label": "Person", "properties": {"name": "Alice"}}
            ],
            "relationships": [
                {"id": "r1", "source_id": "1", "target_id": "2", "type": "KNOWS", "properties": {}}
            ],
            "stats": {
                "node_count": 1,
                "relationship_count": 1,
                "filtered_by": {"namespace": "test"}
            }
        }
        
        response = client.get("/api/kg-falkor/network?namespace=test&node_limit=100")
        
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "relationships" in data
        assert "stats" in data
        assert len(data["nodes"]) == 1


class TestQAEndpoints:
    """Test question answering endpoints."""
    
    def test_graph_qa_community(self, client, mock_kg_logic):
        """Test community/global search QA."""
        mock_kg_logic.query_graph.return_value = {
            "answer": "GraphRAG is a technique for...",
            "reports_used": 3
        }
        
        response = client.post("/api/kg-falkor/qa", json={
            "question": "What is GraphRAG?",
            "namespace": "test_ns",
            "llm_key": "test-key",
            "engine": {"name": "qdrant", "index_name": "test_idx"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_graph_qa_hybrid(self, client, mock_kg_logic):
        """Test hybrid search QA."""
        mock_kg_logic.context_fusion_graph_search.return_value = {
            "answer": "Based on the graph...",
            "entities": [],
            "relationships": [],
            "scores": {},
            "expanded_nodes": []
        }
        
        response = client.post("/api/kg-falkor/hybrid", json={
            "question": "Tell me about AI",
            "namespace": "test_ns",
            "llm_key": "test-key",
            "engine": {"name": "qdrant", "index_name": "test_idx"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "entities" in data
        assert "retrieval_strategy" in data
    
    def test_multimodal_search(self, client, mock_kg_logic):
        """Test multimodal search."""
        mock_kg_logic.multimodal_search.return_value = {
            "answer": "Found information...",
            "sources": {"text_chunks": [], "tables": [], "images": []}
        }
        
        response = client.post("/api/kg-falkor/multimodal-search", json={
            "question": "Show me reports about sales",
            "namespace": "test_ns",
            "llm_key": "test-key",
            "engine": {"name": "qdrant", "index_name": "test_idx"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data


class TestGraphManagementEndpoints:
    """Test graph management endpoints."""
    
    def test_add_document(self, client, mock_kg_logic):
        """Test adding document to graph."""
        mock_kg_logic.add_document_to_graph.return_value = {
            "metadata_id": "doc_123",
            "chunks_processed": 5,
            "entities_extracted": 10,
            "entities_new": 8,
            "entities_reused": 2,
            "relationships_created": 15,
            "status": "success"
        }
        
        response = client.post("/api/kg-falkor/add-document", json={
            "metadata_id": "doc_123",
            "namespace": "test_ns",
            "engine": {"name": "qdrant", "index_name": "test_idx"},
            "llm_key": "test-key"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata_id"] == "doc_123"
        assert data["status"] == "success"
    
    def test_create_graph(self, client, mock_kg_logic):
        """Test creating a graph."""
        mock_kg_logic.create_graph.return_value = {
            "namespace": "test_ns",
            "chunks_processed": 100,
            "nodes_created": 250,
            "relationships_created": 500,
            "status": "success"
        }
        
        response = client.post("/api/kg-falkor/create", json={
            "namespace": "test_ns",
            "engine": {"name": "qdrant", "index_name": "test_idx"},
            "llm_key": "test-key"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["namespace"] == "test_ns"
        assert data["status"] == "success"
    
    def test_louvain_cluster(self, client, mock_kg_logic):
        """Test Louvain clustering."""
        mock_kg_logic.cluster_graph_louvain.return_value = {
            "status": "success",
            "communities_detected": 5,
            "reports_created": 5
        }
        
        response = client.post("/api/kg-falkor/louvain-cluster", json={
            "namespace": "test_ns",
            "engine": {"name": "qdrant", "index_name": "test_idx"},
            "llm_key": "test-key"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["communities_detected"] == 5
    
    def test_leiden_cluster(self, client, mock_kg_logic):
        """Test Leiden clustering."""
        mock_kg_logic.cluster_graph_leiden.return_value = {
            "status": "success",
            "communities_detected": 3,
            "reports_created": 3
        }
        
        response = client.post("/api/kg-falkor/leiden-cluster", json={
            "namespace": "test_ns",
            "engine": {"name": "qdrant", "index_name": "test_idx"},
            "llm_key": "test-key"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["communities_detected"] == 3
    
    def test_hierarchical_cluster(self, client, mock_kg_logic):
        """Test hierarchical clustering."""
        mock_kg_logic.cluster_graph_hierarchical.return_value = {
            "status": "success",
            "communities_detected": 8,
            "reports_created": 8
        }
        
        response = client.post("/api/kg-falkor/hierarchical", json={
            "namespace": "test_ns",
            "engine": {"name": "qdrant", "index_name": "test_idx"},
            "llm_key": "test-key"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["communities_detected"] == 8
    
    def test_community_analysis(self, client, mock_kg_logic):
        """Test community analysis."""
        mock_kg_logic.analyze_community.return_value = {
            "status": "success",
            "communities": 4,
            "reports": 4
        }
        
        response = client.post("/api/kg-falkor/community-analysis", json={
            "namespace": "test_ns",
            "engine": {"name": "qdrant", "index_name": "test_idx"},
            "llm_key": "test-key"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["communities_detected"] == 4


class TestAsyncTaskEndpoints:
    """Test async task endpoints."""
    
    def test_get_task_status_not_enabled(self, client):
        """Test task status when TaskIQ not enabled."""
        with patch('tilellm.modules.knowledge_graph_falkor.controllers.ENABLE_TASKIQ', False):
            response = client.get("/api/kg-falkor/tasks/task_123")
            
            assert response.status_code == 501
            assert "Async tasks not enabled" in response.json()["detail"]
    
    def test_get_task_status_success(self, client):
        """Test getting task status."""
        mock_result = Mock()
        mock_result.is_err = False
        mock_result.return_value = {"nodes_created": 10}
        
        with patch('tilellm.modules.knowledge_graph_falkor.controllers.ENABLE_TASKIQ', True), \
             patch('tilellm.modules.knowledge_graph_falkor.controllers.TASKIQ_AVAILABLE', True), \
             patch('tilellm.modules.knowledge_graph_falkor.controllers.broker') as mock_broker:
            
            mock_backend = Mock()
            mock_backend.is_result_ready = AsyncMock(return_value=True)
            mock_backend.get_result = AsyncMock(return_value=mock_result)
            mock_broker.result_backend = mock_backend
            
            response = client.get("/api/kg-falkor/tasks/task_123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == "task_123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
