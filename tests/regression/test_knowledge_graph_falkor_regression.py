"""
Regression tests for Knowledge Graph Falkor critical workflows.
These tests ensure that core functionality continues to work after changes.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from tilellm.modules.knowledge_graph_falkor.models import Node, Relationship


class TestCriticalNodeWorkflows:
    """
    Regression tests for node operations.
    These workflows are critical for graph management.
    """
    
    @pytest.fixture
    def mock_services(self):
        """Setup mocked services for regression tests."""
        with patch('tilellm.modules.knowledge_graph_falkor.controllers.kg_logic') as mock_logic:
            mock_logic.create_node = AsyncMock()
            mock_logic.get_node = AsyncMock()
            mock_logic.update_node = AsyncMock()
            mock_logic.delete_node = AsyncMock()
            mock_logic.get_nodes_by_label = AsyncMock()
            mock_logic.search_nodes = AsyncMock()
            yield mock_logic
    
    def test_node_crud_regression(self, mock_services):
        """
        REGRESSION TEST: Node CRUD operations must work end-to-end.
        
        This test ensures that the basic node lifecycle (Create, Read, Update, Delete)
        continues to work after any changes to the codebase.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        # Setup mocks for full CRUD flow
        created_node = Node(id="100", label="Person", properties={"name": "Test", "namespace": "regression"})
        updated_node = Node(id="100", label="Person", properties={"name": "Updated", "namespace": "regression"})
        
        mock_services.create_node.return_value = created_node
        mock_services.get_node.side_effect = [created_node, updated_node, None]  # After delete returns None
        mock_services.update_node.return_value = updated_node
        mock_services.delete_node.return_value = True
        
        with TestClient(app) as client:
            # CREATE
            response = client.post("/api/kg-falkor/nodes", json={
                "label": "Person",
                "properties": {"name": "Test"}
            })
            assert response.status_code == 201, "CREATE node failed - REGRESSION"
            node_id = response.json()["id"]
            
            # READ
            response = client.get(f"/api/kg-falkor/nodes/{node_id}")
            assert response.status_code == 200, "READ node failed - REGRESSION"
            assert response.json()["properties"]["name"] == "Test"
            
            # UPDATE
            response = client.put(f"/api/kg-falkor/nodes/{node_id}", json={
                "properties": {"name": "Updated"}
            })
            assert response.status_code == 200, "UPDATE node failed - REGRESSION"
            assert response.json()["properties"]["name"] == "Updated"
            
            # DELETE
            response = client.delete(f"/api/kg-falkor/nodes/{node_id}")
            assert response.status_code == 204, "DELETE node failed - REGRESSION"
    
    def test_node_label_filtering_regression(self, mock_services):
        """
        REGRESSION TEST: Node label filtering must work correctly.
        
        This ensures that filtering nodes by label returns correct results.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        nodes = [
            Node(id="1", label="Person", properties={"name": "Alice"}),
            Node(id="2", label="Person", properties={"name": "Bob"}),
            Node(id="3", label="Organization", properties={"name": "Acme"})
        ]
        mock_services.get_nodes_by_label.return_value = [n for n in nodes if n.label == "Person"]
        
        with TestClient(app) as client:
            response = client.get("/api/kg-falkor/nodes?label=Person")
            
            assert response.status_code == 200, "Label filtering failed - REGRESSION"
            data = response.json()
            assert len(data) == 2, "Wrong number of nodes returned - REGRESSION"
            assert all(n["label"] == "Person" for n in data), "Wrong labels returned - REGRESSION"


class TestCriticalRelationshipWorkflows:
    """
    Regression tests for relationship operations.
    Relationships are essential for graph structure.
    """
    
    @pytest.fixture
    def mock_services(self):
        """Setup mocked services for relationship tests."""
        with patch('tilellm.modules.knowledge_graph_falkor.controllers.kg_logic') as mock_logic:
            mock_logic.create_relationship = AsyncMock()
            mock_logic.get_relationship = AsyncMock()
            mock_logic.get_node_relationships = AsyncMock()
            mock_logic.update_relationship = AsyncMock()
            mock_logic.delete_relationship = AsyncMock()
            yield mock_logic
    
    def test_relationship_crud_regression(self, mock_services):
        """
        REGRESSION TEST: Relationship CRUD must work end-to-end.
        
        Relationships connect nodes and are fundamental to graph queries.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        created_rel = Relationship(
            id="500",
            source_id="10",
            target_id="20",
            type="KNOWS",
            properties={"since": "2020"}
        )
        updated_rel = Relationship(
            id="500",
            source_id="10",
            target_id="20",
            type="FRIENDS_WITH",
            properties={"strength": 0.9}
        )
        
        mock_services.create_relationship.return_value = created_rel
        mock_services.get_relationship.side_effect = [created_rel, updated_rel]
        mock_services.update_relationship.return_value = updated_rel
        mock_services.delete_relationship.return_value = True
        
        with TestClient(app) as client:
            # CREATE
            response = client.post("/api/kg-falkor/relationships", json={
                "source_id": "10",
                "target_id": "20",
                "type": "KNOWS",
                "properties": {"since": "2020"}
            })
            assert response.status_code == 201, "CREATE relationship failed - REGRESSION"
            rel_id = response.json()["id"]
            
            # READ
            response = client.get(f"/api/kg-falkor/relationships/{rel_id}")
            assert response.status_code == 200, "READ relationship failed - REGRESSION"
            
            # UPDATE
            response = client.put(f"/api/kg-falkor/relationships/{rel_id}", json={
                "type": "FRIENDS_WITH",
                "properties": {"strength": 0.9}
            })
            assert response.status_code == 200, "UPDATE relationship failed - REGRESSION"
            
            # DELETE
            response = client.delete(f"/api/kg-falkor/relationships/{rel_id}")
            assert response.status_code == 204, "DELETE relationship failed - REGRESSION"


class TestCriticalSearchWorkflows:
    """
    Regression tests for search operations.
    Search is critical for QA and retrieval.
    """
    
    @pytest.fixture
    def mock_services(self):
        """Setup mocked services for search tests."""
        with patch('tilellm.modules.knowledge_graph_falkor.controllers.kg_logic') as mock_logic:
            mock_logic.search_nodes = AsyncMock()
            mock_logic.query_graph = AsyncMock()
            mock_logic.context_fusion_graph_search = AsyncMock()
            mock_services = mock_logic
            yield mock_services
    
    def test_node_search_regression(self, mock_services):
        """
        REGRESSION TEST: Node search by property must work.
        
        This is critical for finding entities in the graph.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        results = [
            Node(id="1", label="Person", properties={"name": "Alice"}),
            Node(id="2", label="Person", properties={"name": "Alex"})
        ]
        mock_services.search_nodes.return_value = results
        
        with TestClient(app) as client:
            response = client.get(
                "/api/kg-falkor/nodes/search?label=Person&property_key=name&property_value=Al"
            )
            
            assert response.status_code == 200, "Node search failed - REGRESSION"
            data = response.json()
            assert len(data) >= 1, "Search returned no results - REGRESSION"
    
    def test_community_qa_regression(self, mock_services):
        """
        REGRESSION TEST: Community QA endpoint must work.
        
        This is the primary query endpoint for global search.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        mock_services.query_graph.return_value = {
            "answer": "GraphRAG is a retrieval technique...",
            "reports_used": 5
        }
        
        with TestClient(app) as client:
            response = client.post("/api/kg-falkor/qa", json={
                "question": "What is GraphRAG?",
                "namespace": "test_ns",
                "llm_key": "test-key",
                "engine": {"name": "qdrant", "index_name": "idx"}
            })
            
            assert response.status_code == 200, "Community QA failed - REGRESSION"
            data = response.json()
            assert "answer" in data, "QA response missing answer - REGRESSION"
    
    def test_hybrid_search_regression(self, mock_services):
        """
        REGRESSION TEST: Hybrid search endpoint must work.
        
        Hybrid search combines multiple retrieval strategies.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        mock_services.context_fusion_graph_search.return_value = {
            "answer": "Found information...",
            "entities": [{"name": "AI"}],
            "relationships": [],
            "scores": {"vector": 0.8},
            "expanded_nodes": []
        }
        
        with TestClient(app) as client:
            response = client.post("/api/kg-falkor/hybrid", json={
                "question": "Tell me about AI",
                "namespace": "test_ns",
                "llm_key": "test-key",
                "engine": {"name": "qdrant", "index_name": "idx"}
            })
            
            assert response.status_code == 200, "Hybrid search failed - REGRESSION"
            data = response.json()
            assert "answer" in data, "Hybrid response missing answer - REGRESSION"
            assert "retrieval_strategy" in data, "Hybrid response missing strategy - REGRESSION"


class TestCriticalGraphManagementWorkflows:
    """
    Regression tests for graph management operations.
    These operations manage the lifecycle of knowledge graphs.
    """
    
    @pytest.fixture
    def mock_services(self):
        """Setup mocked services for graph management tests."""
        with patch('tilellm.modules.knowledge_graph_falkor.controllers.kg_logic') as mock_logic:
            mock_logic.add_document_to_graph = AsyncMock()
            mock_logic.create_graph = AsyncMock()
            mock_logic.cluster_graph_louvain = AsyncMock()
            mock_logic.cluster_graph_leiden = AsyncMock()
            mock_logic.cluster_graph_hierarchical = AsyncMock()
            mock_logic.get_graph_network = AsyncMock()
            yield mock_logic
    
    def test_add_document_regression(self, mock_services):
        """
        REGRESSION TEST: Add document endpoint must work.
        
        Adding documents is critical for incremental graph updates.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        mock_services.add_document_to_graph.return_value = {
            "metadata_id": "doc_123",
            "chunks_processed": 5,
            "entities_extracted": 10,
            "entities_new": 8,
            "entities_reused": 2,
            "relationships_created": 15,
            "status": "success"
        }
        
        with TestClient(app) as client:
            response = client.post("/api/kg-falkor/add-document", json={
                "metadata_id": "doc_123",
                "namespace": "test_ns",
                "engine": {"name": "qdrant", "index_name": "idx"},
                "llm_key": "test-key"
            })
            
            assert response.status_code == 200, "Add document failed - REGRESSION"
            data = response.json()
            assert data["status"] == "success", "Add document returned wrong status - REGRESSION"
    
    def test_create_graph_regression(self, mock_services):
        """
        REGRESSION TEST: Create graph endpoint must work.
        
        Graph creation is the foundation of the knowledge graph.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        mock_services.create_graph.return_value = {
            "namespace": "test_ns",
            "chunks_processed": 100,
            "nodes_created": 250,
            "relationships_created": 500,
            "status": "success"
        }
        
        with TestClient(app) as client:
            response = client.post("/api/kg-falkor/create", json={
                "namespace": "test_ns",
                "engine": {"name": "qdrant", "index_name": "idx"},
                "llm_key": "test-key"
            })
            
            assert response.status_code == 200, "Create graph failed - REGRESSION"
            data = response.json()
            assert data["status"] == "success", "Create graph returned wrong status - REGRESSION"
    
    def test_clustering_regression(self, mock_services):
        """
        REGRESSION TEST: Clustering endpoints must work.
        
        Clustering generates community reports for global search.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        cluster_result = {
            "status": "success",
            "communities_detected": 5,
            "reports_created": 5
        }
        
        mock_services.cluster_graph_louvain.return_value = cluster_result
        mock_services.cluster_graph_leiden.return_value = cluster_result
        mock_services.cluster_graph_hierarchical.return_value = cluster_result
        
        with TestClient(app) as client:
            # Test Louvain
            response = client.post("/api/kg-falkor/louvain-cluster", json={
                "namespace": "test_ns",
                "engine": {"name": "qdrant", "index_name": "idx"},
                "llm_key": "test-key"
            })
            assert response.status_code == 200, "Louvain clustering failed - REGRESSION"
            
            # Test Leiden
            response = client.post("/api/kg-falkor/leiden-cluster", json={
                "namespace": "test_ns",
                "engine": {"name": "qdrant", "index_name": "idx"},
                "llm_key": "test-key"
            })
            assert response.status_code == 200, "Leiden clustering failed - REGRESSION"
            
            # Test Hierarchical
            response = client.post("/api/kg-falkor/hierarchical", json={
                "namespace": "test_ns",
                "engine": {"name": "qdrant", "index_name": "idx"},
                "llm_key": "test-key"
            })
            assert response.status_code == 200, "Hierarchical clustering failed - REGRESSION"
    
    def test_graph_network_regression(self, mock_services):
        """
        REGRESSION TEST: Graph network endpoint must work.
        
        This provides visualization data for the graph.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        mock_services.get_graph_network.return_value = {
            "nodes": [
                {"id": "1", "label": "Person", "properties": {"name": "Alice"}}
            ],
            "relationships": [
                {"id": "r1", "source_id": "1", "target_id": "2", "type": "KNOWS", "properties": {}}
            ],
            "stats": {
                "node_count": 1,
                "relationship_count": 1
            }
        }
        
        with TestClient(app) as client:
            response = client.get("/api/kg-falkor/network?namespace=test")
            
            assert response.status_code == 200, "Graph network failed - REGRESSION"
            data = response.json()
            assert "nodes" in data, "Network missing nodes - REGRESSION"
            assert "relationships" in data, "Network missing relationships - REGRESSION"


class TestCriticalHealthWorkflows:
    """
    Regression tests for health and connectivity.
    Health checks are critical for monitoring.
    """
    
    @pytest.fixture
    def mock_services(self):
        """Setup mocked services for health tests."""
        with patch('tilellm.modules.knowledge_graph_falkor.controllers.kg_logic') as mock_logic:
            mock_logic.check_health = Mock()
            mock_logic.get_stats = Mock()
            yield mock_logic
    
    def test_health_check_regression(self, mock_services):
        """
        REGRESSION TEST: Health check endpoint must work.
        
        This is used by load balancers and monitoring systems.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        mock_services.check_health.return_value = {"status": "healthy", "database": "connected"}
        
        with TestClient(app) as client:
            response = client.get("/api/kg-falkor/health")
            
            assert response.status_code == 200, "Health check failed - REGRESSION"
            data = response.json()
            assert data["status"] == "healthy", "Health check wrong status - REGRESSION"
    
    def test_stats_endpoint_regression(self, mock_services):
        """
        REGRESSION TEST: Stats endpoint must work.
        
        This provides metrics for monitoring and debugging.
        """
        from fastapi import FastAPI
        from tilellm.modules.knowledge_graph_falkor.controllers import router
        
        app = FastAPI()
        app.include_router(router)
        
        mock_services.get_stats.return_value = {
            "node_count": 1000,
            "relationship_count": 500
        }
        
        with TestClient(app) as client:
            response = client.get("/api/kg-falkor/stats")
            
            assert response.status_code == 200, "Stats endpoint failed - REGRESSION"
            data = response.json()
            assert "node_count" in data, "Stats missing node_count - REGRESSION"


class TestDataIntegrity:
    """
    Regression tests for data integrity.
    These ensure that data transformations preserve information.
    """
    
    def test_node_serialization_integrity(self):
        """
        REGRESSION TEST: Node serialization must preserve all data.
        """
        from tilellm.modules.knowledge_graph_falkor.models import Node
        
        original = Node(
            id="123",
            label="Person",
            properties={
                "name": "Alice",
                "age": 30,
                "tags": ["developer", "ai"],
                "metadata": {"source": "import"}
            }
        )
        
        # Serialize and deserialize
        json_data = original.model_dump()
        restored = Node(**json_data)
        
        assert restored.id == original.id, "ID not preserved - REGRESSION"
        assert restored.label == original.label, "Label not preserved - REGRESSION"
        assert restored.properties == original.properties, "Properties not preserved - REGRESSION"
    
    def test_relationship_serialization_integrity(self):
        """
        REGRESSION TEST: Relationship serialization must preserve all data.
        """
        from tilellm.modules.knowledge_graph_falkor.models import Relationship
        
        original = Relationship(
            id="500",
            source_id="10",
            target_id="20",
            type="KNOWS",
            properties={"since": "2020", "strength": 0.9}
        )
        
        # Serialize and deserialize
        json_data = original.model_dump()
        restored = Relationship(**json_data)
        
        assert restored.id == original.id, "ID not preserved - REGRESSION"
        assert restored.source_id == original.source_id, "Source not preserved - REGRESSION"
        assert restored.target_id == original.target_id, "Target not preserved - REGRESSION"
        assert restored.type == original.type, "Type not preserved - REGRESSION"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
