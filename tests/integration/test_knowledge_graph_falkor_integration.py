"""
Integration tests for Knowledge Graph Falkor module.
Tests integration between repository, services, and controllers.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from tilellm.modules.knowledge_graph_falkor.models import Node, Relationship
from tilellm.modules.knowledge_graph_falkor.services.services import GraphService, GraphRAGService
from tilellm.modules.knowledge_graph_falkor.controllers import router


@pytest.fixture
def mock_falkor_repository():
    """Create a fully mocked FalkorDB repository."""
    repo = Mock()
    repo.create_node = AsyncMock()
    repo.find_node_by_id = AsyncMock()
    repo.find_nodes_by_label = AsyncMock()
    repo.find_nodes_by_property = AsyncMock()
    repo.update_node = AsyncMock()
    repo.delete_node = AsyncMock()
    repo.create_relationship = AsyncMock()
    repo.find_relationship_by_id = AsyncMock()
    repo.find_relationships_by_node = AsyncMock()
    repo.update_relationship = AsyncMock()
    repo.delete_relationship = AsyncMock()
    repo.verify_connection = AsyncMock()
    repo.get_database_info = AsyncMock()
    repo.delete_nodes_by_metadata = AsyncMock()
    repo.delete_graph = AsyncMock()
    return repo


@pytest.fixture
def mock_graph_service(mock_falkor_repository):
    """Create GraphService with mocked repository."""
    return GraphService(repository=mock_falkor_repository)


class TestNodeOperationsIntegration:
    """Test node operations end-to-end."""
    
    @pytest.mark.asyncio
    async def test_create_and_retrieve_node(self, mock_graph_service, mock_falkor_repository):
        """Test creating and retrieving a node."""
        # Setup mock
        created_node = Node(id="123", label="Person", properties={"name": "John", "namespace": "test"})
        mock_falkor_repository.create_node.return_value = created_node
        mock_falkor_repository.find_node_by_id.return_value = created_node
        
        # Create node
        node = Node(label="Person", properties={"name": "John"})
        result = await mock_graph_service.create_node(node, namespace="test")
        
        assert result.id == "123"
        
        # Retrieve node
        retrieved = await mock_graph_service.get_node("123")
        
        assert retrieved is not None
        assert retrieved.id == "123"
        assert retrieved.properties["name"] == "John"
    
    @pytest.mark.asyncio
    async def test_update_node_flow(self, mock_graph_service, mock_falkor_repository):
        """Test full update node flow."""
        # Setup mocks
        existing = Node(id="1", label="Person", properties={"name": "John"})
        updated = Node(id="1", label="Person", properties={"name": "John", "age": 30})
        
        mock_falkor_repository.find_node_by_id.side_effect = [existing, updated]
        mock_falkor_repository.update_node.return_value = updated
        
        # Update node
        from tilellm.modules.knowledge_graph_falkor.models import NodeUpdate
        update = NodeUpdate(properties={"age": 30})
        result = await mock_graph_service.update_node("1", update)
        
        assert result is not None
        assert result.properties["age"] == 30
    
    @pytest.mark.asyncio
    async def test_delete_node_flow(self, mock_graph_service, mock_falkor_repository):
        """Test full delete node flow."""
        existing = Node(id="1", label="Person", properties={"name": "John"})
        mock_falkor_repository.find_node_by_id.return_value = existing
        mock_falkor_repository.delete_node.return_value = True
        
        result = await mock_graph_service.delete_node("1", detach=True)
        
        assert result is True
        mock_falkor_repository.delete_node.assert_called_once_with("1", True)


class TestRelationshipOperationsIntegration:
    """Test relationship operations end-to-end."""
    
    @pytest.mark.asyncio
    async def test_create_relationship_flow(self, mock_graph_service, mock_falkor_repository):
        """Test full relationship creation flow."""
        # Setup mocks
        source = Node(id="10", label="Person", properties={"name": "Alice"})
        target = Node(id="20", label="Person", properties={"name": "Bob"})
        created_rel = Relationship(
            id="100",
            source_id="10",
            target_id="20",
            type="KNOWS",
            properties={"since": "2020"}
        )
        
        mock_falkor_repository.find_node_by_id.side_effect = [source, target]
        mock_falkor_repository.create_relationship.return_value = created_rel
        
        # Create relationship
        rel = Relationship(source_id="10", target_id="20", type="KNOWS", properties={"since": "2020"})
        result = await mock_graph_service.create_relationship(rel)
        
        assert result is not None
        assert result.id == "100"
        assert result.type == "KNOWS"
    
    @pytest.mark.asyncio
    async def test_node_relationships_flow(self, mock_graph_service, mock_falkor_repository):
        """Test getting node relationships flow."""
        rels = [
            Relationship(id="1", source_id="10", target_id="20", type="KNOWS", properties={}),
            Relationship(id="2", source_id="30", target_id="10", type="WORKS_WITH", properties={})
        ]
        mock_falkor_repository.find_relationships_by_node.return_value = rels
        
        results = await mock_graph_service.get_node_relationships("10", direction="both")
        
        assert len(results) == 2
        mock_falkor_repository.find_relationships_by_node.assert_called_once_with("10", "both")


class TestGraphServiceIntegration:
    """Test GraphService integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_entity_deduplication_flow(self, mock_graph_service, mock_falkor_repository):
        """Test entity deduplication by name and type."""
        existing = Node(id="5", label="PERSON", properties={"name": "John Doe"})
        mock_falkor_repository.find_nodes_by_property.return_value = [existing]
        
        result = await mock_graph_service.find_entity_by_name_and_type(
            "John Doe",
            "PERSON",
            namespace="test"
        )
        
        assert result is not None
        assert result.id == "5"
    
    @pytest.mark.asyncio
    async def test_search_nodes_flow(self, mock_graph_service, mock_falkor_repository):
        """Test search nodes flow."""
        nodes = [
            Node(id="1", label="Person", properties={"name": "Alice"}),
            Node(id="2", label="Person", properties={"name": "Alex"})
        ]
        mock_falkor_repository.find_nodes_by_property.return_value = nodes
        
        results = await mock_graph_service.search_nodes(
            "Person",
            "name",
            "Al",
            limit=10,
            namespace="test"
        )
        
        assert len(results) == 2


class TestRepositoryConnectionIntegration:
    """Test repository connection integration."""
    
    @pytest.mark.asyncio
    async def test_verify_connection_integration(self, mock_graph_service, mock_falkor_repository):
        """Test connection verification through service."""
        mock_falkor_repository.verify_connection.return_value = True
        
        result = await mock_graph_service.verify_connection()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_stats_integration(self, mock_graph_service, mock_falkor_repository):
        """Test getting database stats through service."""
        mock_stats = {
            "node_count": 100,
            "relationship_count": 50,
            "graph_name": "test_graph"
        }
        mock_falkor_repository.get_database_info.return_value = mock_stats
        
        result = await mock_graph_service.get_database_stats()
        
        assert result["node_count"] == 100
        assert result["relationship_count"] == 50


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_create_relationship_missing_nodes(self, mock_graph_service, mock_falkor_repository):
        """Test error when creating relationship with missing nodes."""
        mock_falkor_repository.find_node_by_id.return_value = None
        
        rel = Relationship(source_id="999", target_id="20", type="KNOWS", properties={})
        
        with pytest.raises(RuntimeError, match="does not exist"):
            await mock_graph_service.create_relationship(rel)
    
    @pytest.mark.asyncio
    async def test_invalid_direction(self, mock_graph_service):
        """Test error with invalid relationship direction."""
        with pytest.raises(ValueError, match="Invalid direction"):
            await mock_graph_service.get_node_relationships("10", direction="invalid")
    
    @pytest.mark.asyncio
    async def test_repository_error_propagation(self, mock_graph_service, mock_falkor_repository):
        """Test that repository errors are propagated."""
        mock_falkor_repository.find_node_by_id.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            await mock_graph_service.get_node("123")


class TestComplexWorkflows:
    """Test complex multi-step workflows."""
    
    @pytest.mark.asyncio
    async def test_full_crud_workflow(self, mock_graph_service, mock_falkor_repository):
        """Test full CRUD workflow."""
        # Create
        node = Node(label="Person", properties={"name": "Alice"})
        created = Node(id="1", label="Person", properties={"name": "Alice", "namespace": "test"})
        mock_falkor_repository.create_node.return_value = created
        
        result = await mock_graph_service.create_node(node, namespace="test")
        assert result.id == "1"
        
        # Read
        mock_falkor_repository.find_node_by_id.return_value = created
        read = await mock_graph_service.get_node("1")
        assert read.properties["name"] == "Alice"
        
        # Update
        from tilellm.modules.knowledge_graph_falkor.models import NodeUpdate
        updated = Node(id="1", label="Person", properties={"name": "Alice Smith", "namespace": "test"})
        mock_falkor_repository.find_node_by_id.side_effect = [created, updated]
        mock_falkor_repository.update_node.return_value = updated
        
        update_result = await mock_graph_service.update_node("1", NodeUpdate(properties={"name": "Alice Smith"}))
        assert update_result.properties["name"] == "Alice Smith"
        
        # Delete
        mock_falkor_repository.find_node_by_id.return_value = updated
        mock_falkor_repository.delete_node.return_value = True
        
        delete_result = await mock_graph_service.delete_node("1", detach=True)
        assert delete_result is True
    
    @pytest.mark.asyncio
    async def test_bulk_operations_workflow(self, mock_graph_service, mock_falkor_repository):
        """Test bulk operations workflow."""
        # Get nodes by label
        nodes = [
            Node(id=str(i), label="Person", properties={"name": f"Person{i}"})
            for i in range(5)
        ]
        mock_falkor_repository.find_nodes_by_label.return_value = nodes
        
        results = await mock_graph_service.get_nodes_by_label("Person", limit=10)
        
        assert len(results) == 5
        
        # Get relationships for each node
        rels = [
            Relationship(id="r1", source_id="0", target_id="1", type="KNOWS", properties={})
        ]
        mock_falkor_repository.find_relationships_by_node.return_value = rels
        
        all_rels = []
        for node in results:
            node_rels = await mock_graph_service.get_node_relationships(node.id, "both")
            all_rels.extend(node_rels)
        
        assert len(all_rels) == 5  # Called for each of 5 nodes


class TestGraphRAGServiceIntegration:
    """Test GraphRAGService integration."""
    
    @pytest.mark.asyncio
    async def test_graph_rag_service_with_mocked_dependencies(self):
        """Test GraphRAGService with all mocked dependencies."""
        mock_graph_service = Mock()
        mock_llm = Mock()
        mock_vector_repo = Mock()
        
        service = GraphRAGService(
            graph_service=mock_graph_service,
            llm=mock_llm,
            vector_store_repository=mock_vector_repo
        )
        
        # Test initialization
        assert service.graph_service == mock_graph_service
        assert service.llm == mock_llm
        assert service.vector_store_repository == mock_vector_repo
        
        # Test setter methods
        new_llm = Mock()
        service.set_llm(new_llm)
        assert service.llm == new_llm
        
        new_repo = Mock()
        service.set_vector_store_repository(new_repo)
        assert service.vector_store_repository == new_repo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
