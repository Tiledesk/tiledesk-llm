"""
Unit tests for Graph Service and GraphRAG Service.
Tests business logic with mocked repositories.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from tilellm.modules.knowledge_graph_falkor.services.services import GraphService, GraphRAGService
from tilellm.modules.knowledge_graph_falkor.models import Node, Relationship, NodeUpdate, RelationshipUpdate


@pytest.fixture
def mock_repository():
    """Create a mock repository for testing."""
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
    return repo


class TestGraphService:
    """Test suite for GraphService."""
    
    @pytest.fixture
    def graph_service(self, mock_repository):
        """Create a GraphService with mocked repository."""
        service = GraphService(repository=mock_repository)
        return service
    
    @pytest.mark.asyncio
    async def test_create_node_success(self, graph_service, mock_repository):
        """Test successful node creation."""
        mock_repo_node = Node(id="123", label="Person", properties={"name": "John"})
        mock_repository.create_node.return_value = mock_repo_node
        
        node = Node(label="Person", properties={"name": "John"})
        result = await graph_service.create_node(node, namespace="test")
        
        assert result is not None
        assert result.id == "123"
        assert result.label == "Person"
        mock_repository.create_node.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_node_validation_error(self, graph_service):
        """Test node creation with validation error (no label)."""
        node = Node(label="", properties={"name": "John"})
        
        with pytest.raises(ValueError, match="Node must have a label"):
            await graph_service.create_node(node)
    
    @pytest.mark.asyncio
    async def test_get_node_success(self, graph_service, mock_repository):
        """Test retrieving an existing node."""
        mock_node = Node(id="456", label="Organization", properties={"name": "Acme"})
        mock_repository.find_node_by_id.return_value = mock_node
        
        result = await graph_service.get_node("456")
        
        assert result is not None
        assert result.id == "456"
        assert result.label == "Organization"
    
    @pytest.mark.asyncio
    async def test_get_node_not_found(self, graph_service, mock_repository):
        """Test retrieving a non-existent node."""
        mock_repository.find_node_by_id.return_value = None
        
        result = await graph_service.get_node("999")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_nodes_by_label(self, graph_service, mock_repository):
        """Test retrieving nodes by label."""
        mock_nodes = [
            Node(id="1", label="Person", properties={"name": "Alice"}),
            Node(id="2", label="Person", properties={"name": "Bob"})
        ]
        mock_repository.find_nodes_by_label.return_value = mock_nodes
        
        results = await graph_service.get_nodes_by_label("Person", limit=10)
        
        assert len(results) == 2
        assert all(n.label == "Person" for n in results)
    
    @pytest.mark.asyncio
    async def test_search_nodes(self, graph_service, mock_repository):
        """Test searching nodes by property."""
        mock_nodes = [Node(id="1", label="Person", properties={"name": "Alice", "age": 30})]
        mock_repository.find_nodes_by_property.return_value = mock_nodes
        
        results = await graph_service.search_nodes("Person", "name", "Alice", limit=5)
        
        assert len(results) == 1
        assert results[0].properties.get("name") == "Alice"
    
    @pytest.mark.asyncio
    async def test_find_entity_by_name_and_type(self, graph_service, mock_repository):
        """Test finding entity by name and type."""
        mock_node = Node(id="1", label="PERSON", properties={"name": "John Doe"})
        mock_repository.find_nodes_by_property.return_value = [mock_node]
        
        result = await graph_service.find_entity_by_name_and_type("John Doe", "PERSON", namespace="test")
        
        assert result is not None
        assert result.properties.get("name") == "John Doe"
    
    @pytest.mark.asyncio
    async def test_update_node_success(self, graph_service, mock_repository):
        """Test successful node update."""
        existing = Node(id="1", label="Person", properties={"name": "John"})
        updated = Node(id="1", label="Person", properties={"name": "John", "age": 30})
        
        mock_repository.find_node_by_id.return_value = existing
        mock_repository.update_node.return_value = updated
        
        update = NodeUpdate(properties={"age": 30})
        result = await graph_service.update_node("1", update)
        
        assert result is not None
        assert result.properties.get("age") == 30
    
    @pytest.mark.asyncio
    async def test_update_node_not_found(self, graph_service, mock_repository):
        """Test updating a non-existent node."""
        mock_repository.find_node_by_id.return_value = None
        
        update = NodeUpdate(properties={"age": 30})
        result = await graph_service.update_node("999", update)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_node_success(self, graph_service, mock_repository):
        """Test successful node deletion."""
        existing = Node(id="1", label="Person", properties={"name": "John"})
        mock_repository.find_node_by_id.return_value = existing
        mock_repository.delete_node.return_value = True
        
        result = await graph_service.delete_node("1", detach=True)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_node_not_found(self, graph_service, mock_repository):
        """Test deleting a non-existent node."""
        mock_repository.find_node_by_id.return_value = None
        
        result = await graph_service.delete_node("999", detach=True)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_relationship_success(self, graph_service, mock_repository):
        """Test successful relationship creation."""
        source = Node(id="1", label="Person", properties={})
        target = Node(id="2", label="Person", properties={})
        mock_rel = Relationship(id="10", source_id="1", target_id="2", type="KNOWS", properties={})
        
        mock_repository.find_node_by_id.side_effect = [source, target]
        mock_repository.create_relationship.return_value = mock_rel
        
        rel = Relationship(source_id="1", target_id="2", type="KNOWS", properties={})
        result = await graph_service.create_relationship(rel, namespace="test")
        
        assert result is not None
        assert result.type == "KNOWS"
        assert result.source_id == "1"
        assert result.target_id == "2"
    
    @pytest.mark.asyncio
    async def test_create_relationship_missing_source(self, graph_service, mock_repository):
        """Test relationship creation with missing source node."""
        mock_repository.find_node_by_id.return_value = None
        
        rel = Relationship(source_id="999", target_id="2", type="KNOWS", properties={})
        
        with pytest.raises(RuntimeError, match="Source node.*does not exist"):
            await graph_service.create_relationship(rel)
    
    @pytest.mark.asyncio
    async def test_create_relationship_missing_target(self, graph_service, mock_repository):
        """Test relationship creation with missing target node."""
        source = Node(id="1", label="Person", properties={})
        mock_repository.find_node_by_id.side_effect = [source, None]
        
        rel = Relationship(source_id="1", target_id="999", type="KNOWS", properties={})
        
        with pytest.raises(RuntimeError, match="Target node.*does not exist"):
            await graph_service.create_relationship(rel)
    
    @pytest.mark.asyncio
    async def test_get_relationship(self, graph_service, mock_repository):
        """Test retrieving a relationship."""
        mock_rel = Relationship(id="5", source_id="1", target_id="2", type="WORKS_WITH", properties={})
        mock_repository.find_relationship_by_id.return_value = mock_rel
        
        result = await graph_service.get_relationship("5")
        
        assert result is not None
        assert result.id == "5"
        assert result.type == "WORKS_WITH"
    
    @pytest.mark.asyncio
    async def test_get_node_relationships(self, graph_service, mock_repository):
        """Test getting relationships for a node."""
        mock_rels = [
            Relationship(id="1", source_id="10", target_id="20", type="KNOWS", properties={}),
            Relationship(id="2", source_id="30", target_id="10", type="WORKS_WITH", properties={})
        ]
        mock_repository.find_relationships_by_node.return_value = mock_rels
        
        results = await graph_service.get_node_relationships("10", direction="both")
        
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_get_node_relationships_invalid_direction(self, graph_service):
        """Test getting relationships with invalid direction."""
        with pytest.raises(ValueError, match="Invalid direction"):
            await graph_service.get_node_relationships("10", direction="invalid")
    
    @pytest.mark.asyncio
    async def test_verify_connection(self, graph_service, mock_repository):
        """Test connection verification."""
        mock_repository.verify_connection.return_value = True
        
        result = await graph_service.verify_connection()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_database_stats(self, graph_service, mock_repository):
        """Test getting database statistics."""
        mock_stats = {"node_count": 100, "relationship_count": 50, "graph_name": "test"}
        mock_repository.get_database_info.return_value = mock_stats
        
        result = await graph_service.get_database_stats()
        
        assert result["node_count"] == 100
        assert result["relationship_count"] == 50


class TestGraphRAGService:
    """Test suite for GraphRAGService."""
    
    @pytest.fixture
    def mock_graph_service(self):
        """Create a mock GraphService."""
        service = Mock()
        service.create_node = AsyncMock()
        service.find_entity_by_name_and_type = AsyncMock()
        service._get_repository = Mock()
        service._get_repository.return_value = Mock()
        return service
    
    @pytest.fixture
    def graph_rag_service(self, mock_graph_service):
        """Create a GraphRAGService with mocked dependencies."""
        return GraphRAGService(graph_service=mock_graph_service)
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_graph_service):
        """Test service initialization."""
        service = GraphRAGService(
            graph_service=mock_graph_service,
            llm=Mock(),
            vector_store_repository=Mock()
        )
        
        assert service.graph_service == mock_graph_service
        assert service.llm is not None
        assert service.vector_store_repository is not None
    
    @pytest.mark.asyncio
    async def test_set_llm(self, graph_rag_service):
        """Test setting LLM."""
        mock_llm = Mock()
        graph_rag_service.set_llm(mock_llm)
        
        assert graph_rag_service.llm == mock_llm
    
    @pytest.mark.asyncio
    async def test_set_vector_store_repository(self, graph_rag_service):
        """Test setting vector store repository."""
        mock_repo = Mock()
        graph_rag_service.set_vector_store_repository(mock_repo)
        
        assert graph_rag_service.vector_store_repository == mock_repo


class TestGraphServiceWithoutRepository:
    """Test GraphService when repository is not set."""
    
    def test_init_without_repository(self):
        """Test initialization without repository."""
        service = GraphService(repository=None)
        assert service.repository is None
    
    def test_set_repository(self):
        """Test setting repository after initialization."""
        service = GraphService(repository=None)
        mock_repo = Mock()
        
        service.set_repository(mock_repo)
        
        assert service.repository == mock_repo
    
    @pytest.mark.asyncio
    async def test_operation_without_repository(self):
        """Test operation fails without repository."""
        service = GraphService(repository=None)
        
        with pytest.raises(RuntimeError, match="Repository not initialized"):
            await service.get_node("123")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
