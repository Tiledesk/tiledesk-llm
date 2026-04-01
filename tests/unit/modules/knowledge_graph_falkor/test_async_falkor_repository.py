"""
Unit tests for Async FalkorDB Repository.
Tests async CRUD operations with mocked FalkorDB client.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository import AsyncFalkorGraphRepository
from tilellm.modules.knowledge_graph_falkor.models import Node, Relationship


@pytest.fixture
def mock_falkor_client():
    """Fixture providing a mocked FalkorDB client."""
    client = Mock()
    graph = Mock()
    client.select_graph.return_value = graph
    return client, graph


@pytest.fixture
def reset_repository():
    """Reset repository class state before each test."""
    AsyncFalkorGraphRepository._client = None
    AsyncFalkorGraphRepository._pool = None
    AsyncFalkorGraphRepository._default_graph_name = "knowledge_graph"
    AsyncFalkorGraphRepository._indexed_graphs = set()


class TestAsyncFalkorRepositoryInit:
    """Test repository initialization."""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, mock_falkor_client, reset_repository):
        """Test successful repository initialization."""
        client, graph = mock_falkor_client
        
        with patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.FalkorDB', 
                   return_value=client), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.BlockingConnectionPool'), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.get_service_config') as mock_config:
            
            mock_config.return_value = {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test_graph"
                }
            }
            
            repo = AsyncFalkorGraphRepository()
            assert repo is not None
            assert AsyncFalkorGraphRepository._client is not None
            assert AsyncFalkorGraphRepository._default_graph_name == "test_graph"
    
    @pytest.mark.asyncio
    async def test_initialization_missing_config(self, reset_repository):
        """Test initialization fails without configuration."""
        with patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.get_service_config') as mock_config:
            mock_config.return_value = {}
            
            with pytest.raises(ValueError, match="FalkorDB configuration not found"):
                AsyncFalkorGraphRepository()
    
    @pytest.mark.asyncio
    async def test_initialization_import_error(self, reset_repository):
        """Test initialization fails when FalkorDB not installed."""
        with patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.FALKORDB_ASYNC_AVAILABLE', False):
            with pytest.raises(ImportError, match="FalkorDB async client not installed"):
                AsyncFalkorGraphRepository()


class TestAsyncFalkorRepositoryNodeOperations:
    """Test async node CRUD operations."""
    
    @pytest.fixture
    async def repo_with_mock(self, mock_falkor_client, reset_repository):
        """Create repository with mocked client."""
        client, graph = mock_falkor_client
        
        with patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.FalkorDB', 
                   return_value=client), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.BlockingConnectionPool'), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.get_service_config') as mock_config:
            
            mock_config.return_value = {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test_graph"
                }
            }
            
            repo = AsyncFalkorGraphRepository()
            yield repo, graph
    
    @pytest.mark.asyncio
    async def test_create_node(self, repo_with_mock):
        """Test creating a node."""
        repo, graph = await repo_with_mock.__anext__()
        
        # Mock query result
        mock_result = Mock()
        mock_result.header = ["id", "labels", "properties"]
        mock_result.result_set = [[1, ["Person"], {"name": "John", "namespace": "test"}]]
        graph.query = AsyncMock(return_value=mock_result)
        
        node = Node(label="Person", properties={"name": "John"})
        result = await repo.create_node(node, namespace="test")
        
        assert result is not None
        assert result.id == "1"
        assert result.label == "Person"
        assert result.properties.get("name") == "John"
    
    @pytest.mark.asyncio
    async def test_find_node_by_id(self, repo_with_mock):
        """Test finding a node by ID."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.header = ["id", "labels", "properties"]
        mock_result.result_set = [[123, ["Person"], {"name": "Jane"}]]
        graph.query = AsyncMock(return_value=mock_result)
        
        result = await repo.find_node_by_id("123")
        
        assert result is not None
        assert result.id == "123"
        assert result.label == "Person"
    
    @pytest.mark.asyncio
    async def test_find_node_by_id_not_found(self, repo_with_mock):
        """Test finding a non-existent node."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.header = ["id", "labels", "properties"]
        mock_result.result_set = []
        graph.query = AsyncMock(return_value=mock_result)
        
        result = await repo.find_node_by_id("999")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_find_nodes_by_label(self, repo_with_mock):
        """Test finding nodes by label."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.header = ["id", "labels", "properties"]
        mock_result.result_set = [
            [1, ["Person"], {"name": "Alice"}],
            [2, ["Person"], {"name": "Bob"}]
        ]
        graph.query = AsyncMock(return_value=mock_result)
        
        results = await repo.find_nodes_by_label("Person", limit=10, namespace="test")
        
        assert len(results) == 2
        assert all(n.label == "Person" for n in results)
    
    @pytest.mark.asyncio
    async def test_find_nodes_by_property(self, repo_with_mock):
        """Test finding nodes by property value."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.header = ["id", "labels", "properties"]
        mock_result.result_set = [[1, ["Person"], {"name": "Alice", "age": 30}]]
        graph.query = AsyncMock(return_value=mock_result)
        
        results = await repo.find_nodes_by_property("Person", "name", "Alice", namespace="test")
        
        assert len(results) == 1
        assert results[0].properties.get("name") == "Alice"
    
    @pytest.mark.asyncio
    async def test_update_node(self, repo_with_mock):
        """Test updating a node."""
        repo, graph = await repo_with_mock.__anext__()
        
        # First call returns existing node
        find_result = Mock()
        find_result.header = ["id", "labels", "properties"]
        find_result.result_set = [[1, ["Person"], {"name": "John"}]]
        
        # Second call returns updated node
        update_result = Mock()
        update_result.header = ["id", "labels", "properties"]
        update_result.result_set = [[1, ["Person"], {"name": "John", "age": 30}]]
        
        graph.query = AsyncMock(side_effect=[find_result, update_result, update_result])
        
        result = await repo.update_node("1", properties={"age": 30})
        
        assert result is not None
        assert result.properties.get("age") == 30
    
    @pytest.mark.asyncio
    async def test_delete_node(self, repo_with_mock):
        """Test deleting a node."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.header = ["deleted_count"]
        mock_result.result_set = [[1]]
        graph.query = AsyncMock(return_value=mock_result)
        
        result = await repo.delete_node("123", detach=True)
        
        assert result is True


class TestAsyncFalkorRepositoryRelationshipOperations:
    """Test async relationship CRUD operations."""
    
    @pytest.fixture
    async def repo_with_mock(self, mock_falkor_client, reset_repository):
        """Create repository with mocked client."""
        client, graph = mock_falkor_client
        
        with patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.FalkorDB', 
                   return_value=client), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.BlockingConnectionPool'), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.get_service_config') as mock_config:
            
            mock_config.return_value = {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test_graph"
                }
            }
            
            repo = AsyncFalkorGraphRepository()
            yield repo, graph
    
    @pytest.mark.asyncio
    async def test_create_relationship(self, repo_with_mock):
        """Test creating a relationship."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.header = ["id", "type", "properties", "source_id", "target_id"]
        mock_result.result_set = [[1, "KNOWS", {"since": "2020"}, 10, 20]]
        graph.query = AsyncMock(return_value=mock_result)
        
        rel = Relationship(source_id="10", target_id="20", type="KNOWS", properties={"since": "2020"})
        result = await repo.create_relationship(rel)
        
        assert result is not None
        assert result.id == "1"
        assert result.type == "KNOWS"
    
    @pytest.mark.asyncio
    async def test_find_relationship_by_id(self, repo_with_mock):
        """Test finding a relationship by ID."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.header = ["id", "type", "properties", "source_id", "target_id"]
        mock_result.result_set = [[5, "WORKS_WITH", {"department": "IT"}, 1, 2]]
        graph.query = AsyncMock(return_value=mock_result)
        
        result = await repo.find_relationship_by_id("5")
        
        assert result is not None
        assert result.id == "5"
        assert result.type == "WORKS_WITH"
    
    @pytest.mark.asyncio
    async def test_find_relationships_by_node(self, repo_with_mock):
        """Test finding relationships by node."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.header = ["id", "type", "properties", "source_id", "target_id"]
        mock_result.result_set = [
            [1, "KNOWS", {}, 10, 20],
            [2, "WORKS_WITH", {}, 10, 30]
        ]
        graph.query = AsyncMock(return_value=mock_result)
        
        results = await repo.find_relationships_by_node("10", direction="both")
        
        assert len(results) == 2
        assert all(r.source_id == "10" or r.target_id == "10" for r in results)


class TestAsyncFalkorRepositoryUtilityOperations:
    """Test async utility operations."""
    
    @pytest.fixture
    async def repo_with_mock(self, mock_falkor_client, reset_repository):
        """Create repository with mocked client."""
        client, graph = mock_falkor_client
        
        with patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.FalkorDB', 
                   return_value=client), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.BlockingConnectionPool'), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.get_service_config') as mock_config:
            
            mock_config.return_value = {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test_graph"
                }
            }
            
            repo = AsyncFalkorGraphRepository()
            yield repo, graph
    
    @pytest.mark.asyncio
    async def test_verify_connection_success(self, repo_with_mock):
        """Test successful connection verification."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.result_set = [[1]]
        graph.query = AsyncMock(return_value=mock_result)
        
        result = await repo.verify_connection()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_connection_failure(self, repo_with_mock):
        """Test connection verification failure."""
        repo, graph = await repo_with_mock.__anext__()
        
        graph.query = AsyncMock(side_effect=Exception("Connection refused"))
        
        result = await repo.verify_connection()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_database_info(self, repo_with_mock):
        """Test getting database statistics."""
        repo, graph = await repo_with_mock.__anext__()
        
        node_result = Mock()
        node_result.result_set = [[100]]
        
        rel_result = Mock()
        rel_result.result_set = [[50]]
        
        graph.query = AsyncMock(side_effect=[node_result, rel_result])
        
        result = await repo.get_database_info()
        
        assert result["node_count"] == 100
        assert result["relationship_count"] == 50
    
    @pytest.mark.asyncio
    async def test_delete_nodes_by_metadata(self, repo_with_mock):
        """Test deleting nodes by metadata."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.result_set = [[5]]
        graph.query = AsyncMock(return_value=mock_result)
        
        result = await repo.delete_nodes_by_metadata(
            namespace="test_ns",
            index_name="test_idx",
            engine_name="pinecone"
        )
        
        assert result["nodes_deleted"] == 5


class TestAsyncFalkorRepositorySearchOperations:
    """Test search operations."""
    
    @pytest.fixture
    async def repo_with_mock(self, mock_falkor_client, reset_repository):
        """Create repository with mocked client."""
        client, graph = mock_falkor_client
        
        with patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.FalkorDB', 
                   return_value=client), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.BlockingConnectionPool'), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.get_service_config') as mock_config:
            
            mock_config.return_value = {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test_graph"
                }
            }
            
            repo = AsyncFalkorGraphRepository()
            yield repo, graph
    
    @pytest.mark.asyncio
    async def test_search_nodes_by_text(self, repo_with_mock):
        """Test searching nodes by text."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.header = ["id", "labels", "properties"]
        mock_result.result_set = [
            [1, ["Person"], {"name": "Alice", "description": "Software engineer"}]
        ]
        graph.query = AsyncMock(return_value=mock_result)
        
        results = await repo.search_nodes_by_text("engineer", limit=10)
        
        assert len(results) == 1
        assert "engineer" in results[0].properties.get("description", "")


class TestAsyncFalkorRepositoryGraphManagement:
    """Test graph management operations."""
    
    @pytest.fixture
    async def repo_with_mock(self, mock_falkor_client, reset_repository):
        """Create repository with mocked client."""
        client, graph = mock_falkor_client
        
        with patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.FalkorDB', 
                   return_value=client), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.BlockingConnectionPool'), \
             patch('tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository.get_service_config') as mock_config:
            
            mock_config.return_value = {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test_graph"
                }
            }
            
            repo = AsyncFalkorGraphRepository()
            yield repo, graph
    
    @pytest.mark.asyncio
    async def test_delete_graph(self, repo_with_mock):
        """Test deleting an entire graph."""
        repo, graph = await repo_with_mock.__anext__()
        
        graph.delete = AsyncMock(return_value=None)
        
        result = await repo.delete_graph("test_namespace")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_ensure_indexes_async(self, repo_with_mock):
        """Test ensuring indexes are created."""
        repo, graph = await repo_with_mock.__anext__()
        
        mock_result = Mock()
        mock_result.result_set = []
        graph.query = AsyncMock(return_value=mock_result)
        
        await repo.ensure_indexes_async()
        
        # Should have attempted to create indexes
        assert graph.query.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
