"""
Unit tests for FalkorDB repository.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from tilellm.modules.knowledge_graph_falkor.repository.falkor_repository import FalkorGraphRepository


class TestFalkorGraphRepository:
    """Test suite for FalkorGraphRepository."""
    
    def setup_method(self):
        """Reset class variables before each test."""
        FalkorGraphRepository._client = None
        FalkorGraphRepository._graph = None
        FalkorGraphRepository._graph_name = "knowledge_graph"
    
    def test_repository_initialization(self):
        """Test that repository can be initialized with mock connection."""
        with patch('tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.Redis') as mock_redis, \
             patch('tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.Graph') as mock_graph, \
             patch.object(FalkorGraphRepository, 'ensure_indexes') as mock_ensure:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_graph_instance = Mock()
            mock_graph.return_value = mock_graph_instance
            repo = FalkorGraphRepository()
            # Should not raise
            assert repo is not None
            # Graph name should be set
            assert FalkorGraphRepository._graph_name == "knowledge_graph"
            mock_ensure.assert_called_once()
    
    def test_verify_connection(self):
        """Test verify_connection returns True when ping succeeds."""
        with patch('tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.Redis') as mock_redis, \
             patch.object(FalkorGraphRepository, 'ensure_indexes'):
            mock_client = Mock()
            mock_client.ping = Mock(return_value=True)
            mock_redis.return_value = mock_client
            repo = FalkorGraphRepository()
            result = repo.verify_connection()
            assert result is True
            # ping called twice: once during init, once during verify_connection
            assert mock_client.ping.call_count == 2
    
    def test_verify_connection_failure(self):
        """Test verify_connection returns False when ping raises exception after init."""
        with patch('tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.Redis') as mock_redis, \
             patch.object(FalkorGraphRepository, 'ensure_indexes'):
            mock_client = Mock()
            # First ping succeeds (during init), second ping fails
            mock_client.ping = Mock(side_effect=[True, Exception("Connection failed")])
            mock_redis.return_value = mock_client
            repo = FalkorGraphRepository()
            result = repo.verify_connection()
            assert result is False
            assert mock_client.ping.call_count == 2
    
    def test_execute_query_with_graph_api(self):
        """Test execute_query calls Redis graph command using Graph API."""
        with patch('tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.Redis') as mock_redis, \
             patch('tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.Graph') as mock_graph, \
             patch.object(FalkorGraphRepository, 'ensure_indexes'):
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_graph_instance = Mock()
            mock_graph.return_value = mock_graph_instance
            # Mock query result
            mock_result = Mock()
            mock_result.result_set = []
            mock_result.header = []
            mock_graph_instance.query.return_value = mock_result
            
            repo = FalkorGraphRepository()
            result = repo.execute_query("MATCH (n) RETURN n")
            # Ensure our query was called (may be called after index queries, but we patched ensure_indexes)
            mock_graph_instance.query.assert_called_with("MATCH (n) RETURN n", params={})
            assert result == []
    
    def test_execute_query_with_raw_command(self):
        """Test execute_query falls back to raw Redis command when Graph API unavailable."""
        with patch('tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.Redis') as mock_redis, \
             patch('tilellm.modules.knowledge_graph_falkor.repository.falkor_repository.Graph', None), \
             patch.object(FalkorGraphRepository, 'ensure_indexes'):
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_client.execute_command.return_value = []
            
            repo = FalkorGraphRepository()
            # Ensure Graph API is not used
            repo._graph = None
            result = repo.execute_query("MATCH (n) RETURN n")
            mock_client.execute_command.assert_called_with("GRAPH.QUERY", "knowledge_graph", "MATCH (n) RETURN n")
            assert result == []


if __name__ == "__main__":
    pytest.main([__file__])