#!/usr/bin/env python3
"""
Unit tests for knowledge graph services.
Test GraphService and GraphRAGService with mocked dependencies.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import uuid

from tilellm.modules.knowledge_graph.services.services import (
    GraphService,
    GraphRAGService
)


class TestGraphService:
    """Test GraphService with mocked Neo4j repository."""
    
    def test_graph_service_initialization(self):
        """Test GraphService initialization."""
        mock_repository = Mock()
        service = GraphService(repository=mock_repository)
        assert service.repository == mock_repository
    
class TestGraphRAGService:
    """Test GraphRAGService with mocked dependencies."""
    
    def test_graph_rag_service_initialization(self):
        """Test GraphRAGService initialization."""
        mock_graph_service = Mock()
        mock_vector_store_repository = Mock()
        mock_llm = Mock()
        
        service = GraphRAGService(
            graph_service=mock_graph_service,
            vector_store_repository=mock_vector_store_repository,
            llm=mock_llm
        )
        
        assert service.graph_service == mock_graph_service
        assert service.vector_store_repository == mock_vector_store_repository
        assert service.llm == mock_llm
    