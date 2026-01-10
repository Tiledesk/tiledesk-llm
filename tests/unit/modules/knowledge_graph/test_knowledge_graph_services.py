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
    
    def test_get_graph_status(self):
        """Test get_graph_status method."""
        mock_repository = Mock()
        mock_repository.get_graph_status = Mock(return_value={
            "node_count": 100,
            "relationship_count": 50,
            "labels": ["Document", "Entity"]
        })
        
        service = GraphService(repository=mock_repository)
        result = service.get_graph_status()
        
        assert result["node_count"] == 100
        assert result["relationship_count"] == 50
        assert "Document" in result["labels"]
        mock_repository.get_graph_status.assert_called_once()
    
    def test_get_nodes_by_namespace(self):
        """Test get_nodes_by_namespace method."""
        mock_repository = Mock()
        mock_nodes = [
            {"id": "1", "labels": ["Document"], "properties": {"text": "Test"}},
            {"id": "2", "labels": ["Entity"], "properties": {"name": "AI"}}
        ]
        mock_repository.get_nodes_by_namespace = Mock(return_value=mock_nodes)
        
        service = GraphService(repository=mock_repository)
        result = service.get_nodes_by_namespace("test_ns", "qdrant", limit=10)
        
        assert len(result) == 2
        assert result[0]["id"] == "1"
        mock_repository.get_nodes_by_namespace.assert_called_once_with(
            "test_ns", "qdrant", limit=10
        )
    
    def test_get_community_graph(self):
        """Test get_community_graph method."""
        mock_repository = Mock()
        mock_communities = {
            "communities": [
                {"id": 1, "nodes": ["node1", "node2"], "size": 2}
            ],
            "connections": [{"source": 1, "target": 2, "weight": 0.5}]
        }
        mock_repository.get_community_graph = Mock(return_value=mock_communities)
        
        service = GraphService(repository=mock_repository)
        result = service.get_community_graph("test_ns", "qdrant", resolution=1.0)
        
        assert "communities" in result
        assert len(result["communities"]) == 1
        mock_repository.get_community_graph.assert_called_once_with(
            "test_ns", "qdrant", resolution=1.0
        )
    
    def test_get_seed_matching(self):
        """Test get_seed_matching method."""
        mock_repository = Mock()
        mock_matches = [
            {"node_id": "1", "score": 0.9, "text": "Matching text"}
        ]
        mock_repository.get_seed_matching = Mock(return_value=mock_matches)
        
        service = GraphService(repository=mock_repository)
        result = service.get_seed_matching("test_ns", "qdrant", "query", top_k=5)
        
        assert len(result) == 1
        assert result[0]["score"] == 0.9
        mock_repository.get_seed_matching.assert_called_once_with(
            "test_ns", "qdrant", "query", top_k=5
        )


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
    
    @pytest.mark.asyncio
    async def test_import_from_vector_store_simple(self):
        """Test import_from_vector_store with simple document nodes (no LLM)."""
        mock_graph_service = Mock()
        mock_vector_store_repository = Mock()
        
        # Mock vector store documents
        mock_documents = [
            Mock(
                metadata={
                    "namespace": "test_ns",
                    "chunk_id": "chunk1",
                    "metadata_id": "doc1"
                },
                page_content="Document content 1"
            ),
            Mock(
                metadata={
                    "namespace": "test_ns",
                    "chunk_id": "chunk2",
                    "metadata_id": "doc2"
                },
                page_content="Document content 2"
            )
        ]
        
        mock_vector_store_repository.get_documents = AsyncMock(return_value=mock_documents)
        mock_graph_service.repository.create_document_node = Mock(return_value={"id": "node1"})
        
        service = GraphRAGService(
            graph_service=mock_graph_service,
            vector_store_repository=mock_vector_store_repository,
            llm=None  # No LLM -> simple import
        )
        
        mock_engine = Mock(name="qdrant")
        
        result = await service.import_from_vector_store(
            namespace="test_ns",
            engine=mock_engine,
            limit=2,
            overwrite=True,
            llm=None
        )
        
        assert result["imported_count"] == 2
        assert result["failed_count"] == 0
        mock_vector_store_repository.get_documents.assert_called_once()
        # Should create document nodes
        assert mock_graph_service.repository.create_document_node.call_count == 2
    
    @pytest.mark.asyncio
    async def test_import_from_vector_store_with_llm(self):
        """Test import_from_vector_store with LLM for entity extraction."""
        mock_graph_service = Mock()
        mock_vector_store_repository = Mock()
        mock_llm = AsyncMock()
        
        # Mock documents
        mock_documents = [
            Mock(
                metadata={"namespace": "test_ns", "chunk_id": "chunk1", "metadata_id": "doc1"},
                page_content="Apple is a technology company."
            )
        ]
        
        mock_vector_store_repository.get_documents = AsyncMock(return_value=mock_documents)
        
        # Mock LLM response for entity extraction
        mock_llm_response = Mock()
        mock_llm_response.content = '{"entities": ["Apple", "technology"], "relationships": [{"source": "Apple", "target": "technology", "type": "INDUSTRY"}]}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        
        # Mock repository methods
        mock_graph_service.repository.create_document_node = Mock(return_value={"id": "doc_node1"})
        mock_graph_service.repository.create_entity_node = Mock(return_value={"id": "entity1"})
        mock_graph_service.repository.create_relationship = Mock(return_value={"id": "rel1"})
        
        service = GraphRAGService(
            graph_service=mock_graph_service,
            vector_store_repository=mock_vector_store_repository,
            llm=mock_llm
        )
        
        mock_engine = Mock(name="qdrant")
        
        result = await service.import_from_vector_store(
            namespace="test_ns",
            engine=mock_engine,
            limit=1,
            overwrite=False,
            llm=mock_llm
        )
        
        assert result["imported_count"] == 1
        mock_llm.ainvoke.assert_called_once()
        mock_graph_service.repository.create_entity_node.assert_called()
        mock_graph_service.repository.create_relationship.assert_called()
    
    @pytest.mark.asyncio
    async def test_import_from_vector_store_error(self):
        """Test import_from_vector_store with error handling."""
        mock_graph_service = Mock()
        mock_vector_store_repository = Mock()
        
        # Mock repository to raise exception
        mock_vector_store_repository.get_documents = AsyncMock(
            side_effect=Exception("Vector store error")
        )
        
        service = GraphRAGService(
            graph_service=mock_graph_service,
            vector_store_repository=mock_vector_store_repository,
            llm=None
        )
        
        mock_engine = Mock(name="qdrant")
        
        result = await service.import_from_vector_store(
            namespace="test_ns",
            engine=mock_engine,
            limit=5,
            overwrite=True,
            llm=None
        )
        
        assert result["imported_count"] == 0
        assert result["failed_count"] == 0  # Or error count if implemented
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_retrieve_from_graph(self):
        """Test retrieve_from_graph method."""
        mock_graph_service = Mock()
        mock_vector_store_repository = Mock()
        
        # Mock graph retrieval
        mock_graph_nodes = [
            {
                "id": "node1",
                "labels": ["Document"],
                "properties": {"text": "Relevant document", "chunk_id": "chunk1"}
            }
        ]
        mock_graph_service.get_seed_matching = Mock(return_value=mock_graph_nodes)
        
        # Mock vector store retrieval for chunks
        mock_chunk_docs = [
            Mock(
                metadata={"chunk_id": "chunk1", "metadata_id": "doc1"},
                page_content="Relevant document chunk"
            )
        ]
        mock_vector_store_repository.get_documents_by_chunk_ids = AsyncMock(
            return_value=mock_chunk_docs
        )
        
        service = GraphRAGService(
            graph_service=mock_graph_service,
            vector_store_repository=mock_vector_store_repository,
            llm=None
        )
        
        mock_engine = Mock(name="qdrant")
        
        result = await service.retrieve_from_graph(
            namespace="test_ns",
            engine=mock_engine,
            query="test query",
            top_k=5
        )
        
        assert "documents" in result
        assert len(result["documents"]) == 1
        assert result["documents"][0].page_content == "Relevant document chunk"
        mock_graph_service.get_seed_matching.assert_called_once_with(
            "test_ns", mock_engine.name, "test query", top_k=5
        )
        mock_vector_store_repository.get_documents_by_chunk_ids.assert_called_once()