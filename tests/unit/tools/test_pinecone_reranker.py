"""
Unit tests for Pinecone reranker integration.
"""
import sys
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pydantic import ValidationError

# Add project root to path
sys.path.insert(0, '/home/lor/sviluppo/tiledesk/tiledesk-llm')

from langchain_core.documents import Document
from tilellm.models.llm import PineconeRerankerConfig
from tilellm.tools.reranker import PineconeReranker, TileReranker


class TestPineconeRerankerConfig:
    """Test PineconeRerankerConfig Pydantic model."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = PineconeRerankerConfig(
            api_key="test-key",
            name="bge-reranker-v2-m3"
        )
        assert config.provider == "pinecone"
        assert config.api_key.get_secret_value() == "test-key"
        assert config.name == "bge-reranker-v2-m3"
        assert config.top_n is None
        assert config.rank_fields == ["chunk_text"]
        assert config.parameters == {"truncate": "END"}
    
    def test_config_with_custom_params(self):
        """Test configuration with custom parameters."""
        config = PineconeRerankerConfig(
            api_key="test-key",
            name="bge-reranker-v2-m3",
            top_n=10,
            rank_fields=["chunk_text", "title"],
            parameters={"truncate": "END", "return_scores": True}
        )
        assert config.top_n == 10
        assert config.rank_fields == ["chunk_text", "title"]
        assert config.parameters == {"truncate": "END", "return_scores": True}
    
    def test_missing_api_key(self):
        """Test validation error when api_key is missing."""
        with pytest.raises(ValidationError):
            PineconeRerankerConfig(
            name="bge-reranker-v2-m3"
            )
    
    def test_default_values(self):
        """Test default values are correctly set."""
        config = PineconeRerankerConfig(
            api_key="test-key"
        )
        assert config.name == "bge-reranker-v2-m3"
        assert config.rank_fields == ["chunk_text"]
        assert config.parameters == {"truncate": "END"}


class TestPineconeReranker:
    """Test PineconeReranker class."""
    
    @pytest.fixture
    def mock_pinecone(self):
        """Mock Pinecone client."""
        with patch('tilellm.tools.reranker.Pinecone') as mock_pinecone_class:
            mock_client = Mock()
            mock_inference = Mock()
            mock_client.inference = mock_inference
            mock_pinecone_class.return_value = mock_client
            yield mock_pinecone_class, mock_client, mock_inference
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return PineconeRerankerConfig(
            api_key="test-api-key",
            name="bge-reranker-v2-m3"
        )
    
    @pytest.fixture
    def documents(self):
        """Create test documents."""
        return [
            Document(
                page_content="First document content",
                metadata={"id": "doc1", "source": "test"}
            ),
            Document(
                page_content="Second document content that is longer",
                metadata={"id": "doc2", "source": "test"}
            ),
            Document(
                page_content="Third document",
                metadata={"id": "doc3", "source": "test"}
            )
        ]
    
    def test_init_without_pinecone(self, config):
        """Test initialization when pinecone is not installed."""
        with patch('tilellm.tools.reranker.Pinecone', None):
            with pytest.raises(ImportError, match="Pinecone is not available"):
                PineconeReranker(config)
    
    def test_init_success(self, config, mock_pinecone):
        """Test successful initialization."""
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        reranker = PineconeReranker(config)
        
        # Verify Pinecone was called with correct API key
        mock_pinecone_class.assert_called_once_with(api_key="test-api-key")
        assert reranker.client == mock_client
        assert reranker.name == "bge-reranker-v2-m3"
        assert reranker.api_key == "test-api-key"
        assert reranker.logger is not None
    
    def test_rerank_documents_no_documents(self, config, mock_pinecone):
        """Test rerank_documents with empty documents list."""
        reranker = PineconeReranker(config)
        result = reranker.rerank_documents("query", [], top_k=5)
        assert result == []
    
    def test_rerank_documents_success(self, config, mock_pinecone, documents):
        """Test successful reranking."""
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        # Mock the rerank response
        mock_response = Mock()
        mock_response.data = [
            Mock(index=1, score=0.9, document={"id": "doc2", "chunk_text": "Second document content that is longer"}),
            Mock(index=0, score=0.7, document={"id": "doc1", "chunk_text": "First document content"}),
            Mock(index=2, score=0.5, document={"id": "doc3", "chunk_text": "Third document"})
        ]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        result = reranker.rerank_documents("test query", documents, top_k=2)
        
        # Verify Pinecone API was called correctly
        mock_inference.rerank.assert_called_once_with(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=[
                {"id": "doc1", "chunk_text": "First document content"},
                {"id": "doc2", "chunk_text": "Second document content that is longer"},
                {"id": "doc3", "chunk_text": "Third document"}
            ],
            top_n=2,  # top_k = 2, top_n defaults to top_k
            rank_fields=["chunk_text"],
            return_documents=True,
            parameters={"truncate": "END"}
        )
        
        # Verify result - should return top 2 documents
        assert len(result) == 2
        assert result[0].page_content == "Second document content that is longer"
        assert result[0].metadata["id"] == "doc2"
        assert result[1].page_content == "First document content"
        assert result[1].metadata["id"] == "doc1"
    
    def test_rerank_documents_custom_top_n(self, config, mock_pinecone, documents):
        """Test reranking with custom top_n from config."""
        config.top_n = 5
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        mock_response = Mock()
        mock_response.data = [
            Mock(index=0, score=0.8, document={"id": "doc1", "chunk_text": "First document content"})
        ]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        reranker.rerank_documents("query", documents, top_k=2)
        
        # Should use config.top_n (5) not top_k (2) for Pinecone API
        mock_inference.rerank.assert_called_once()
        call_kwargs = mock_inference.rerank.call_args[1]
        assert call_kwargs["top_n"] == 3  # min(5, len(documents)=3) = 3
    
    def test_rerank_documents_api_error(self, config, mock_pinecone, documents):
        """Test handling of Pinecone API error."""
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        mock_inference.rerank.side_effect = Exception("API error")
        
        reranker = PineconeReranker(config)
        
        with pytest.raises(Exception, match="API error"):
            reranker.rerank_documents("query", documents, top_k=2)
    
    def test_model_specific_parameter_filtering_cohere(self, mock_pinecone):
        """Test that truncate parameter is filtered for cohere model."""
        config = PineconeRerankerConfig(
            api_key="test-key",
            name="cohere-rerank-3.5",
            parameters={"truncate": "END", "return_scores": True}
        )
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        # Mock the rerank response
        mock_response = Mock()
        mock_response.data = [Mock(index=0, score=0.8, document={"id": "doc1", "chunk_text": "content"})]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        
        # Verify truncate parameter was filtered out (cohere doesn't support it)
        # return_scores should remain
        assert "truncate" not in reranker.filtered_parameters
        assert "return_scores" in reranker.filtered_parameters
        assert reranker.filtered_parameters["return_scores"] is True
    
    def test_model_specific_parameter_filtering_bge(self, mock_pinecone):
        """Test that truncate parameter is kept for bge model."""
        config = PineconeRerankerConfig(
            api_key="test-key",
            name="bge-reranker-v2-m3",
            parameters={"truncate": "END", "return_scores": True}
        )
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        mock_response = Mock()
        mock_response.data = [Mock(index=0, score=0.8, document={"id": "doc1", "chunk_text": "content"})]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        
        # Verify truncate parameter is kept (bge supports it)
        assert "truncate" in reranker.filtered_parameters
        assert reranker.filtered_parameters["truncate"] == "END"
        assert "return_scores" in reranker.filtered_parameters
    
    def test_adaptive_maxp_strategy_cohere(self, mock_pinecone):
        """Test that Max-P strategy is disabled for cohere model."""
        config = PineconeRerankerConfig(
            api_key="test-key",
            name="cohere-rerank-3.5"
        )
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        mock_response = Mock()
        mock_response.data = [Mock(index=0, score=0.8, document={"id": "doc1", "chunk_text": "content"})]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        
        # Cohere model should have use_maxp = False
        assert reranker.use_maxp is False
        assert reranker.model_spec["requires_maxp"] is False
    
    def test_adaptive_maxp_strategy_bge(self, mock_pinecone):
        """Test that Max-P strategy is enabled for bge model."""
        config = PineconeRerankerConfig(
            api_key="test-key",
            name="bge-reranker-v2-m3"
        )
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        mock_response = Mock()
        mock_response.data = [Mock(index=0, score=0.8, document={"id": "doc1", "chunk_text": "content"})]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        
        # BGE model should have use_maxp = True
        assert reranker.use_maxp is True
        assert reranker.model_spec["requires_maxp"] is True
    
    def test_rank_field_validation_single_field_model(self, mock_pinecone):
        """Test rank field validation for models that support only single field."""
        config = PineconeRerankerConfig(
            api_key="test-key",
            name="bge-reranker-v2-m3",
            rank_fields=["chunk_text", "title"]  # Two fields
        )
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        mock_response = Mock()
        mock_response.data = [Mock(index=0, score=0.8, document={"id": "doc1", "chunk_text": "content"})]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        
        # BGE model supports only 1 rank field, should be truncated to first field
        assert len(reranker.rank_fields) == 1
        assert reranker.rank_fields == ["chunk_text"]
    
    def test_rank_field_validation_multiple_field_model(self, mock_pinecone):
        """Test rank field validation for models that support multiple fields."""
        config = PineconeRerankerConfig(
            api_key="test-key",
            name="cohere-rerank-3.5",
            rank_fields=["chunk_text", "title", "summary"]  # Three fields
        )
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        mock_response = Mock()
        mock_response.data = [Mock(index=0, score=0.8, document={"id": "doc1", "chunk_text": "content"})]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        
        # Cohere model supports unlimited rank fields, all should be kept
        assert len(reranker.rank_fields) == 3
        assert reranker.rank_fields == ["chunk_text", "title", "summary"]
    
    def test_batch_processing_exceeds_max_documents(self, mock_pinecone, documents):
        """Test batch processing when documents exceed model's max_documents limit."""
        config = PineconeRerankerConfig(
            api_key="test-key",
            name="bge-reranker-v2-m3"  # max_documents: 100
        )
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        # Create 150 documents (exceeds 100 limit)
        many_documents = []
        for i in range(150):
            many_documents.append(
                Document(
                    page_content=f"Document {i} content",
                    metadata={"id": f"doc{i}", "source": "test"}
                )
            )
        
        # Mock response for each batch
        mock_response = Mock()
        mock_response.data = [
            Mock(index=0, score=0.9, document={"id": "doc0", "chunk_text": "content"}),
            Mock(index=1, score=0.8, document={"id": "doc1", "chunk_text": "content"})
        ]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        reranker.rerank_documents("query", many_documents, top_k=10)
        
        # Should be called twice (2 batches of 100 and 50)
        assert mock_inference.rerank.call_count == 2
        
        # First batch should have top_n=10 (min(10, 100))
        first_call = mock_inference.rerank.call_args_list[0]
        assert len(first_call[1]["documents"]) == 100
        assert first_call[1]["top_n"] == 10
        
        # Second batch should have top_n=10 (min(10, 50))
        second_call = mock_inference.rerank.call_args_list[1]
        assert len(second_call[1]["documents"]) == 50
        assert second_call[1]["top_n"] == 10
    
    @pytest.mark.asyncio
    async def test_arerank_documents(self, config, mock_pinecone, documents):
        """Test async reranking."""
        mock_pinecone_class, mock_client, mock_inference = mock_pinecone
        
        mock_response = Mock()
        mock_response.data = [
            Mock(index=0, score=0.8, document={"id": "doc1", "chunk_text": "First document content"})
        ]
        mock_inference.rerank.return_value = mock_response
        
        reranker = PineconeReranker(config)
        result = await reranker.arerank_documents("query", documents, top_k=2)
        
        # Should call sync rerank_documents via thread pool
        mock_inference.rerank.assert_called_once()
        assert len(result) == 1
        assert result[0].metadata["id"] == "doc1"


class TestTileRerankerWithPinecone:
    """Test TileReranker integration with Pinecone provider."""
    
    @pytest.fixture
    def pinecone_config(self):
        """Create Pinecone reranker config."""
        return PineconeRerankerConfig(
            api_key="test-key",
            name="bge-reranker-v2-m3"
        )
    
    @pytest.fixture
    def documents(self):
        """Create test documents."""
        return [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
            Document(page_content="Doc 3", metadata={"id": "3"})
        ]
    
    def test_init_with_pinecone_config(self, pinecone_config):
        """Test TileReranker initialization with PineconeRerankerConfig."""
        with patch('tilellm.tools.reranker.PineconeReranker') as mock_reranker_class:
            mock_reranker = Mock()
            mock_reranker_class.return_value = mock_reranker
            
            reranker = TileReranker(pinecone_config)
            
            # Should create PineconeReranker instance
            mock_reranker_class.assert_called_once()
            assert reranker.model == mock_reranker
            assert "pinecone_bge-reranker-v2-m3" in reranker.model_name
    
    def test_rerank_documents_pinecone(self, pinecone_config, documents):
        """Test TileReranker.rerank_documents with Pinecone provider."""
        import tilellm.tools.reranker
        mock_reranker = Mock()
        mock_reranker.rerank_documents.return_value = documents[:2]
        
        # Patch _get_cached_model to return our mock
        with patch.object(TileReranker, '_get_cached_model', return_value=mock_reranker):
            # Patch isinstance to recognize our mock as PineconeReranker
            original_isinstance = isinstance
            def mock_isinstance(obj, classinfo):
                if obj is mock_reranker and classinfo is tilellm.tools.reranker.PineconeReranker:
                    return True
                return original_isinstance(obj, classinfo)
            
            with patch('tilellm.tools.reranker.isinstance', side_effect=mock_isinstance):
                reranker = TileReranker(pinecone_config)
                result = reranker.rerank_documents("query", documents, top_k=2)
                
                mock_reranker.rerank_documents.assert_called_once_with(
                    "query", documents, 2, 8
                )
                assert result == documents[:2]
    
    @pytest.mark.asyncio
    async def test_arerank_documents_pinecone(self, pinecone_config, documents):
        """Test TileReranker.arerank_documents with Pinecone provider."""
        import tilellm.tools.reranker
        mock_reranker = Mock()
        mock_reranker.arerank_documents = AsyncMock(return_value=documents[:2])
        
        # Patch _get_cached_model to return our mock
        with patch.object(TileReranker, '_get_cached_model', return_value=mock_reranker):
            # Patch isinstance to recognize our mock as PineconeReranker
            original_isinstance = isinstance
            def mock_isinstance(obj, classinfo):
                if obj is mock_reranker and classinfo is tilellm.tools.reranker.PineconeReranker:
                    return True
                return original_isinstance(obj, classinfo)
            
            with patch('tilellm.tools.reranker.isinstance', side_effect=mock_isinstance):
                reranker = TileReranker(pinecone_config)
                result = await reranker.arerank_documents("query", documents, top_k=2)
                
                mock_reranker.arerank_documents.assert_called_once_with(
                    query="query", documents=documents, top_k=2, batch_size=8
                )
                assert result == documents[:2]
    
    def test_cache_lru_pinecone(self, pinecone_config):
        """Test LRU cache with Pinecone reranker."""
        with patch('tilellm.tools.reranker.PineconeReranker') as mock_reranker_class:
            mock_reranker = Mock()
            mock_reranker_class.return_value = mock_reranker
            
            # Clear cache before test
            TileReranker.clear_cache()
            
            # First instance
            reranker1 = TileReranker(pinecone_config)
            assert mock_reranker_class.call_count == 1
            
            # Second instance with same config should use cache
            reranker2 = TileReranker(pinecone_config)
            assert mock_reranker_class.call_count == 1  # Still 1, from cache
            
            # Different Pinecone config should create new instance
            config2 = PineconeRerankerConfig(
                api_key="different-key",
                name="different-model"
            )
            reranker3 = TileReranker(config2)
            assert mock_reranker_class.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])