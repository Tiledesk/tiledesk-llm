import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase
from tilellm.store.milvus.milvus_repository import MilvusRepository
from tilellm.models import Engine
from langchain_core.documents import Document

# Concrete class for testing PineconeRepositoryBase
class TestPineconeRepo(PineconeRepositoryBase):
    async def add_item(self, item): pass
    async def add_item_hybrid(self, item): pass
    async def get_chunks_from_repo(self, question_answer): pass
    async def delete_ids_namespace(self, engine, metadata_id, namespace): pass
    async def search_community_report(self, question_answer, index, dense_vector, sparse_vector): pass
    async def initialize_embeddings_and_index(self, engine, embedding_model, emb_dimension): pass
    async def perform_hybrid_search(self, index, dense_vector, sparse_vector, top_k, namespace, filter): pass
    async def list_namespaces(self, engine): pass
    async def get_all_obj_namespace(self, engine, namespace, with_text=False): pass
    async def get_desc_namespace(self, engine, namespace): pass
    async def get_sources_namespace(self, engine, source, namespace): pass

@pytest.mark.asyncio
async def test_pinecone_get_by_doc_id_success(mocker):
    """Test get_by_doc_id for Pinecone with metadata filter."""
    mock_engine = Engine(
        index_name="test-index",
        apikey="test-key",
        text_key="text",
        type="serverless"
    )
    
    # Mock Pinecone
    mock_pc = MagicMock()
    mock_index_info = MagicMock()
    mock_index_info.host = "http://test-host"
    mock_index_info.dimension = 1536
    mock_pc.describe_index.return_value = mock_index_info
    
    # Use a MagicMock for the index, and manually mock __aenter__ and __aexit__
    mock_index = MagicMock()
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)
    
    # Mock query (async method)
    mock_index.query = AsyncMock(return_value={
        "matches": [
            {"id": "doc1#1", "metadata": {"doc_id": "doc1", "text": "chunk1"}},
            {"id": "doc1#2", "metadata": {"doc_id": "doc1", "text": "chunk2"}}
        ]
    })
    
    # Mock list (async iterator) - not expected to be called if query succeeds
    mock_index.list = MagicMock()
    
    mock_pc.IndexAsyncio.return_value = mock_index
    
    mocker.patch("pinecone.Pinecone", return_value=mock_pc)
    
    repo = TestPineconeRepo()
    
    # Act
    result = await repo.get_by_doc_id(mock_engine, "test-ns", "doc1")
    
    # Assert
    assert len(result) == 2
    assert result[0].page_content == "chunk1"
    assert result[1].page_content == "chunk2"
    mock_index.query.assert_called()

@pytest.mark.asyncio
async def test_milvus_get_by_doc_id_success(mocker):
    """Test get_by_doc_id for Milvus with metadata filter."""
    mock_engine = Engine(
        index_name="test-collection",
        apikey="test-key",
        host="http://localhost",
        port=19530
    )
    
    mock_client = MagicMock()
    mock_client.query.return_value = [
        {"id": "1", "metadata": {"doc_id": "doc1"}, "page_content": "chunk1"},
        {"id": "2", "metadata": {"doc_id": "doc1"}, "page_content": "chunk2"}
    ]
    
    repo = MilvusRepository()
    mocker.patch.object(repo, "_get_milvus_client", return_value=mock_client)
    
    # Act
    result = await repo.get_by_doc_id(mock_engine, "test-ns", "doc1")
    
    # Assert
    assert len(result) == 2
    assert result[0].page_content == "chunk1"
    assert result[1].page_content == "chunk2"
    mock_client.query.assert_called()
