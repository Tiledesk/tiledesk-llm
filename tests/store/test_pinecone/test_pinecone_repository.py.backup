import pytest
from unittest.mock import AsyncMock, MagicMock

# Assuming the serverless implementation is the one to be tested.
# This might need to be adjusted if 'pod' is the target.
from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
from tilellm.models import Engine





# Custom mock classes to simulate Pinecone's response structure
class MockNamespaceSummary:
    def __init__(self, vector_count):
        self.vector_count = vector_count

class MockIndexStatsResponse:
    def __init__(self, namespaces_data):
        self.namespaces = {
            k: MockNamespaceSummary(v['vector_count']) for k, v in namespaces_data.items()
        }

@pytest.mark.asyncio
async def test_repository_initialization():
    """Test that the PineconeRepository can be initialized."""
    repo = PineconeRepositoryServerless()
    assert isinstance(repo, PineconeRepositoryServerless)

@pytest.mark.asyncio
async def test_delete_namespace_success(mocker):
    """Test delete_namespace successfully calls the client's delete method."""
    # Arrange
    namespace_to_delete = "my-namespace"
    mock_engine = Engine(
        name="pinecone",
        type="serverless",
        apikey="fake-api-key",
        index_name="test-index"
    )
    namespace_obj = MagicMock()
    namespace_obj.engine = mock_engine
    namespace_obj.namespace = namespace_to_delete

    # Mock the Pinecone client and its methods
    mock_pinecone_client = MagicMock()
    mock_index_async = AsyncMock()
    
    mocker.patch('pinecone.Pinecone', return_value=mock_pinecone_client)
    mock_pinecone_client.IndexAsyncio.return_value = mock_index_async
    mock_pinecone_client.describe_index.return_value.host = "dummy-host"

    repo = PineconeRepositoryServerless()

    # Act
    await repo.delete_namespace(namespace_obj)

    # Assert
    mock_pinecone_client.describe_index.assert_called_once_with(mock_engine.index_name)
    mock_pinecone_client.IndexAsyncio.assert_called_once_with(name=mock_engine.index_name, host="dummy-host")
    
    # We need to assert the call on the async context manager
    mock_index_async.__aenter__.return_value.delete.assert_called_once_with(
        delete_all=True, namespace=namespace_to_delete
    )

@pytest.mark.asyncio
async def test_delete_ids_namespace_success(mocker):
    """Test delete_ids_namespace successfully calls the client's delete method with correct filter."""
    # Arrange
    metadata_id = "doc-123"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="pinecone",
        type="serverless",
        apikey="fake-api-key",
        index_name="test-index"
    )

    # Mock the synchronous Pinecone client
    mock_sync_pinecone_client = MagicMock()
    mock_sync_index = MagicMock()

    # Configure the synchronous client mocks
    mock_sync_pinecone_client.describe_index.return_value.host = "dummy-host"
    mock_sync_pinecone_client.Index.return_value = mock_sync_index
    mock_sync_index.list.return_value = [[f"{metadata_id}#chunk1", f"{metadata_id}#chunk2"]] # Simulates a list of IDs to delete
    
    # Patch the synchronous Pinecone client class
    mocker.patch('pinecone.Pinecone', return_value=mock_sync_pinecone_client)

    repo = PineconeRepositoryServerless()

    # Act
    await repo.delete_ids_namespace(mock_engine, metadata_id, namespace)

    # Assert
    mock_sync_pinecone_client.describe_index.assert_called_once_with(mock_engine.index_name)
    mock_sync_pinecone_client.Index.assert_called_once_with(name=mock_engine.index_name, host="dummy-host")
    mock_sync_index.list.assert_called_once_with(prefix=f"{metadata_id}#", namespace=namespace)
    mock_sync_index.delete.assert_called_once_with(ids=[f"{metadata_id}#chunk1", f"{metadata_id}#chunk2"], namespace=namespace)

@pytest.mark.asyncio
async def test_delete_chunk_id_namespace_success(mocker):
    """Test delete_chunk_id_namespace successfully calls the client's delete method."""
    # Arrange
    chunk_id = "chunk-456"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="pinecone",
        type="serverless",
        apikey="fake-api-key",
        index_name="test-index"
    )

    mock_pinecone_client = MagicMock()
    mock_index_async = AsyncMock()
    
    mocker.patch('pinecone.Pinecone', return_value=mock_pinecone_client)
    mock_pinecone_client.IndexAsyncio.return_value = mock_index_async
    mock_pinecone_client.describe_index.return_value.host = "dummy-host"

    repo = PineconeRepositoryServerless()

    # Act
    await repo.delete_chunk_id_namespace(mock_engine, chunk_id, namespace)

    # Assert
    mock_pinecone_client.describe_index.assert_called_once_with(mock_engine.index_name)
    mock_pinecone_client.IndexAsyncio.assert_called_once_with(name=mock_engine.index_name, host="dummy-host")
    mock_index_async.__aenter__.return_value.delete.assert_called_once_with(ids=[chunk_id], namespace=namespace)


@pytest.mark.asyncio
async def test_get_ids_namespace_success(mocker):
    """Test get_ids_namespace successfully returns items for a given metadata_id and namespace."""
    # Arrange
    metadata_id = "doc-123"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="pinecone",
        type="serverless",
        apikey="fake-api-key",
        index_name="test-index",
        text_key="text"
    )

    mock_pinecone_client = MagicMock()
    mock_index_async = AsyncMock() # This is the object returned by pc.IndexAsyncio()

    # This configures what 'async with mock_async_index_context_manager as index_in_block:' will yield
    mock_index_in_block = MagicMock()
    mock_async_index_context_manager.__aenter__.return_value = mock_index_in_block
    mock_async_index_context_manager.__aexit__.return_value = False # To not suppress exceptions
    

    # Now, configure the describe_index_stats on the object yielded by the context manager
    mock_index_stats_response_instance = MockIndexStatsResponse(namespaces_data={
        "my-namespace": {"vector_count": 100}
    })
    mock_index_in_block.describe_index_stats.return_value = AsyncMock(return_value=mock_index_stats_response_instance)

    # Mock the query response on the object yielded by the context manager
    mock_query_response = MagicMock(
        matches=[
            MagicMock(
                id="chunk1",
                metadata={
                    "id": metadata_id,
                    "source": "source1",
                    "type": "type1",
                    "date": "2025-01-01",
                    "text": "content of chunk 1"
                }
            ),
            MagicMock(
                id="chunk2",
                metadata={
                    "id": metadata_id,
                    "source": "source1",
                    "type": "type1",
                    "date": "2025-01-02",
                    "text": "content of chunk 2"
                }
            )
        ]
    )
    mock_index_in_block.query.return_value = AsyncMock(return_value=mock_query_response)

    mocker.patch('pinecone.Pinecone', return_value=mock_pinecone_client)
    mock_pinecone_client.IndexAsyncio.return_value = mock_index_async
    mock_pinecone_client.describe_index.return_value.host = "dummy-host"

    repo = PineconeRepositoryServerless()

    # Act
    result = await repo.get_ids_namespace(mock_engine, metadata_id, namespace)

    # Assert
    assert len(result.matches) == 2
    assert result.matches[0].id == "chunk1"
    assert result.matches[0].metadata_id == metadata_id
    assert result.matches[0].metadata_source == "source1"
    assert result.matches[0].metadata_type == "type1"
    assert result.matches[0].date == "2025-01-01"
    assert result.matches[0].text == "content of chunk 1"

    assert result.matches[1].id == "chunk2"
    assert result.matches[1].metadata_id == metadata_id
    assert result.matches[1].metadata_source == "source1"
    assert result.matches[1].metadata_type == "type1"
    assert result.matches[1].date == "2025-01-02"
    assert result.matches[1].text == "content of chunk 2"

    mock_pinecone_client.describe_index.assert_called_once_with(mock_engine.index_name)
    mock_pinecone_client.IndexAsyncio.assert_called_once_with(name=mock_engine.index_name, host="dummy-host")
    mock_index_async.describe_index_stats.assert_called_once()
    mock_index_async.__aenter__.return_value.query.assert_called_once()
    
    call_kwargs = mock_index_async.__aenter__.return_value.query.call_args[1]
    assert call_kwargs['vector'] == [0] * 1536
    assert call_kwargs['top_k'] == 100 # min([total_vectors, 10000]) and total_vectors=100
    assert call_kwargs['filter'] == {"id": {"$eq": metadata_id}}
    assert call_kwargs['namespace'] == namespace
    assert call_kwargs['include_values'] == False
    assert call_kwargs['include_metadata'] == True

