import pytest
from unittest.mock import MagicMock, AsyncMock

from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository
from tilellm.models import Engine
from tilellm.models.schemas import RepositoryNamespace
from qdrant_client.http import models
from qdrant_client.http.models import FacetValueHit, UpdateResult, UpdateStatus


@pytest.mark.asyncio
async def test_list_namespaces_success(mocker):
    """Test list_namespaces successfully returns namespaces."""
    # Arrange
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )

    mock_async_qdrant_client = AsyncMock()
    mock_facet_response = models.FacetResponse(
        hits=[
            models.FacetValueHit(value="namespace1", count=10),
            models.FacetValueHit(value="namespace2", count=20),
        ]
    )
    mock_async_qdrant_client.facet.return_value = mock_facet_response
    
    # Patch the constructor where it's called
    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()

    # Act
    result = await repo.list_namespaces(mock_engine)

    # Assert
    assert len(result.namespaces) == 2
    assert result.namespaces[0].namespace == "namespace1"
    assert result.namespaces[0].vector_count == 10
    assert result.namespaces[1].namespace == "namespace2"
    assert result.namespaces[1].vector_count == 20
    
    mock_async_qdrant_client.facet.assert_called_once_with(
        collection_name='test-collection',
        key='metadata.namespace'
    )

@pytest.mark.asyncio
async def test_list_namespaces_empty(mocker):
    """Test list_namespaces when there are no namespaces."""
    # Arrange
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )
    mock_async_qdrant_client = AsyncMock()
    mock_facet_response = models.FacetResponse(hits=[])
    mock_async_qdrant_client.facet.return_value = mock_facet_response
    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()

    # Act
    result = await repo.list_namespaces(mock_engine)

    # Assert
    assert len(result.namespaces) == 0

@pytest.mark.asyncio
async def test_list_namespaces_exception(mocker):
    """Test list_namespaces when the client raises an exception."""
    # Arrange
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )
    mock_async_qdrant_client = AsyncMock()
    mock_async_qdrant_client.facet.side_effect = Exception("Qdrant connection error")
    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()

    # Act & Assert
    with pytest.raises(Exception, match="Qdrant connection error"):
        await repo.list_namespaces(mock_engine)

@pytest.mark.asyncio
async def test_delete_namespace_success(mocker):
    """Test delete_namespace successfully calls the client's delete method."""
    # Arrange
    namespace_to_delete = "my-namespace"
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )
    
    mock_async_qdrant_client = AsyncMock()
    # Mock the response of the delete operation
    mock_async_qdrant_client.delete.return_value = models.UpdateResult(
        operation_id=1, status=models.UpdateStatus.COMPLETED
    )
    
    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()
    namespace_obj = RepositoryNamespace(engine=mock_engine, namespace=namespace_to_delete)

    # Act
    await repo.delete_namespace(namespace_obj)

    # Assert
    mock_async_qdrant_client.delete.assert_called_once()
    call_args, call_kwargs = mock_async_qdrant_client.delete.call_args
    
    # Check the collection_name and the points_selector filter
    assert call_kwargs['collection_name'] == "test-collection"
    
    expected_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.namespace",
                match=models.MatchValue(value=namespace_to_delete)
            )
        ]
    )
    
    assert call_kwargs['points_selector'] == models.FilterSelector(filter=expected_filter)

@pytest.mark.asyncio
async def test_delete_ids_namespace_success(mocker):
    """Test delete_ids_namespace successfully calls the client's delete method."""
    # Arrange
    metadata_id_to_delete = "doc-123"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )
    
    mock_async_qdrant_client = AsyncMock()
    mock_async_qdrant_client.delete.return_value = models.UpdateResult(
        operation_id=1, status=models.UpdateStatus.COMPLETED
    )
    
    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()

    # Act
    await repo.delete_ids_namespace(mock_engine, metadata_id_to_delete, namespace)

    # Assert
    mock_async_qdrant_client.delete.assert_called_once()
    call_args, call_kwargs = mock_async_qdrant_client.delete.call_args
    
    assert call_kwargs['collection_name'] == "test-collection"
    
    expected_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.id",
                match=models.MatchValue(value=metadata_id_to_delete)
            ),
            models.FieldCondition(
                key="metadata.namespace",
                match=models.MatchValue(value=namespace)
            )
        ]
    )
    
    assert call_kwargs['points_selector'] == models.FilterSelector(filter=expected_filter)

@pytest.mark.asyncio
async def test_delete_chunk_id_namespace_success(mocker):
    """Test delete_chunk_id_namespace successfully calls the client's delete method."""
    # Arrange
    chunk_id_to_delete = "chunk-456"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )
    
    mock_async_qdrant_client = AsyncMock()
    mock_async_qdrant_client.delete.return_value = models.UpdateResult(
        operation_id=1, status=models.UpdateStatus.COMPLETED
    )
    
    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()

    # Act
    await repo.delete_chunk_id_namespace(mock_engine, chunk_id_to_delete, namespace)

    # Assert
    mock_async_qdrant_client.delete.assert_called_once()
    call_args, call_kwargs = mock_async_qdrant_client.delete.call_args
    
    assert call_kwargs['collection_name'] == "test-collection"
    assert call_kwargs['points_selector'] == [chunk_id_to_delete]


@pytest.mark.asyncio
async def test_get_ids_namespace_success(mocker):
    """Test get_ids_namespace successfully returns items for a given metadata_id and namespace."""
    # Arrange
    metadata_id = "doc-123"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )

    mock_async_qdrant_client = AsyncMock()
    
    # Mock the collection_exists method
    mock_async_qdrant_client.collection_exists.return_value = True

    # Mock the scroll method to return a list of points
    mock_scroll_points = [
        MagicMock(
            id="chunk1",
            payload={
                "page_content": "content of chunk 1",
                "metadata": {
                    "id": metadata_id,
                    "source": "source1",
                    "type": "type1",
                    "namespace": namespace,
                    "date": "2025-01-01 10:00:00,000"
                }
            }
        ),
        MagicMock(
            id="chunk2",
            payload={
                "page_content": "content of chunk 2",
                "metadata": {
                    "id": metadata_id,
                    "source": "source1",
                    "type": "type1",
                    "namespace": namespace,
                    "date": "2025-01-01 10:01:00,000"
                }
            }
        )
    ]
    # Simulate pagination: first call returns points, second call returns empty list
    mock_async_qdrant_client.scroll.side_effect = [(mock_scroll_points, None)]

    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()

    # Act
    result = await repo.get_ids_namespace(mock_engine, metadata_id, namespace)

    # Assert
    assert len(result.matches) == 2
    assert result.matches[0].id == "chunk1"
    assert result.matches[0].metadata_id == metadata_id
    assert result.matches[0].metadata_source == "source1"
    assert result.matches[0].text == "content of chunk 1"
    
    assert result.matches[1].id == "chunk2"
    assert result.matches[1].metadata_id == metadata_id
    assert result.matches[1].metadata_source == "source1"
    assert result.matches[1].text == "content of chunk 2"

    mock_async_qdrant_client.collection_exists.assert_called_once_with(mock_engine.index_name)
    mock_async_qdrant_client.scroll.assert_called_once()
    
    call_args, call_kwargs = mock_async_qdrant_client.scroll.call_args
    assert call_kwargs['collection_name'] == mock_engine.index_name
    assert call_kwargs['scroll_filter'].must[0].key == "metadata.id"
    assert call_kwargs['scroll_filter'].must[0].match.value == metadata_id
    assert call_kwargs['scroll_filter'].must[1].key == "metadata.namespace"
    assert call_kwargs['scroll_filter'].must[1].match.value == namespace
    assert call_kwargs['offset'] is None
    assert call_kwargs['limit'] == 100
    assert call_kwargs['with_payload'] == ['page_content', 'metadata']
    assert call_kwargs['with_vectors'] == False

@pytest.mark.asyncio
async def test_get_all_obj_namespace_success(mocker):
    """Test get_all_obj_namespace successfully returns all items for a given namespace."""
    # Arrange
    namespace = "my-namespace"
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )

    mock_async_qdrant_client = AsyncMock()
    mock_async_qdrant_client.collection_exists.return_value = True

    mock_scroll_points = [
        MagicMock(id="chunk1", payload={"metadata": {"id": "id1", "source": "src1", "type": "type1", "namespace": namespace}}),
        MagicMock(id="chunk2", payload={"metadata": {"id": "id2", "source": "src2", "type": "type2", "namespace": namespace}}),
        MagicMock(id="chunk3", payload={"metadata": {"id": "id3", "source": "src3", "type": "type3", "namespace": namespace}}),
    ]
    mock_async_qdrant_client.scroll.side_effect = [(mock_scroll_points, None)]

    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()

    # Act
    result = await repo.get_all_obj_namespace(mock_engine, namespace)

    # Assert
    assert len(result.matches) == 3
    mock_async_qdrant_client.collection_exists.assert_called_once_with(mock_engine.index_name)
    mock_async_qdrant_client.scroll.assert_called_once()
    
    call_args, call_kwargs = mock_async_qdrant_client.scroll.call_args
    assert call_kwargs['collection_name'] == mock_engine.index_name
    assert call_kwargs['scroll_filter'].must[0].key == "metadata.namespace"
    assert call_kwargs['scroll_filter'].must[0].match.value == namespace
    assert call_kwargs['with_payload'] == ['metadata']

@pytest.mark.asyncio
async def test_get_desc_namespace_success(mocker):
    """Test get_desc_namespace successfully returns a description of the namespace."""
    # Arrange
    namespace = "my-namespace"
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )

    mock_async_qdrant_client = AsyncMock()
    mock_async_qdrant_client.collection_exists.return_value = True

    mock_scroll_points = [
        MagicMock(id="chunk1", payload={"metadata": {"id": "doc1", "source": "src1"}}),
        MagicMock(id="chunk2", payload={"metadata": {"id": "doc1", "source": "src1"}}),
        MagicMock(id="chunk3", payload={"metadata": {"id": "doc2", "source": "src2"}}),
    ]
    mock_async_qdrant_client.scroll.side_effect = [(mock_scroll_points, None)]

    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()

    # Act
    result = await repo.get_desc_namespace(mock_engine, namespace)

    # Assert
    assert result.namespace_desc.namespace == namespace
    assert result.namespace_desc.vector_count == 3
    
    assert len(result.ids) == 2
    assert result.ids[0].metadata_id == "doc1"
    assert result.ids[0].source == "src1"
    assert result.ids[0].chunks_count == 2
    
    assert result.ids[1].metadata_id == "doc2"
    assert result.ids[1].source == "src2"
    assert result.ids[1].chunks_count == 1

    mock_async_qdrant_client.collection_exists.assert_called_once_with(mock_engine.index_name)
    mock_async_qdrant_client.scroll.assert_called_once()
    
    call_args, call_kwargs = mock_async_qdrant_client.scroll.call_args
    assert call_kwargs['collection_name'] == mock_engine.index_name
    assert call_kwargs['scroll_filter'].must[0].key == "metadata.namespace"
    assert call_kwargs['scroll_filter'].must[0].match.value == namespace
    assert call_kwargs['with_payload'] == ['metadata.id', 'metadata.source']

@pytest.mark.asyncio
async def test_get_sources_namespace_success(mocker):
    """Test get_sources_namespace successfully returns items for a given source and namespace."""
    # Arrange
    source = "my-source"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="qdrant",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="test-collection",
        apikey=None
    )

    mock_async_qdrant_client = AsyncMock()
    mock_async_qdrant_client.collection_exists.return_value = True

    mock_scroll_points = [
        MagicMock(id="chunk1", payload={"page_content": "content1", "metadata": {"source": source, "namespace": namespace, "id": "doc1", "type": "type1"}}),
        MagicMock(id="chunk2", payload={"page_content": "content2", "metadata": {"source": source, "namespace": namespace, "id": "doc2", "type": "type2"}}),
    ]
    mock_async_qdrant_client.scroll.side_effect = [(mock_scroll_points, None)]

    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()

    # Act
    result = await repo.get_sources_namespace(mock_engine, source, namespace)

    # Assert
    assert len(result.matches) == 2
    assert result.matches[0].metadata_source == source
    assert result.matches[1].metadata_source == source
    
    mock_async_qdrant_client.collection_exists.assert_called_once_with(mock_engine.index_name)
    mock_async_qdrant_client.scroll.assert_called_once()
    
    call_args, call_kwargs = mock_async_qdrant_client.scroll.call_args
    assert call_kwargs['collection_name'] == mock_engine.index_name
    assert call_kwargs['scroll_filter'].must[0].key == "metadata.source"
    assert call_kwargs['scroll_filter'].must[0].match.value == source
    assert call_kwargs['scroll_filter'].must[1].key == "metadata.namespace"
    assert call_kwargs['scroll_filter'].must[1].match.value == namespace
    assert call_kwargs['with_payload'] == ['page_content', 'metadata']
