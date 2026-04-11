import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from tilellm.store.milvus.milvus_repository import MilvusRepository
from tilellm.models import Engine
from tilellm.models.schemas import RepositoryNamespace, RepositoryNamespaceResult, RepositoryItemNamespaceResult


@pytest.mark.asyncio
async def test_list_namespaces_success(mocker):
    """Test list_namespaces successfully returns namespaces."""
    # Arrange
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )

    # Mock the Milvus client and collection
    mock_client = MagicMock()
    mock_client.list_collections.return_value = ["test-collection"]
    
    # Mock Collection class and its query method with side_effect
    mock_collection = MagicMock()
    mock_collection.load.return_value = None
    
    # Define side_effect for collection.query
    def query_side_effect(**kwargs):
        expr = kwargs.get('expr')
        count = kwargs.get('count', False)
        if count:
            # Return count based on namespace in expr
            if "namespace1" in expr:
                return 10
            elif "namespace2" in expr:
                return 20
            else:
                return 0
        else:
            # First query for distinct namespaces
            return [
                {"metadata": {"namespace": "namespace1"}},
                {"metadata": {"namespace": "namespace2"}},
                {"metadata": {"namespace": "namespace1"}},
            ]
    
    mock_collection.query.side_effect = query_side_effect
    
    # Patch the pymilvus imports used inside the method
    mocker.patch('pymilvus.MilvusClient', return_value=mock_client)
    mocker.patch('pymilvus.Collection', return_value=mock_collection)
    
    # Also patch create_index to return our mock client (since list_namespaces calls create_index)
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', return_value=mock_client)
    
    # Act
    result = await repo.list_namespaces(mock_engine)
    
    # Assert
    assert len(result.namespaces) == 2
    # Check namespace1
    ns1 = next((ns for ns in result.namespaces if ns.namespace == "namespace1"), None)
    assert ns1 is not None
    assert ns1.vector_count == 10
    # Check namespace2
    ns2 = next((ns for ns in result.namespaces if ns.namespace == "namespace2"), None)
    assert ns2 is not None
    assert ns2.vector_count == 20
    
    # Ensure create_index was called with correct parameters
    repo.create_index.assert_called_once_with(
        engine=mock_engine,
        embeddings=None,
        emb_dimension=None
    )
    mock_collection.load.assert_called_once()
    # Verify query called three times: distinct namespaces + two count queries
    assert mock_collection.query.call_count == 3
    # First call should be for distinct namespaces
    first_call = mock_collection.query.call_args_list[0]
    assert first_call.kwargs.get('expr') == "metadata['namespace'] != ''"
    assert first_call.kwargs.get('output_fields') == ["metadata['namespace']"]
    assert first_call.kwargs.get('limit') == 10000
    # Second and third calls are count queries (order may vary)
    # We'll just ensure at least one count query for each namespace
    count_calls = [call for call in mock_collection.query.call_args_list if call.kwargs.get('count') == True]
    assert len(count_calls) == 2


@pytest.mark.asyncio
async def test_list_namespaces_empty(mocker):
    """Test list_namespaces when there are no namespaces."""
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )
    
    mock_client = MagicMock()
    mock_client.list_collections.return_value = ["test-collection"]
    mock_collection = MagicMock()
    mock_collection.load.return_value = None
    mock_collection.query.return_value = []  # empty list, no namespaces
    
    mocker.patch('pymilvus.MilvusClient', return_value=mock_client)
    mocker.patch('pymilvus.Collection', return_value=mock_collection)
    
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', return_value=mock_client)
    
    result = await repo.list_namespaces(mock_engine)
    
    # Should have empty list (or None) of namespaces
    assert result.namespaces is not None
    assert len(result.namespaces) == 0
    repo.create_index.assert_called_once_with(engine=mock_engine, embeddings=None, emb_dimension=None)
    mock_collection.load.assert_called_once()
    mock_collection.query.assert_called_once_with(
        expr="metadata['namespace'] != ''",
        output_fields=["metadata['namespace']"],
        limit=10000
    )


@pytest.mark.asyncio
async def test_list_namespaces_exception(mocker):
    """Test list_namespaces when the client raises an exception."""
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )
    
    mock_client = MagicMock()
    mock_client.list_collections.return_value = ["test-collection"]
    # Mock collection.load to raise exception
    mock_collection = MagicMock()
    mock_collection.load.side_effect = Exception("Milvus connection error")
    
    mocker.patch('pymilvus.MilvusClient', return_value=mock_client)
    mocker.patch('pymilvus.Collection', return_value=mock_collection)
    
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', return_value=mock_client)
    
    with pytest.raises(Exception, match="Milvus connection error"):
        await repo.list_namespaces(mock_engine)


# @pytest.mark.asyncio
async def test_delete_namespace_success(mocker):
    """Test delete_namespace successfully calls the client's delete method."""
    namespace_to_delete = "my-namespace"
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )
    
    mock_client = AsyncMock()
    mock_client.delete.return_value = {"delete_count": 5}
    
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', AsyncMock(return_value=mock_client))
    
    namespace_obj = RepositoryNamespace(engine=mock_engine, namespace=namespace_to_delete)
    
    # Act
    await repo.delete_namespace(namespace_obj)
    
    # Assert
    repo.create_index.assert_called_once_with(
        engine=mock_engine,
        embeddings=None,
        emb_dimension=None
    )
    mock_client.delete.assert_called_once_with(
        collection_name="test-collection",
        filter='metadata["namespace"] == "my-namespace"'
    )


@pytest.mark.asyncio
async def test_delete_ids_namespace_success(mocker):
    """Test delete_ids_namespace successfully calls the client's delete method."""
    metadata_id_to_delete = "doc-123"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )
    
    mock_client = MagicMock()
    mock_client.delete.return_value = {"delete_count": 2}
    
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', return_value=mock_client)
    
    # Act
    await repo.delete_ids_namespace(mock_engine, metadata_id_to_delete, namespace)
    
    # Assert
    repo.create_index.assert_called_once_with(
        engine=mock_engine,
        embeddings=None,
        emb_dimension=None
    )
    mock_client.delete.assert_called_once_with(
        collection_name="test-collection",
        filter='metadata["id"] == "doc-123" and metadata["namespace"] == "my-namespace"'
    )


@pytest.mark.asyncio
async def test_delete_chunk_id_namespace_success(mocker):
    """Test delete_chunk_id_namespace successfully calls the client's delete method."""
    chunk_id = "chunk-456"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )
    
    mock_client = MagicMock()
    mock_client.delete.return_value = {"delete_count": 1}
    
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', return_value=mock_client)
    
    # Act
    await repo.delete_chunk_id_namespace(mock_engine, chunk_id, namespace)
    
    # Assert
    repo.create_index.assert_called_once_with(
        engine=mock_engine,
        embeddings=None,
        emb_dimension=None
    )
    mock_client.delete.assert_called_once_with(
        collection_name="test-collection",
        filter='id == "chunk-456" and metadata["namespace"] == "my-namespace"'
    )


@pytest.mark.asyncio
async def test_get_ids_namespace_success(mocker):
    """Test get_ids_namespace successfully returns items."""
    metadata_id = "doc-123"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )
    
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.load.return_value = None
    
    # Mock query results - simulate pagination
    mock_results = [
        {
            "id": "chunk-1",
            "metadata": {"id": metadata_id, "source": "source1", "type": "pdf", "date": "2024-01-01", "namespace": namespace},
            "page_content": "Content 1"
        },
        {
            "id": "chunk-2", 
            "metadata": {"id": metadata_id, "source": "source1", "type": "pdf", "date": "2024-01-01", "namespace": namespace},
            "page_content": "Content 2"
        }
    ]
    
    # First query returns results, second returns empty (break loop)
    mock_collection.query.side_effect = [mock_results, []]
    
    mocker.patch('pymilvus.Collection', return_value=mock_collection)
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', return_value=mock_client)
    
    # Act
    result = await repo.get_ids_namespace(mock_engine, metadata_id, namespace)
    
    # Assert
    repo.create_index.assert_called_once_with(
        engine=mock_engine,
        embeddings=None,
        emb_dimension=None
    )
    mock_collection.load.assert_called_once()
    
    # Check query calls - only one call because results count (2) < limit (1000)
    assert mock_collection.query.call_count == 1
    first_call = mock_collection.query.call_args_list[0]
    assert first_call.kwargs.get('expr') == f'metadata["id"] == "{metadata_id}" and metadata["namespace"] == "{namespace}"'
    assert first_call.kwargs.get('output_fields') == ["metadata", "page_content"]
    assert first_call.kwargs.get('limit') == 1000
    assert first_call.kwargs.get('offset') == 0
    
    # Check result
    assert len(result.matches) == 2
    match1 = result.matches[0]
    assert match1.id == "chunk-1"
    assert match1.metadata_id == metadata_id
    assert match1.metadata_source == "source1"
    assert match1.metadata_type == "pdf"
    assert match1.date == "2024-01-01"
    assert match1.text == "Content 1"


@pytest.mark.asyncio
async def test_get_all_obj_namespace_success(mocker):
    """Test get_all_obj_namespace successfully returns items."""
    namespace = "my-namespace"
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )
    
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.load.return_value = None
    
    # Mock query results for with_text=False (no page_content)
    mock_results = [
        {
            "id": "chunk-1",
            "metadata": {"id": "doc-1", "source": "source1", "type": "pdf", "date": "2024-01-01", "namespace": namespace}
        },
        {
            "id": "chunk-2",
            "metadata": {"id": "doc-2", "source": "source2", "type": "url", "date": "2024-01-02", "namespace": namespace}
        }
    ]
    
    mock_collection.query.side_effect = [mock_results, []]
    
    mocker.patch('pymilvus.Collection', return_value=mock_collection)
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', return_value=mock_client)
    
    # Act - with_text=False
    result = await repo.get_all_obj_namespace(mock_engine, namespace, with_text=False)
    
    # Assert
    repo.create_index.assert_called_once_with(
        engine=mock_engine,
        embeddings=None,
        emb_dimension=None
    )
    mock_collection.load.assert_called_once()
    
    # Check query calls
    assert mock_collection.query.call_count == 1
    first_call = mock_collection.query.call_args_list[0]
    assert first_call.kwargs.get('expr') == f'metadata["namespace"] == "{namespace}"'
    assert first_call.kwargs.get('output_fields') == ["metadata"]
    assert first_call.kwargs.get('limit') == 1000
    assert first_call.kwargs.get('offset') == 0
    
    # Check result
    assert len(result.matches) == 2
    match1 = result.matches[0]
    assert match1.id == "chunk-1"
    assert match1.metadata_id == "doc-1"
    assert match1.metadata_source == "source1"
    assert match1.metadata_type == "pdf"
    assert match1.date == "2024-01-01"
    assert match1.text is None  # with_text=False
    
    # Test with_text=True
    # Reset mocks
    mock_collection.reset_mock()
    repo.create_index.reset_mock()
    
    mock_results_with_text = [
        {
            "id": "chunk-3",
            "metadata": {"id": "doc-3", "source": "source3", "type": "txt", "date": "2024-01-03", "namespace": namespace},
            "page_content": "Full text content"
        }
    ]
    mock_collection.query.side_effect = [mock_results_with_text, []]
    
    result_with_text = await repo.get_all_obj_namespace(mock_engine, namespace, with_text=True)
    
    # Check query includes page_content
    first_call = mock_collection.query.call_args_list[0]
    assert "page_content" in first_call.kwargs.get('output_fields')
    assert len(result_with_text.matches) == 1
    assert result_with_text.matches[0].text == "Full text content"


@pytest.mark.asyncio
async def test_get_desc_namespace_success(mocker):
    """Test get_desc_namespace successfully returns namespace description."""
    namespace = "my-namespace"
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )
    
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.load.return_value = None
    
    # Mock count query returns integer
    mock_collection.query.side_effect = [
        5,  # count result (total vectors)
        [
            {"metadata": {"id": "doc-1", "source": "source1", "namespace": namespace}},
            {"metadata": {"id": "doc-1", "source": "source1", "namespace": namespace}},
            {"metadata": {"id": "doc-2", "source": "source2", "namespace": namespace}},
            {"metadata": {"id": "doc-3", "source": "source3", "namespace": namespace}},
            {"metadata": {"id": "doc-3", "source": "source3", "namespace": namespace}}
        ],
        []  # empty second page
    ]
    
    mocker.patch('pymilvus.Collection', return_value=mock_collection)
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', return_value=mock_client)
    
    # Act
    result = await repo.get_desc_namespace(mock_engine, namespace)
    
    # Assert
    repo.create_index.assert_called_once_with(
        engine=mock_engine,
        embeddings=None,
        emb_dimension=None
    )
    mock_collection.load.assert_called_once()
    
    # Check query calls
    assert mock_collection.query.call_count == 2
    # First call should be count query
    count_call = mock_collection.query.call_args_list[0]
    assert count_call.kwargs.get('expr') == f'metadata["namespace"] == "{namespace}"'
    assert count_call.kwargs.get('count') == True
    
    # Second call should be query for metadata
    metadata_call = mock_collection.query.call_args_list[1]
    assert metadata_call.kwargs.get('expr') == f'metadata["namespace"] == "{namespace}"'
    assert metadata_call.kwargs.get('output_fields') == ["metadata"]
    assert metadata_call.kwargs.get('limit') == 1000
    assert metadata_call.kwargs.get('offset') == 0
    
    # Check result
    assert result.namespace_desc.namespace == namespace
    assert result.namespace_desc.vector_count == 5
    assert len(result.ids) == 3  # 3 unique document IDs
    
    # Check grouping
    id_summaries = {summary.metadata_id: summary for summary in result.ids}
    assert "doc-1" in id_summaries
    assert id_summaries["doc-1"].chunks_count == 2
    assert id_summaries["doc-1"].source == "source1"
    
    assert "doc-2" in id_summaries
    assert id_summaries["doc-2"].chunks_count == 1
    assert id_summaries["doc-2"].source == "source2"
    
    assert "doc-3" in id_summaries
    assert id_summaries["doc-3"].chunks_count == 2
    assert id_summaries["doc-3"].source == "source3"


@pytest.mark.asyncio
async def test_get_sources_namespace_success(mocker):
    """Test get_sources_namespace successfully returns items by source."""
    source = "my-source"
    namespace = "my-namespace"
    mock_engine = Engine(
        name="milvus",
        deployment="local",
        host="localhost",
        port=19530,
        index_name="test-collection",
        apikey=None,
        database="default"
    )
    
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.load.return_value = None
    
    # Mock query results
    mock_results = [
        {
            "id": "chunk-1",
            "metadata": {"id": "doc-1", "source": source, "type": "pdf", "date": "2024-01-01", "namespace": namespace},
            "page_content": "Content from source"
        }
    ]
    
    mock_collection.query.side_effect = [mock_results, []]
    
    mocker.patch('pymilvus.Collection', return_value=mock_collection)
    repo = MilvusRepository()
    mocker.patch.object(repo, 'create_index', return_value=mock_client)
    
    # Act
    result = await repo.get_sources_namespace(mock_engine, source, namespace)
    
    # Assert
    repo.create_index.assert_called_once_with(
        engine=mock_engine,
        embeddings=None,
        emb_dimension=None
    )
    mock_collection.load.assert_called_once()
    
    # Check query calls
    assert mock_collection.query.call_count == 1
    first_call = mock_collection.query.call_args_list[0]
    assert first_call.kwargs.get('expr') == f'metadata["source"] == "{source}" and metadata["namespace"] == "{namespace}"'
    assert first_call.kwargs.get('output_fields') == ["metadata", "page_content"]
    assert first_call.kwargs.get('limit') == 1000
    assert first_call.kwargs.get('offset') == 0
    
    # Check result
    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.id == "chunk-1"
    assert match.metadata_id == "doc-1"
    assert match.metadata_source == source
    assert match.metadata_type == "pdf"
    assert match.date == "2024-01-01"
    assert match.text == "Content from source"