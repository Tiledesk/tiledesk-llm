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
async def test_get_all_obj_namespace_passes_through_full_metadata(mocker):
    """RepositoryQueryResult.metadata must carry the full raw metadata dict
    (already fetched from Qdrant's payload) — needed so callers like lgraph's
    build_lgraph can see custom fields (e.g. page_number, doc_type) that the
    fixed id/source/type/date fields don't cover."""
    namespace = "my-namespace"
    mock_engine = Engine(
        name="qdrant", deployment="local", host="localhost", port=6333,
        index_name="test-collection", apikey=None,
    )

    mock_async_qdrant_client = AsyncMock()
    mock_async_qdrant_client.collection_exists.return_value = True

    raw_metadata = {
        "id": "id1", "source": "src1", "type": "type1",
        "namespace": namespace, "page_number": 7, "doc_type": "delibera",
    }
    mock_scroll_points = [MagicMock(id="chunk1", payload={"metadata": raw_metadata})]
    mock_async_qdrant_client.scroll.side_effect = [(mock_scroll_points, None)]

    mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient', return_value=mock_async_qdrant_client)

    repo = QdrantRepository()
    result = await repo.get_all_obj_namespace(mock_engine, namespace)

    assert result.matches[0].metadata == raw_metadata


@pytest.mark.asyncio
async def test_get_chunks_from_repo_exposes_chunk_ids(mocker):
    """RetrievalChunksResult.chunk_ids must carry each match's point id (same id
    space as get_all_obj_namespace) — needed as PPR seed_chunk_ids for lgraph
    hybrid retrieval. Point ids were already read onto Document.id and discarded."""
    from pydantic import SecretStr
    from tilellm.models import QuestionAnswer
    from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository

    mock_engine = Engine(
        name="qdrant", deployment="local", host="localhost", port=6333,
        index_name="test-collection", apikey=None,
    )
    question_answer = QuestionAnswer(
        question="q", namespace="ns", engine=mock_engine, search_type="similarity", top_k=2,
        gptkey=SecretStr("test-key"),
    )

    mock_point = MagicMock()
    mock_point.id = "point-123"
    mock_point.payload = {"page_content": "hello", "metadata": {"source": "s1"}}

    mock_client = MagicMock()
    mock_client.query_points = MagicMock(return_value=MagicMock(points=[mock_point]))

    mock_vector_store = MagicMock()
    mock_vector_store.client = mock_client

    mock_embedding_obj = AsyncMock()
    mock_embedding_obj.aembed_query = AsyncMock(return_value=[0.1, 0.2])

    # Patch create() on the class itself, not the constructor: the decorator caches
    # its CachedAsyncEmbeddingFactory() instance in a closure the first time any test
    # calls it, so a later mocker.patch(..., return_value=...) on the constructor is a
    # no-op once another test already triggered that cache — order-dependent flakiness
    # seen empirically. Patching the method works regardless of which instance is cached.
    from tilellm.shared.embeddings.embedding_client_manager import CachedAsyncEmbeddingFactory
    mocker.patch.object(
        CachedAsyncEmbeddingFactory, "create",
        AsyncMock(return_value=(mock_embedding_obj, 1536)),
    )

    repo = QdrantRepository()
    mocker.patch.object(repo, "create_index", AsyncMock(return_value=mock_vector_store))

    result = await repo.get_chunks_from_repo(question_answer)

    assert result.chunk_ids == ["point-123"]


@pytest.mark.asyncio
async def test_get_chunks_from_repo_uses_retrieval_query_when_set(mocker):
    """HyDE (and future Self-RAG): when question_answer.retrieval_query is set, the
    embedding call must use it instead of .question — otherwise the hypothetical
    document generated upstream (tilellm/agents/nodes.py::hyde_node) never actually
    reaches the vector store, silently no-opping HyDE through this repository."""
    from pydantic import SecretStr
    from tilellm.models import QuestionAnswer
    from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository

    mock_engine = Engine(
        name="qdrant", deployment="local", host="localhost", port=6333,
        index_name="test-collection", apikey=None,
    )
    question_answer = QuestionAnswer(
        question="domanda originale", retrieval_query="passaggio ipotetico HyDE",
        namespace="ns", engine=mock_engine, search_type="similarity", top_k=2,
        gptkey=SecretStr("test-key"),
    )

    mock_point = MagicMock()
    mock_point.id = "point-1"
    mock_point.payload = {"page_content": "hello", "metadata": {"source": "s1"}}

    mock_client = MagicMock()
    mock_client.query_points = MagicMock(return_value=MagicMock(points=[mock_point]))
    mock_vector_store = MagicMock()
    mock_vector_store.client = mock_client

    mock_embedding_obj = AsyncMock()
    mock_embedding_obj.aembed_query = AsyncMock(return_value=[0.1, 0.2])

    # Patch create() on the class itself, not the constructor: the decorator caches
    # its CachedAsyncEmbeddingFactory() instance in a closure the first time any test
    # calls it, so a later mocker.patch(..., return_value=...) on the constructor is a
    # no-op once another test already triggered that cache — order-dependent flakiness
    # seen empirically. Patching the method works regardless of which instance is cached.
    from tilellm.shared.embeddings.embedding_client_manager import CachedAsyncEmbeddingFactory
    mocker.patch.object(
        CachedAsyncEmbeddingFactory, "create",
        AsyncMock(return_value=(mock_embedding_obj, 1536)),
    )

    repo = QdrantRepository()
    mocker.patch.object(repo, "create_index", AsyncMock(return_value=mock_vector_store))

    await repo.get_chunks_from_repo(question_answer)

    mock_embedding_obj.aembed_query.assert_awaited_once_with("passaggio ipotetico HyDE")


@pytest.mark.asyncio
async def test_get_chunks_from_repo_hybrid_uses_retrieval_query_when_set(mocker):
    """Same guarantee as above, on the hybrid (dense+sparse) search path — the one
    actually used by compliance_checker/discretionary_check_service."""
    from pydantic import SecretStr
    from tilellm.models import QuestionAnswer
    from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository

    mock_engine = Engine(
        name="qdrant", deployment="local", host="localhost", port=6333,
        index_name="test-collection", apikey=None,
    )
    question_answer = QuestionAnswer(
        question="domanda originale", retrieval_query="passaggio ipotetico HyDE",
        namespace="ns", engine=mock_engine, search_type="hybrid", top_k=2,
        gptkey=SecretStr("test-key"), sparse_encoder="splade",
    )

    mock_point = MagicMock()
    mock_point.id = "point-1"
    mock_point.payload = {"page_content": "hello", "metadata": {"source": "s1"}}

    mock_client = MagicMock()
    mock_client.query_points = MagicMock(return_value=MagicMock(points=[mock_point]))
    mock_vector_store = MagicMock()
    mock_vector_store.client = mock_client

    mock_embedding_obj = AsyncMock()
    mock_embedding_obj.aembed_query = AsyncMock(return_value=[0.1, 0.2])

    # Patch create() on the class itself, not the constructor: the decorator caches
    # its CachedAsyncEmbeddingFactory() instance in a closure the first time any test
    # calls it, so a later mocker.patch(..., return_value=...) on the constructor is a
    # no-op once another test already triggered that cache — order-dependent flakiness
    # seen empirically. Patching the method works regardless of which instance is cached.
    from tilellm.shared.embeddings.embedding_client_manager import CachedAsyncEmbeddingFactory
    mocker.patch.object(
        CachedAsyncEmbeddingFactory, "create",
        AsyncMock(return_value=(mock_embedding_obj, 1536)),
    )

    mock_sparse_encoder = AsyncMock()
    mock_sparse_encoder.aencode_queries = AsyncMock(return_value={"indices": [1], "values": [0.5]})
    mocker.patch(
        "tilellm.store.qdrant.qdrant_repository_local.TiledeskSparseEncoders",
        return_value=mock_sparse_encoder,
    )

    repo = QdrantRepository()
    mocker.patch.object(repo, "create_index", AsyncMock(return_value=mock_vector_store))
    mocker.patch.object(repo, "get_embeddings_dimension", AsyncMock(return_value=1536))

    await repo.get_chunks_from_repo(question_answer)

    mock_embedding_obj.aembed_query.assert_awaited_once_with("passaggio ipotetico HyDE")
    mock_sparse_encoder.aencode_queries.assert_awaited_once_with("passaggio ipotetico HyDE")


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


@pytest.mark.asyncio
async def test_upsert_vector_store_enforces_namespace_on_every_chunk():
    """
    regex_custom builds its MetadataItem without `namespace` (defaults to None,
    clobbering document.metadata via the merge in add_item). upsert_vector_store
    must stamp the real namespace on every chunk before upsert regardless of what
    upstream produced, mirroring upsert_vector_store_hybrid's existing behavior.
    """
    from langchain_core.documents import Document

    namespace = "tenant-a"
    chunks = [
        Document(page_content="c1", metadata={"id": "doc1", "namespace": None}),  # regex_custom bug
        Document(page_content="c2", metadata={"id": "doc1"}),  # namespace missing entirely
        Document(page_content="c3", metadata={"id": "doc1", "namespace": "stale-namespace"}),
    ]

    mock_vector_store = MagicMock()
    mock_vector_store.aadd_documents = AsyncMock(return_value=["id1", "id2", "id3"])

    await QdrantRepository.upsert_vector_store(
        vector_store=mock_vector_store,
        chunks=chunks,
        metadata_id="doc1",
        namespace=namespace,
    )

    assert all(chunk.metadata["namespace"] == namespace for chunk in chunks)
    mock_vector_store.aadd_documents.assert_awaited_once()
    awaited_chunks = mock_vector_store.aadd_documents.call_args.kwargs["documents"]
    assert all(chunk.metadata["namespace"] == namespace for chunk in awaited_chunks)


class TestNormalizeUpsertMetadata:
    """Shared normalization applied by both upsert_vector_store and upsert_vector_store_hybrid."""

    def test_sets_namespace_and_defaults_missing_tags_to_empty_list(self):
        metadata = {"id": "doc1"}
        result = QdrantRepository._normalize_upsert_metadata(metadata, "tenant-a")

        assert result["namespace"] == "tenant-a"
        assert result["tags"] == []

    def test_overrides_stale_namespace_and_none_tags(self):
        metadata = {"id": "doc1", "namespace": "wrong-namespace", "tags": None}
        result = QdrantRepository._normalize_upsert_metadata(metadata, "tenant-a")

        assert result["namespace"] == "tenant-a"
        assert result["tags"] == []

    def test_preserves_existing_tags(self):
        metadata = {"id": "doc1", "tags": ["billing", "urgent"]}
        result = QdrantRepository._normalize_upsert_metadata(metadata, "tenant-a")

        assert result["tags"] == ["billing", "urgent"]


@pytest.mark.asyncio
async def test_upsert_vector_store_defaults_tags_to_empty_list():
    """Used for metadata filtering (build_tags_filter/build_filter): must always be a list, never absent."""
    from langchain_core.documents import Document

    chunks = [Document(page_content="c1", metadata={"id": "doc1"})]
    mock_vector_store = MagicMock()
    mock_vector_store.aadd_documents = AsyncMock(return_value=["id1"])

    await QdrantRepository.upsert_vector_store(
        vector_store=mock_vector_store, chunks=chunks, metadata_id="doc1", namespace="tenant-a"
    )

    assert chunks[0].metadata["tags"] == []


@pytest.mark.asyncio
async def test_upsert_vector_store_hybrid_defaults_tags_to_empty_list():
    from langchain_core.documents import Document

    chunks = [Document(page_content="c1", metadata={"id": "doc1"})]
    mock_vector_store = MagicMock()
    mock_vector_store.client.upsert = MagicMock()
    mock_engine = Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="test-collection")
    mock_embeddings = MagicMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])

    await QdrantRepository.upsert_vector_store_hybrid(
        vector_store=mock_vector_store,
        contents=["c1"],
        chunks=chunks,
        metadata_id="doc1",
        engine=mock_engine,
        namespace="tenant-a",
        embeddings=mock_embeddings,
        sparse_vectors=[{"indices": [0], "values": [1.0]}],
    )

    payload = mock_vector_store.client.upsert.call_args.kwargs["points"][0].payload
    assert payload["metadata"]["namespace"] == "tenant-a"
    assert payload["metadata"]["tags"] == []
    # Original chunk metadata must be untouched (hybrid builds a copy, unlike upsert_vector_store).
    assert "tags" not in chunks[0].metadata


class TestBuildRegexCustomChunks:
    """Shared by add_item and add_item_hybrid so both stay consistent."""

    @staticmethod
    def _make_item(**overrides):
        from types import SimpleNamespace
        defaults = dict(
            id="doc1", source="https://example.com/doc", type="regex_custom",
            embedding="text-embedding-3-small", namespace="tenant-a", tags=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_sets_namespace_file_name_page_and_stringified_embedding(self):
        from langchain_core.documents import Document
        from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository as Repo

        item = self._make_item()
        documents = [Document(page_content="chunk one", metadata={})]

        chunks = Repo._build_regex_custom_chunks(item, documents)

        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert meta["namespace"] == "tenant-a"
        assert meta["file_name"]  # non-empty, derived from source
        assert meta["page"] == 1
        assert meta["embedding"] == "text-embedding-3-small"
        assert isinstance(meta["embedding"], str)

    def test_preserves_existing_file_name_and_page(self):
        from langchain_core.documents import Document
        from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository as Repo

        item = self._make_item()
        documents = [Document(page_content="chunk one", metadata={"file_name": "custom.txt", "page": 3})]

        chunks = Repo._build_regex_custom_chunks(item, documents)

        assert chunks[0].metadata["file_name"] == "custom.txt"
        assert chunks[0].metadata["page"] == 3

    def test_includes_tags_when_present(self):
        from langchain_core.documents import Document
        from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository as Repo

        item = self._make_item(tags=["billing"])
        documents = [Document(page_content="chunk one", metadata={})]

        chunks = Repo._build_regex_custom_chunks(item, documents)

        assert chunks[0].metadata["tags"] == ["billing"]


# ---------------------------------------------------------------------------
# Hybrid prefetch limit — every RRF branch must fetch at least top_k candidates.
# Without an explicit limit Qdrant defaults each Prefetch to 10, so the fused pool
# silently caps at 20 whatever top_k asks for (seen on real data: top_k=45 -> 20),
# which made the reranking oversample (top_k x reranking_multiplier) a no-op.
# ---------------------------------------------------------------------------

def _prefetch_limits(mock_client):
    prefetch = mock_client.query_points.call_args.kwargs["prefetch"]
    return [p.limit for p in prefetch]


def _hybrid_qa(top_k):
    from pydantic import SecretStr
    from tilellm.models import QuestionAnswer

    engine = Engine(name="qdrant", deployment="local", host="localhost", port=6333,
                    index_name="test-collection", apikey=None)
    return QuestionAnswer(question="q", namespace="ns", engine=engine, search_type="hybrid",
                          top_k=top_k, gptkey=SecretStr("test-key"), sparse_encoder="splade")


def _client_returning_one_point():
    point = MagicMock()
    point.id = "point-1"
    point.payload = {"page_content": "hello", "metadata": {"source": "s1"}}
    client = MagicMock()
    client.query_points = MagicMock(return_value=MagicMock(points=[point]))
    return client


@pytest.mark.asyncio
async def test_get_chunks_from_repo_hybrid_prefetch_limit_matches_top_k(mocker):
    from tilellm.shared.embeddings.embedding_client_manager import CachedAsyncEmbeddingFactory

    client = _client_returning_one_point()
    vector_store = MagicMock()
    vector_store.client = client
    embedding_obj = AsyncMock()
    embedding_obj.aembed_query = AsyncMock(return_value=[0.1, 0.2])
    mocker.patch.object(CachedAsyncEmbeddingFactory, "create", AsyncMock(return_value=(embedding_obj, 1536)))
    sparse_encoder = AsyncMock()
    sparse_encoder.aencode_queries = AsyncMock(return_value={"indices": [1], "values": [0.5]})
    mocker.patch("tilellm.store.qdrant.qdrant_repository_local.TiledeskSparseEncoders", return_value=sparse_encoder)

    repo = QdrantRepository()
    mocker.patch.object(repo, "create_index", AsyncMock(return_value=vector_store))
    mocker.patch.object(repo, "get_embeddings_dimension", AsyncMock(return_value=1536))

    await repo.get_chunks_from_repo(_hybrid_qa(top_k=45))

    assert _prefetch_limits(client) == [45, 45]


@pytest.mark.asyncio
async def test_perform_hybrid_search_prefetch_limit_matches_top_k():
    client = _client_returning_one_point()

    await QdrantRepository().perform_hybrid_search(_hybrid_qa(top_k=30), client, [0.1, 0.2],
                                                   {"indices": [1], "values": [0.5]})

    assert _prefetch_limits(client) == [30, 30]


@pytest.mark.asyncio
async def test_search_community_report_prefetch_limit_matches_top_k():
    client = _client_returning_one_point()

    await QdrantRepository().search_community_report(_hybrid_qa(top_k=25), client, [0.1, 0.2],
                                                     {"indices": [1], "values": [0.5]})

    assert _prefetch_limits(client) == [25, 25]


# ---------------------------------------------------------------------------
# get_chunks_by_index — targeted fetch of specific chunks of one document, used to
# re-attach the neighbours of a retrieved chunk (a table row that docling split
# into consecutive sections, e.g. requirement | justification | applicable).
# ---------------------------------------------------------------------------

def _neighbours_client(mocker, points):
    """get_chunks_by_index must reuse the repository's cached client (a new
    AsyncQdrantClient per call meant one version-check round trip and a never-closed
    connection per criterion) — under its own cache key, so a wrapper built without
    embeddings can never be handed to ingestion."""
    client = MagicMock()
    client.scroll = MagicMock(return_value=(points, None))
    wrapper = MagicMock()
    wrapper.get_client = AsyncMock(return_value=client)
    factory = mocker.patch.object(QdrantRepository, "create_index_cache_wrapper",
                                  AsyncMock(return_value=wrapper))
    return client, factory


@pytest.mark.asyncio
async def test_get_chunks_by_index_fetches_only_the_requested_chunks(mocker):
    point = MagicMock()
    point.id = "p-79"
    point.payload = {"page_content": "ISO 10993-5 citotossicità",
                     "metadata": {"doc_id": "doc-1", "chunk_index": 79}}
    client, factory = _neighbours_client(mocker, [point])
    constructor = mocker.patch('tilellm.store.qdrant.qdrant_repository_local.AsyncQdrantClient')
    engine = Engine(name="qdrant", deployment="local", host="localhost", port=6333,
                    index_name="test-collection", apikey=None)

    docs = await QdrantRepository().get_chunks_by_index(engine, "ns", "doc-1", [77, 79])

    assert [d.page_content for d in docs] == ["ISO 10993-5 citotossicità"]
    assert docs[0].metadata["chunk_index"] == 79
    conditions = {c.key: c.match for c in client.scroll.call_args.kwargs["scroll_filter"].must}
    assert conditions["metadata.doc_id"].value == "doc-1"
    assert conditions["metadata.namespace"].value == "ns"
    assert conditions["metadata.chunk_index"].any == [77, 79]
    assert client.scroll.call_args.kwargs["limit"] == 2
    constructor.assert_not_called()  # no fresh client per call
    assert factory.call_args.kwargs["cache_suffix"] == "neighbours"


@pytest.mark.asyncio
async def test_get_chunks_by_index_with_no_indexes_skips_the_query(mocker):
    client, _ = _neighbours_client(mocker, [])
    engine = Engine(name="qdrant", deployment="local", host="localhost", port=6333,
                    index_name="test-collection", apikey=None)

    assert await QdrantRepository().get_chunks_by_index(engine, "ns", "doc-1", []) == []
    client.scroll.assert_not_called()


@pytest.mark.asyncio
async def test_get_vector_store_skips_langchain_sync_embedding_validation():
    """QdrantVectorStore's collection validation embeds a dummy text with a SYNC
    embed_documents call — from inside the event loop, on every retrieval. That sync
    call deadlocked a gunicorn worker on a real run. _ensure_client already checks
    the collection asynchronously."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from tilellm.store.qdrant import qdrant_repository_local as mod

    wrapper = mod.CachedVectorStore(MagicMock(index_name="c"), MagicMock(), 1024)
    with patch.object(wrapper, "_ensure_client", new=AsyncMock()), \
         patch.object(mod, "QdrantVectorStore") as vs_cls:
        await wrapper.get_vector_store()

    assert vs_cls.call_args.kwargs["validate_collection_config"] is False
