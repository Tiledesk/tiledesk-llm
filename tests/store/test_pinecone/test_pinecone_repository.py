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
    mock_pinecone_client.describe_index.return_value.dimension = 1536

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
    mock_sync_pinecone_client.describe_index.return_value.dimension = 1536
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
    mock_pinecone_client.describe_index.return_value.dimension = 1536

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
    mock_index_async.__aenter__.return_value = mock_index_in_block
    mock_index_async.__aexit__.return_value = False # To not suppress exceptions
    

    # Now, configure the describe_index_stats on the object yielded by the context manager
    mock_index_stats_response_instance = MockIndexStatsResponse(namespaces_data={
        "my-namespace": {"vector_count": 100}
    })
    mock_index_in_block.describe_index_stats = AsyncMock(return_value=mock_index_stats_response_instance)

    # Mock the query response on the object yielded by the context manager
    mock_query_response = {
        'matches': [
            {
                'id': 'chunk1',
                'metadata': {
                    'id': metadata_id,
                    'source': 'source1',
                    'type': 'type1',
                    'date': '2025-01-01',
                    'text': 'content of chunk 1'
                }
            },
            {
                'id': 'chunk2',
                'metadata': {
                    'id': metadata_id,
                    'source': 'source1',
                    'type': 'type1',
                    'date': '2025-01-02',
                    'text': 'content of chunk 2'
                }
            }
        ]
    }
    mock_index_in_block.query = AsyncMock(return_value=mock_query_response)

    mocker.patch('pinecone.Pinecone', return_value=mock_pinecone_client)
    mock_pinecone_client.IndexAsyncio.return_value = mock_index_async
    mock_pinecone_client.describe_index.return_value.host = "dummy-host"
    mock_pinecone_client.describe_index.return_value.dimension = 1536

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


@pytest.mark.asyncio
async def test_get_all_obj_namespace_passes_through_full_metadata(mocker):
    """RepositoryQueryResult.metadata must carry the full raw metadata dict
    (already fetched with include_metadata=True) — needed so callers like
    lgraph's build_lgraph can see custom fields (e.g. page_number, doc_type)
    that the fixed id/source/type/date fields don't cover."""
    namespace = "my-namespace"
    mock_engine = Engine(
        name="pinecone", type="serverless", apikey="fake-api-key",
        index_name="test-index", text_key="text",
    )

    mock_pinecone_client = MagicMock()
    mock_index_async = AsyncMock()
    mock_index_in_block = MagicMock()
    mock_index_async.__aenter__.return_value = mock_index_in_block
    mock_index_async.__aexit__.return_value = False

    mock_index_in_block.describe_index_stats = AsyncMock(
        return_value=MockIndexStatsResponse(namespaces_data={namespace: {"vector_count": 1}})
    )
    raw_metadata = {
        "id": "doc1", "source": "src1", "type": "regex_custom",
        "date": "2026-07-23", "page_number": 7, "doc_type": "delibera",
    }
    mock_index_in_block.query = AsyncMock(return_value={
        'matches': [{'id': 'chunk1', 'metadata': raw_metadata}]
    })

    mocker.patch('pinecone.Pinecone', return_value=mock_pinecone_client)
    mock_pinecone_client.IndexAsyncio.return_value = mock_index_async
    mock_pinecone_client.describe_index.return_value.host = "dummy-host"
    mock_pinecone_client.describe_index.return_value.dimension = 1536

    repo = PineconeRepositoryServerless()
    result = await repo.get_all_obj_namespace(mock_engine, namespace)

    assert result.matches[0].metadata == raw_metadata


@pytest.mark.asyncio
async def test_get_chunks_from_repo_exposes_chunk_ids(mocker):
    """RetrievalChunksResult.chunk_ids must carry each match's vector id (same id
    space as get_all_obj_namespace) — needed as PPR seed_chunk_ids for lgraph
    hybrid retrieval. Vector ids were already read onto Document.id and discarded."""
    from langchain_core.documents import Document
    from pydantic import SecretStr
    from tilellm.models import QuestionAnswer

    mock_engine = Engine(
        name="pinecone", type="serverless", apikey="fake-api-key",
        index_name="test-index", text_key="text",
    )
    question_answer = QuestionAnswer(
        question="q", namespace="ns", engine=mock_engine, search_type="similarity", top_k=2,
        gptkey=SecretStr("test-key"),
    )

    mock_vector_store = AsyncMock()
    mock_vector_store.asearch = AsyncMock(return_value=[
        Document(id="vec-123", metadata={"source": "s1"}, page_content="hello"),
    ])

    mock_embedding_obj = AsyncMock()
    mock_factory = AsyncMock()
    mock_factory.create = AsyncMock(return_value=(mock_embedding_obj, 1536))
    mocker.patch(
        "tilellm.shared.embeddings.embedding_client_manager.CachedAsyncEmbeddingFactory",
        return_value=mock_factory,
    )

    repo = PineconeRepositoryServerless()
    mocker.patch.object(repo, "create_index", AsyncMock(return_value=mock_vector_store))

    result = await repo.get_chunks_from_repo(question_answer)

    assert result.chunk_ids == ["vec-123"]


class TestNormalizeUpsertMetadata:
    """Shared base-class normalization, reused by Pod, Serverless and the generic aadd_documents path."""

    def test_defaults_missing_tags_to_empty_list(self):
        from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase

        metadata = {"id": "doc1"}
        result = PineconeRepositoryBase._normalize_upsert_metadata(metadata)

        assert result["tags"] == []

    def test_normalizes_none_tags(self):
        from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase

        metadata = {"id": "doc1", "tags": None}
        result = PineconeRepositoryBase._normalize_upsert_metadata(metadata)

        assert result["tags"] == []

    def test_preserves_existing_tags(self):
        from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase

        metadata = {"id": "doc1", "tags": ["billing"]}
        result = PineconeRepositoryBase._normalize_upsert_metadata(metadata)

        assert result["tags"] == ["billing"]

    def test_does_not_touch_namespace(self):
        """Pinecone namespaces are native (passed via namespace= param), not stored in metadata."""
        from tilellm.store.pinecone.pinecone_repository_base import PineconeRepositoryBase

        metadata = {"id": "doc1"}
        result = PineconeRepositoryBase._normalize_upsert_metadata(metadata)

        assert "namespace" not in result


class TestBuildRegexCustomChunksServerless:
    """Shared by add_item and add_item_hybrid (PineconeRepositoryServerless) so both stay consistent."""

    @staticmethod
    def _make_item(**overrides):
        from types import SimpleNamespace
        defaults = dict(
            id="doc1", source="https://example.com/doc", type="regex_custom",
            embedding="text-embedding-3-small", tags=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_sets_file_name_page_and_stringified_embedding(self):
        from langchain_core.documents import Document

        item = self._make_item()
        documents = [Document(page_content="chunk one", metadata={})]

        chunks = PineconeRepositoryServerless._build_regex_custom_chunks(item, documents)

        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert meta["file_name"]
        assert meta["page"] == 1
        assert meta["embedding"] == "text-embedding-3-small"
        assert isinstance(meta["embedding"], str)

    def test_preserves_existing_file_name_and_page(self):
        from langchain_core.documents import Document

        item = self._make_item()
        documents = [Document(page_content="chunk one", metadata={"file_name": "custom.txt", "page": 3})]

        chunks = PineconeRepositoryServerless._build_regex_custom_chunks(item, documents)

        assert chunks[0].metadata["file_name"] == "custom.txt"
        assert chunks[0].metadata["page"] == 3

    def test_includes_tags_when_present(self):
        from langchain_core.documents import Document

        item = self._make_item(tags=["billing"])
        documents = [Document(page_content="chunk one", metadata={})]

        chunks = PineconeRepositoryServerless._build_regex_custom_chunks(item, documents)

        assert chunks[0].metadata["tags"] == ["billing"]


@pytest.mark.asyncio
async def test_serverless_upsert_vector_store_defaults_tags_to_empty_list():
    from langchain_core.documents import Document

    chunks = [Document(page_content="c1", metadata={"id": "doc1"})]
    mock_vector_store = MagicMock()
    mock_vector_store.aadd_documents = AsyncMock(return_value=["id1"])

    await PineconeRepositoryServerless.upsert_vector_store(
        vector_store=mock_vector_store, chunks=chunks, metadata_id="doc1", namespace="tenant-a"
    )

    assert chunks[0].metadata["tags"] == []


@pytest.mark.asyncio
async def test_serverless_upsert_vector_store_hybrid_defaults_tags_to_empty_list():
    from langchain_core.documents import Document

    chunks = [Document(page_content="c1", metadata={"id": "doc1"})]
    mock_indice = MagicMock()
    mock_indice.upsert = AsyncMock()
    mock_engine = Engine(name="pinecone", type="serverless", apikey="fake-api-key", index_name="test-index", text_key="text")
    mock_embeddings = MagicMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])

    await PineconeRepositoryServerless.upsert_vector_store_hybrid(
        indice=mock_indice,
        contents=["c1"],
        chunks=chunks,
        metadata_id="doc1",
        engine=mock_engine,
        namespace="tenant-a",
        embeddings=mock_embeddings,
        sparse_vectors=[{"indices": [0], "values": [1.0]}],
    )

    vector_tuples = mock_indice.upsert.call_args.kwargs["vectors"]
    assert vector_tuples[0]["metadata"]["tags"] == []
    # Original chunk metadata must be untouched (hybrid builds a copy).
    assert "tags" not in chunks[0].metadata


class TestBuildRegexCustomChunksPod:
    """PineconeRepositoryPod's regex_custom used to store an unclean str(item.embedding)
    (e.g. the LlmEmbeddingModel repr) instead of the resolved model name, and never set
    file_name/page like the standard url/pdf/etc branch does."""

    @staticmethod
    def _make_item(**overrides):
        from types import SimpleNamespace
        defaults = dict(id="doc1", source="https://example.com/doc", type="regex_custom", tags=None)
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_sets_file_name_page_and_uses_resolved_embedding_name(self):
        from langchain_core.documents import Document
        from tilellm.store.pinecone.pinecone_repository_pod import PineconeRepositoryPod

        item = self._make_item()
        documents = [Document(page_content="chunk one", metadata={})]

        chunks = PineconeRepositoryPod._build_regex_custom_chunks(item, documents, "text-embedding-3-small")

        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert meta["file_name"]
        assert meta["page"] == 1
        assert meta["embedding"] == "text-embedding-3-small"

    def test_preserves_existing_file_name_and_page(self):
        from langchain_core.documents import Document
        from tilellm.store.pinecone.pinecone_repository_pod import PineconeRepositoryPod

        item = self._make_item()
        documents = [Document(page_content="chunk one", metadata={"file_name": "custom.txt", "page": 3})]

        chunks = PineconeRepositoryPod._build_regex_custom_chunks(item, documents, "text-embedding-3-small")

        assert chunks[0].metadata["file_name"] == "custom.txt"
        assert chunks[0].metadata["page"] == 3

    def test_includes_tags_when_present(self):
        from langchain_core.documents import Document
        from tilellm.store.pinecone.pinecone_repository_pod import PineconeRepositoryPod

        item = self._make_item(tags=["billing"])
        documents = [Document(page_content="chunk one", metadata={})]

        chunks = PineconeRepositoryPod._build_regex_custom_chunks(item, documents, "text-embedding-3-small")

        assert chunks[0].metadata["tags"] == ["billing"]

