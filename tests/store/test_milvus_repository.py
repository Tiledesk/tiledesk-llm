"""
MilvusRepository against its current client contract: every operation gets a
MilvusClient from _get_milvus_client(engine) and calls client.query /
client.delete with a filter expression. (The previous tests mocked the old
Collection/create_index API, so most of them reached a real Milvus on
localhost:19530.) Filter escaping is covered in test_milvus_filter_injection.py.
"""
from unittest.mock import MagicMock

import pytest

from tilellm.models import Engine
from tilellm.models.schemas import RepositoryNamespace
from tilellm.store.milvus.milvus_repository import MilvusRepository

ENGINE = Engine(name="milvus", deployment="local", host="localhost", port=19530,
                index_name="test-collection", database="default")

ROWS = [
    {"id": "chunk-1", "page_content": "Content 1",
     "metadata": {"id": "doc-1", "namespace": "ns1", "source": "source1", "type": "pdf", "date": "2024-01-01"}},
    {"id": "chunk-2", "page_content": "Content 2",
     "metadata": {"id": "doc-1", "namespace": "ns1", "source": "source1", "type": "pdf", "date": "2024-01-02"}},
    {"id": "chunk-3", "page_content": "Content 3",
     "metadata": {"id": "doc-2", "namespace": "ns2", "source": "source2", "type": "url"}},
]


@pytest.fixture
def client(monkeypatch):
    c = MagicMock()
    c.query.return_value = ROWS

    async def fake_client(engine):
        assert engine is ENGINE
        return c

    monkeypatch.setattr(MilvusRepository, "_get_milvus_client", staticmethod(fake_client))
    return c


# ---------------------------------------------------------------------------
# list_namespaces
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_namespaces_counts_vectors_per_namespace(client):
    result = await MilvusRepository().list_namespaces(ENGINE)

    assert {ns.namespace: ns.vector_count for ns in result.namespaces} == {"ns1": 2, "ns2": 1}
    assert client.query.call_args.kwargs["collection_name"] == "test-collection"


@pytest.mark.asyncio
async def test_list_namespaces_empty(client):
    client.query.return_value = []

    result = await MilvusRepository().list_namespaces(ENGINE)

    assert result.namespaces == []


@pytest.mark.asyncio
async def test_list_namespaces_propagates_errors(client):
    client.query.side_effect = RuntimeError("milvus down")

    with pytest.raises(RuntimeError, match="milvus down"):
        await MilvusRepository().list_namespaces(ENGINE)


# ---------------------------------------------------------------------------
# deletes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_namespace(client):
    await MilvusRepository().delete_namespace(RepositoryNamespace(namespace="ns1", engine=ENGINE))

    client.delete.assert_called_once_with(collection_name="test-collection", filter='metadata["namespace"] == "ns1"')


@pytest.mark.asyncio
async def test_delete_ids_namespace(client):
    await MilvusRepository().delete_ids_namespace(ENGINE, "doc-1", "ns1")

    client.delete.assert_called_once_with(
        collection_name="test-collection",
        filter='metadata["id"] == "doc-1" and metadata["namespace"] == "ns1"',
    )


@pytest.mark.asyncio
async def test_delete_chunk_id_namespace(client):
    await MilvusRepository().delete_chunk_id_namespace(ENGINE, "chunk-1", "ns1")

    client.delete.assert_called_once_with(collection_name="test-collection", ids=["chunk-1"])


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_ids_namespace(client):
    client.query.return_value = ROWS[:2]

    result = await MilvusRepository().get_ids_namespace(ENGINE, "doc-1", "ns1")

    kwargs = client.query.call_args.kwargs
    assert kwargs["filter"] == 'metadata["id"] == "doc-1" and metadata["namespace"] == "ns1"'
    assert kwargs["output_fields"] == ["id", "metadata", "page_content"]
    first = result.matches[0]
    assert (first.id, first.metadata_id, first.metadata_source, first.metadata_type, first.date, first.text) == (
        "chunk-1", "doc-1", "source1", "pdf", "2024-01-01", "Content 1")
    assert len(result.matches) == 2


@pytest.mark.asyncio
async def test_get_all_obj_namespace_with_and_without_text(client):
    client.query.return_value = ROWS[:2]
    repo = MilvusRepository()

    without_text = await repo.get_all_obj_namespace(ENGINE, "ns1")
    assert client.query.call_args.kwargs["output_fields"] == ["id", "metadata"]
    assert without_text.matches[0].text is None

    with_text = await repo.get_all_obj_namespace(ENGINE, "ns1", with_text=True)
    assert "page_content" in client.query.call_args.kwargs["output_fields"]
    assert with_text.matches[0].text == "Content 1"


@pytest.mark.asyncio
async def test_get_desc_namespace_groups_chunks_by_document(client):
    client.query.return_value = ROWS[:2]

    result = await MilvusRepository().get_desc_namespace(ENGINE, "ns1")

    assert result.namespace_desc.vector_count == 2
    assert [(i.metadata_id, i.chunks_count) for i in result.ids] == [("doc-1", 2)]


@pytest.mark.asyncio
async def test_get_sources_namespace(client):
    client.query.return_value = ROWS[2:]

    result = await MilvusRepository().get_sources_namespace(ENGINE, "source2", "ns2")

    assert client.query.call_args.kwargs["filter"] == (
        'metadata["source"] == "source2" and metadata["namespace"] == "ns2"')
    assert [m.id for m in result.matches] == ["chunk-3"]
    assert result.matches[0].date == "Date not defined"
