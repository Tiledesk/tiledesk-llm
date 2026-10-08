"""
Milvus filter expressions were built with f-strings from request values
(namespace, metadata id, source, tags, metadata_filter keys and values): a
namespace like  x" or metadata["namespace"] != "  turned delete_namespace into
"delete the whole collection" (every tenant). String values are now escaped
and field names must be identifiers.
"""
from unittest.mock import MagicMock

import pytest

from tilellm.models import Engine
from tilellm.models.schemas import RepositoryNamespace
from tilellm.store.milvus.milvus_repository import MilvusRepository

EVIL = 'x" or metadata["namespace"] != "'


@pytest.fixture
def client(monkeypatch):
    c = MagicMock()
    c.query.return_value = []

    async def fake_client(engine):
        return c

    monkeypatch.setattr(MilvusRepository, "_get_milvus_client", staticmethod(fake_client))
    return c


ENGINE = Engine(name="milvus", host="localhost", port=19530, index_name="coll", deployment="local")



@pytest.mark.asyncio
async def test_delete_namespace_escapes_the_namespace(client):
    await MilvusRepository().delete_namespace(RepositoryNamespace(namespace=EVIL, engine=ENGINE))

    expr = client.delete.call_args.kwargs["filter"]
    assert expr == 'metadata["namespace"] == "x\\" or metadata[\\"namespace\\"] != \\""'


@pytest.mark.asyncio
@pytest.mark.parametrize("call", [
    lambda r: r.delete_ids_namespace(ENGINE, EVIL, "ns"),
    lambda r: r.get_ids_namespace(ENGINE, "id", EVIL),
    lambda r: r.get_all_obj_namespace(ENGINE, EVIL),
    lambda r: r.get_desc_namespace(ENGINE, EVIL),
    lambda r: r.get_sources_namespace(ENGINE, EVIL, "ns"),
])
async def test_every_string_filter_escapes_request_values(client, call):
    await call(MilvusRepository())

    target = client.delete if client.delete.called else client.query
    expr = target.call_args.kwargs["filter"]
    assert 'x\\" or metadata[\\"namespace\\"] != \\"' in expr


def test_build_filter_escapes_namespace_tags_and_values():
    repo = MilvusRepository()
    expr = repo.build_filter(
        EVIL,
        filter_dict={"tags": {"$in": [EVIL]}},
        metadata_filter={"source": {"$eq": EVIL}},
    )
    assert expr.count('x\\" or metadata[\\"namespace\\"] != \\"') == 3


@pytest.mark.parametrize("bad", [
    {'source"] == "a" or metadata["x': {"$eq": "a"}},   # field name breakout
    {"source": {"$eq": {"nested": "x' or 1==1 or '"}}},  # non-scalar value
])
def test_build_filter_rejects_unsafe_fields_and_values(bad):
    with pytest.raises(ValueError):
        MilvusRepository().build_filter("ns", metadata_filter=bad)


def test_build_filter_keeps_numbers_and_plain_values_working():
    expr = MilvusRepository().build_filter("ns", metadata_filter={"page": {"$gte": 3}, "lang": {"$in": ["it", 2]}})
    assert 'metadata["page"] >= 3' in expr
    assert 'metadata["lang"] == "it"' in expr and 'metadata["lang"] == 2' in expr


def test_in_operator_rejects_non_scalar_values():
    with pytest.raises(ValueError):
        MilvusRepository().build_filter("ns", metadata_filter={"lang": {"$in": ["it", {"x": "y"}]}})
