"""RedisRepository on a real Redis with the Query Engine (fakeredis has no FT.SEARCH/KNN).

Skipped when REDIS_VS_TEST_URL (default redis://localhost:6379) has no search module. Every test uses a
throwaway index_name and drops what it created.
"""
import asyncio
import hashlib
import os
import uuid
from urllib.parse import urlparse

import pytest
import redis
from langchain_core.documents import Document

from tilellm.models import QuestionAnswer
from tilellm.models.schemas import RepositoryNamespace
from tilellm.models.vector_store import Engine
from tilellm.store.redis.redis_vector_repository import RedisRepository, filter_to_query

_URL = urlparse(os.environ.get("REDIS_VS_TEST_URL", "redis://localhost:6379"))


def _search_available():
    try:
        mods = redis.Redis(host=_URL.hostname, port=_URL.port or 6379, socket_timeout=1).execute_command("MODULE", "LIST")
        return any(b"search" in m[1].lower() for m in mods)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _search_available(), reason="no Redis with the search module")


class FakeEmbeddings:
    """Deterministic 8-dim vectors: identical text -> identical vector."""
    dim = 8

    def _vec(self, text):
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255 + 0.01 for b in h[:self.dim]]

    async def aembed_documents(self, texts):
        return [self._vec(t) for t in texts]

    async def aembed_query(self, text):
        return self._vec(text)


@pytest.fixture
def engine():
    e = Engine(name="redis", host=_URL.hostname, port=_URL.port or 6379, index_name=f"pytest-{uuid.uuid4().hex[:8]}")
    yield e
    r = redis.Redis(host=_URL.hostname, port=_URL.port or 6379)
    for name in r.execute_command("FT._LIST"):
        if name.decode().startswith(f"tlvs:{e.index_name}:"):
            r.execute_command("FT.DROPINDEX", name, "DD")


def _docs(*specs):
    return [Document(page_content=text, metadata=meta) for text, meta in specs]


async def _seed(repo, engine, ns, *specs):
    await repo.aadd_documents(engine, _docs(*specs), ns, FakeEmbeddings())


@pytest.mark.asyncio
async def test_nearest_chunk_first_and_namespaces_isolated(engine):
    repo, emb = RedisRepository(), FakeEmbeddings()
    await _seed(repo, engine, "a", ("alpha", {"id": "d1"}), ("beta", {"id": "d2"}), ("gamma", {"id": "d3"}))
    await _seed(repo, engine, "a:b", ("alpha", {"id": "other"}))
    hits = await repo.knn(engine, "a", await emb.aembed_query("alpha"), 3)
    assert hits[0][0].page_content == "alpha" and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert {d.metadata["id"] for d, _ in hits} == {"d1", "d2", "d3"}  # nothing from namespace "a:b"
    names = {n.namespace: n.vector_count for n in (await repo.list_namespaces(engine)).namespaces}
    assert names == {"a": 3, "a:b": 1}


@pytest.mark.asyncio
async def test_filters_on_indexed_fields(engine):
    repo, emb = RedisRepository(), FakeEmbeddings()
    await _seed(repo, engine, "ns",
                ("t1", {"id": "d1", "tags": ["x", "y,z"], "doc_type": "chiarimento", "digest_date_from": "2026-07-27"}),
                ("t2", {"id": "d2", "tags": ["x"], "digest_date_from": "2026-07-29"}),
                ("t3", {"id": "d3", "tags": ["w"], "chunk_index": 2}))
    q = await emb.aembed_query("t1")

    async def ids(flt):
        return {d.metadata["id"] for d, _ in await repo.knn(engine, "ns", q, 10, flt)}

    assert await ids({"tags": {"$in": ["x"]}}) == {"d1", "d2"}
    assert await ids({"tags": {"$in": ["y,z"]}}) == {"d1"}
    assert await ids({"tags": {"$nin": ["x"]}}) == {"d3"}
    assert await ids({"doc_type": {"$ne": "chiarimento"}}) == {"d2", "d3"}  # missing field passes $ne
    assert await ids({"digest_date_from": {"$gte": "2026-07-28"}}) == {"d2"}
    assert await ids({"chunk_index": {"$in": [2]}}) == {"d3"}
    assert await ids({"$or": [{"id": "d1"}, {"id": "d3"}]}) == {"d1", "d3"}
    assert await ids({"$not": {"tags": {"$in": ["x"]}}}) == {"d3"}


def test_filter_on_non_indexed_field_is_rejected():
    with pytest.raises(ValueError, match="non-indexed"):
        filter_to_query({"author": {"$eq": "x"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("evil", ["x} | @id:{*", "*", "d1 | d2", "-d1", "\\", "\"d1\""])
async def test_filter_values_cannot_change_the_query(engine, evil):
    repo, emb = RedisRepository(), FakeEmbeddings()
    await _seed(repo, engine, "ns", ("t1", {"id": "d1"}), ("t2", {"id": "d2"}))
    hits = await repo.knn(engine, "ns", await emb.aembed_query("t1"), 10, {"id": {"$eq": evil}})
    assert hits == []


@pytest.mark.asyncio
async def test_reads_are_not_capped_at_10k(engine):
    repo = RedisRepository()
    n = 10_050
    docs = [Document(page_content=f"c{i}", metadata={"id": "big", "source": "s"}) for i in range(n)]
    await repo._write(engine, "ns", docs, [[1.0, float(i % 7) + 1, 0.5, 0.25] for i in range(n)])
    assert len((await repo.get_all_obj_namespace(engine, "ns")).matches) == n
    assert len((await repo.get_ids_namespace(engine, "big", "ns")).matches) == n
    assert (await repo.get_desc_namespace(engine, "ns")).namespace_desc.vector_count == n


@pytest.mark.asyncio
async def test_deletes(engine):
    repo = RedisRepository()
    await _seed(repo, engine, "ns", ("t1", {"id": "d1"}), ("t2", {"id": "d1"}), ("t3", {"id": "d2"}))
    await repo.delete_ids_namespace(engine, "d1", "ns")
    assert [m.metadata_id for m in (await repo.get_all_obj_namespace(engine, "ns")).matches] == ["d2"]
    chunk = (await repo.get_ids_namespace(engine, "d2", "ns")).matches[0].id
    await repo.delete_chunk_id_namespace(engine, chunk, "ns")
    assert (await repo.get_all_obj_namespace(engine, "ns")).matches == []
    await repo.delete_namespace(RepositoryNamespace(namespace="ns", engine=engine))
    assert (await repo.list_namespaces(engine)).namespaces == []


@pytest.mark.asyncio
async def test_missing_namespace_reads_empty_and_deletes_noop(engine):
    repo = RedisRepository()
    assert await repo.knn(engine, "nope", [1.0] * 8, 5) == []
    assert (await repo.get_all_obj_namespace(engine, "nope")).matches == []
    assert await repo.get_by_doc_id(engine, "nope", "d") == []
    await repo.delete_namespace(RepositoryNamespace(namespace="nope", engine=engine))
    await repo.delete_ids_namespace(engine, "d", "nope")


@pytest.mark.asyncio
async def test_dimension_mismatch_is_an_error_not_silent_loss(engine):
    repo = RedisRepository()
    await _seed(repo, engine, "ns", ("t1", {"id": "d1"}))
    with pytest.raises(ValueError, match="dimension"):
        await repo._write(engine, "ns", _docs(("t2", {"id": "d2"})), [[1.0, 2.0]])


@pytest.mark.asyncio
async def test_aadd_documents_replaces_namespace_or_one_document(engine):
    repo = RedisRepository()
    await _seed(repo, engine, "ns", ("t1", {"id": "d1"}), ("t2", {"id": "d2"}))
    await repo.aadd_documents(engine, _docs(("t3", {"id": "d2"})), "ns", FakeEmbeddings(), metadata_id="d2")
    assert sorted(m.text for m in (await repo.get_all_obj_namespace(engine, "ns", True)).matches) == ["t1", "t3"]
    await _seed(repo, engine, "ns", ("t9", {"id": "d9"}))
    assert [m.text for m in (await repo.get_all_obj_namespace(engine, "ns", True)).matches] == ["t9"]


@pytest.mark.asyncio
async def test_get_chunks_from_repo_and_hybrid_shape(engine):
    repo, emb = RedisRepository(), FakeEmbeddings()
    await _seed(repo, engine, "ns", ("alpha", {"id": "d1", "tags": ["k"], "doc_type": "chiarimento"}),
                ("alpha", {"id": "d2", "tags": ["k"]}), ("beta", {"id": "d3"}))
    qa = QuestionAnswer(question="alpha", namespace="ns", engine=engine, top_k=5, tags=["k"])
    qa._metadata_filter = {"doc_type": {"$ne": "chiarimento"}}
    res = await RedisRepository.get_chunks_from_repo.__wrapped__(repo, qa, embedding_obj=emb)
    assert [m["id"] for m in res.metadata] == ["d2"] and res.chunks == ["alpha"] and res.chunk_ids[0]

    qa2 = QuestionAnswer(question="alpha", namespace="ns", engine=engine, top_k=2)
    out = await repo.perform_hybrid_search(qa2, None, await emb.aembed_query("alpha"), None)
    assert len(out["matches"]) == 2 and all(m["metadata"]["text"] == "alpha" for m in out["matches"])


@pytest.mark.asyncio
async def test_as_retriever_with_namespace_in_search_kwargs(engine):
    repo, emb = RedisRepository(), FakeEmbeddings()
    await _seed(repo, engine, "ns", ("alpha", {"id": "d1"}), ("beta", {"id": "d2"}))
    store = await repo.create_index(engine, emb)
    docs = await store.as_retriever(search_kwargs={"k": 1, "namespace": "ns"}).ainvoke("beta")
    assert [d.page_content for d in docs] == ["beta"]


def test_client_is_not_reused_across_event_loops(engine):
    # Each asyncio.run() is a new loop: a client cached from the first loop would fail on the second.
    repo = RedisRepository()

    async def roundtrip(text):
        await _seed(repo, engine, "ns", (text, {"id": text}))
        return await repo.knn(engine, "ns", await FakeEmbeddings().aembed_query(text), 1)

    assert asyncio.run(roundtrip("one"))[0][0].page_content == "one"
    assert asyncio.run(roundtrip("two"))[0][0].page_content == "two"


@pytest.mark.asyncio
async def test_add_item_replaces_document_and_keeps_it_when_fetch_fails(engine):
    from tilellm.models import ItemSingle
    from tilellm.store.vector_store_repository import VectorStoreIndexingError

    repo, emb = RedisRepository(), FakeEmbeddings()
    add = RedisRepository.add_item.__wrapped__

    def item(**kw):
        return ItemSingle(id="doc", namespace="ns", engine=engine, source="note", **kw)

    await add(repo, item(type="text", content="first version"), embedding_obj=emb)
    await add(repo, item(type="text", content="second version"), embedding_obj=emb)
    texts = [m.text for m in (await repo.get_ids_namespace(engine, "doc", "ns")).matches]
    assert texts == ["second version"]

    with pytest.raises(VectorStoreIndexingError):  # loopback URL: refused before any delete
        await add(repo, ItemSingle(id="doc", namespace="ns", engine=engine, type="url",
                                   source="http://127.0.0.1:1/x"), embedding_obj=emb)
    assert [m.text for m in (await repo.get_ids_namespace(engine, "doc", "ns")).matches] == ["second version"]


@pytest.mark.asyncio
async def test_chunk_ids_with_colon_round_trip_and_delete_the_right_chunk(engine):
    repo, emb = RedisRepository(), FakeEmbeddings()
    docs = [Document(id="c1", page_content="t1", metadata={"id": "docA"}),
            Document(id="x:c1", page_content="t2", metadata={"id": "docB"})]
    await repo.aadd_documents(engine, docs, "ns", emb)
    ids = {m.id for m in (await repo.get_all_obj_namespace(engine, "ns")).matches}
    assert ids == {"c1", "x:c1"}
    assert {d.id for d, _ in await repo.knn(engine, "ns", await emb.aembed_query("t2"), 2)} == {"c1", "x:c1"}
    await repo.delete_ids_namespace(engine, "docB", "ns")
    assert [m.id for m in (await repo.get_all_obj_namespace(engine, "ns")).matches] == ["c1"]


@pytest.mark.asyncio
async def test_unresponsive_host_fails_within_the_timeout(monkeypatch):
    import socket
    import time
    import tilellm.store.redis.redis_vector_repository as mod

    monkeypatch.setattr(mod, "SOCKET_TIMEOUT", 0.5)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)  # accepts the connection, never answers
    try:
        e = Engine(name="redis", host="127.0.0.1", port=srv.getsockname()[1], index_name="blackhole")
        t = time.monotonic()
        with pytest.raises((redis.exceptions.TimeoutError, redis.exceptions.ConnectionError)):
            await RedisRepository().knn(e, "ns", [1.0] * 8, 1)
        assert time.monotonic() - t < 5
    finally:
        srv.close()


@pytest.mark.asyncio
async def test_client_cache_is_bounded():
    import tilellm.store.redis.redis_vector_repository as mod
    from pydantic import SecretStr

    for i in range(mod.MAX_CLIENTS_PER_LOOP + 10):
        mod._client(Engine(name="redis", host="localhost", apikey=SecretStr(f"pw{i}")))
    assert len(mod._clients[asyncio.get_running_loop()]) <= mod.MAX_CLIENTS_PER_LOOP


class FourDimEmbeddings(FakeEmbeddings):
    dim = 4


@pytest.mark.asyncio
async def test_reindex_with_other_dimension_fails_without_losing_the_document(engine):
    from tilellm.models import ItemSingle
    from tilellm.store.vector_store_repository import VectorStoreIndexingError

    repo = RedisRepository()
    add = RedisRepository.add_item.__wrapped__
    item = ItemSingle(id="doc", namespace="ns", engine=engine, source="n", type="text", content="kept")
    await add(repo, item, embedding_obj=FakeEmbeddings())
    with pytest.raises(VectorStoreIndexingError):
        await add(repo, item.model_copy(update={"content": "new"}), embedding_obj=FourDimEmbeddings())
    with pytest.raises(ValueError, match="dimension"):
        await repo.aadd_documents(engine, _docs(("new", {"id": "doc"})), "ns", FourDimEmbeddings(), metadata_id="doc")
    assert [m.text for m in (await repo.get_ids_namespace(engine, "doc", "ns")).matches] == ["kept"]


@pytest.mark.asyncio
async def test_non_finite_numeric_metadata_does_not_hide_the_chunk(engine):
    repo = RedisRepository()
    await _seed(repo, engine, "ns", ("t1", {"id": "d1", "page": "nan"}), ("t2", {"id": "d2", "chunk_index": float("inf")}),
                ("t3", {"id": "d3", "page": float("nan")}))
    assert len((await repo.get_all_obj_namespace(engine, "ns")).matches) == 3


@pytest.mark.asyncio
async def test_empty_logical_filters(engine):
    repo, emb = RedisRepository(), FakeEmbeddings()
    await _seed(repo, engine, "ns", ("t1", {"id": "d1", "chunk_index": 1}))
    q = await emb.aembed_query("t1")
    assert len(await repo.knn(engine, "ns", q, 5, {"$and": []})) == 1
    assert await repo.knn(engine, "ns", q, 5, {"$or": []}) == []
    assert await repo.knn(engine, "ns", q, 5, {"chunk_index": {"$in": []}}) == []
    assert await repo.knn(engine, "ns", q, 5, {"id": {"$in": []}}) == []
    assert len(await repo.knn(engine, "ns", q, 5, {"chunk_index": {"$nin": []}})) == 1


@pytest.mark.asyncio
async def test_semantic_digest_date_filter_on_default_date_field(engine):
    # temporal_digest semantic mode filters on request.date_metadata_field, "date" by default.
    repo, emb = RedisRepository(), FakeEmbeddings()
    await _seed(repo, engine, "ns", ("t1", {"id": "d1", "date": "2026-07-27 10:00:00,123456"}),
                ("t2", {"id": "d2", "date": "2026-08-02 09:00:00,000000"}))
    flt = {"digest_type": {"$ne": "digest"}, "date": {"$gte": "2026-07-27", "$lte": "2026-07-31T23:59:59"}}
    assert {d.metadata["id"] for d, _ in await repo.knn(engine, "ns", await emb.aembed_query("t1"), 5, flt)} == {"d1"}


@pytest.mark.asyncio
async def test_regex_custom_chunks_get_file_name_and_page():
    from unittest.mock import patch
    from tilellm.models import ItemSingle

    repo = RedisRepository()
    item = ItemSingle(id="doc", namespace="ns", engine=Engine(name="redis"), source="https://x.y/dir/atto.pdf",
                      type="regex_custom", chunk_regex=r"\n\n")
    with patch.object(RedisRepository, "fetch_documents", return_value=[Document(page_content="a", metadata={})]):
        chunks, _ = await repo.prepare_item_chunks(item, None)
    assert chunks[0].metadata["file_name"] == "atto.pdf" and chunks[0].metadata["page"] == 1


@pytest.mark.asyncio
async def test_hybrid_request_logs_the_dense_downgrade(engine):
    repo, emb = RedisRepository(), FakeEmbeddings()
    await _seed(repo, engine, "ns", ("t1", {"id": "d1"}))
    qa = QuestionAnswer(question="t1", namespace="ns", engine=engine, search_type="hybrid")
    from unittest.mock import patch
    import tilellm.store.redis.redis_vector_repository as mod
    with patch.object(mod.logger, "warning") as warn:  # the tilellm logger does not propagate to caplog
        res = await RedisRepository.get_chunks_from_repo.__wrapped__(repo, qa, embedding_obj=emb)
    assert res.chunks == ["t1"] and res.duration > 0
    assert "dense" in warn.call_args[0][0]


def test_clients_of_closed_loops_are_dropped(engine):
    import tilellm.store.redis.redis_vector_repository as mod

    async def touch():
        await RedisRepository().knn(engine, "ns", [1.0] * 8, 1)

    for _ in range(4):
        asyncio.run(touch())
    assert len(mod._clients) <= 1


@pytest.mark.asyncio
async def test_delete_namespace_also_removes_hashes_redis_could_not_index(engine):
    # FT.DROPINDEX DD deletes only indexed documents: a hash that failed indexing would stay forever.
    repo = RedisRepository()
    await _seed(repo, engine, "ns", ("t1", {"id": "d1"}))
    r = redis.Redis(host=_URL.hostname, port=_URL.port or 6379)
    orphan = f"tlvs:{engine.index_name}:ns:orphan"
    r.hset(orphan, mapping={"embedding": b"bad", "text": "x", "metadata": "{}"})
    await repo.delete_namespace(RepositoryNamespace(namespace="ns", engine=engine))
    assert not r.exists(orphan)
