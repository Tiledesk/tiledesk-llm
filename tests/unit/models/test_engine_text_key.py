"""Engine.text_key defaults by Pinecone index type, and an explicit value is kept.

A pod→serverless migration copies metadata unchanged, so the chunk stays under
``content``. Callers must be able to pass ``text_key="content"`` on a serverless
engine; the validator used to overwrite that with ``text``.
"""
from tilellm.models import Engine


class TestEngineTextKey:
    def test_serverless_defaults_to_text(self):
        engine = Engine(name="pinecone", type="serverless", apikey="k", index_name="idx")
        assert engine.text_key == "text"

    def test_pod_defaults_to_content(self):
        engine = Engine(name="pinecone", type="pod", apikey="k", index_name="idx")
        assert engine.text_key == "content"

    def test_explicit_text_key_survives_serverless_default(self):
        engine = Engine(
            name="pinecone",
            type="serverless",
            apikey="k",
            index_name="idx",
            text_key="content",
        )
        assert engine.text_key == "content"

    def test_explicit_text_key_survives_pod_default(self):
        engine = Engine(
            name="pinecone",
            type="pod",
            apikey="k",
            index_name="idx",
            text_key="text",
        )
        assert engine.text_key == "text"

    def test_blank_text_key_falls_back_to_type_default(self):
        engine = Engine(
            name="pinecone",
            type="serverless",
            apikey="k",
            index_name="idx",
            text_key="",
        )
        assert engine.text_key == "text"

    def test_qdrant_keeps_field_default(self):
        engine = Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx")
        assert engine.text_key == "text"


import pytest


@pytest.mark.asyncio
async def test_text_key_is_part_of_the_vector_store_cache_key(monkeypatch):
    # the cached wrapper builds PineconeVectorStore(text_key=...) from the first engine it saw:
    # a migrated index read with "content" must not reuse a wrapper built with "text"
    from tilellm.shared.timed_cache import TimedCache
    from tilellm.store.pinecone import pinecone_repository_base as base

    async def fake_instance(engine, embeddings, emb_dimension):
        return engine.text_key

    monkeypatch.setattr(base, "_create_vector_store_instance", fake_instance)
    await TimedCache.async_clear_cache("vector_store_wrapper")
    create = base.PineconeRepositoryBase.create_index_cache_wrapper
    try:
        default = Engine(name="pinecone", type="serverless", apikey="k", index_name="idx-cache-test")
        migrated = default.model_copy(update={"text_key": "content"})
        assert await create(default, None, 1536) == "text"
        assert await create(migrated, None, 1536) == "content"
    finally:
        await TimedCache.async_clear_cache("vector_store_wrapper")
