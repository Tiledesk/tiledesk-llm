"""Un solo punto sceglie la classe di repository dal nome dell'engine (inject_repo sync/async e lgraph)."""
import pytest

from tilellm.models.vector_store import Engine
from tilellm.store import get_vector_repo


@pytest.mark.parametrize("engine, cls_name", [
    (Engine(name="pinecone", type="pod"), "PineconeRepositoryPod"),
    (Engine(name="pinecone", type="serverless"), "PineconeRepositoryServerless"),
    (Engine(name="qdrant", deployment="local"), "QdrantRepository"),
    (Engine(name="milvus"), "MilvusRepository"),
    (Engine(name="redis"), "RedisRepository"),
])
def test_engine_name_selects_repository(engine, cls_name):
    assert type(get_vector_repo(engine)).__name__ == cls_name


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError, match="Unknown engine"):
        get_vector_repo(Engine.model_construct(name="nope"))


def test_redis_engine_defaults_to_redis_port_and_db0():
    e = Engine(name="redis")
    assert (e.port, e.database) == (6379, "0")
    assert Engine(name="redis", port=6390).port == 6390


def test_redis_engine_rejects_db_other_than_0():
    # Redis Search crea indici solo sul DB 0 ("Cannot create index on db != 0"):
    # meglio un 422 subito che un errore a meta' ingestion.
    with pytest.raises(ValueError, match="database"):
        Engine(name="redis", database="3")


@pytest.mark.asyncio
async def test_repository_cache_key_includes_port_and_database():
    from tilellm.shared.utility import _build_repository_cache_key

    class Q:
        def __init__(self, engine):
            self.engine = engine

    a = await _build_repository_cache_key(Q(Engine(name="redis", port=6379)))
    b = await _build_repository_cache_key(Q(Engine(name="redis", port=6390)))
    assert a != b
