def get_vector_repo(engine):
    """The one place that maps engine.name (and the Pinecone type) to a repository class."""
    name = engine.name
    if name == "pinecone":
        if engine.type == "pod":
            from tilellm.store.pinecone.pinecone_repository_pod import PineconeRepositoryPod
            return PineconeRepositoryPod()
        if engine.type == "serverless":
            from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
            return PineconeRepositoryServerless()
        raise ValueError(f"Unknown Pinecone repository type: {engine.type}")
    if name == "qdrant":
        from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository
        return QdrantRepository()
    if name == "milvus":
        from tilellm.store.milvus.milvus_repository import MilvusRepository
        return MilvusRepository()
    if name == "redis":
        from tilellm.store.redis.redis_vector_repository import RedisRepository
        return RedisRepository()
    raise ValueError(f"Unknown engine name: {name}")
