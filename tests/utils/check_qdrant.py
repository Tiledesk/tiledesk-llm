#!/usr/bin/env python3
import sys
sys.path.insert(0, '..')

from tilellm.models.vector_store import Engine
from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository

async def main():
    engine = Engine(
        name="qdrant",
        type="serverless",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="tilellm-hybrid"
    )
    repo = QdrantRepository()
    
    # List namespaces
    from tilellm.models.schemas import RepositoryNamespaceResult
    result = await repo.list_namespaces(engine)
    print(f"Namespaces: {result}")
    
    # Check namespace test_ns-mock
    namespace = "test_ns-mock"
    from tilellm.models.schemas import RepositoryItems
    items = await repo.get_all_obj_namespace(engine, namespace, with_text=True)
    print(f"Total chunks in {namespace}: {len(items.matches)}")
    for i, chunk in enumerate(items.matches[:5]):
        print(f"Chunk {i}: id={chunk.id}, metadata_id={chunk.metadata_id}, metadata_source={chunk.metadata_source}, text={chunk.text[:50] if chunk.text else ''}")
        # Also print metadata dict
        # chunk doesn't have metadata dict, but we can infer
    
    # Also get ids for a specific metadata_id
    if items.matches:
        sample_metadata_id = items.matches[0].metadata_id
        print(f"\nFetching chunks with metadata_id={sample_metadata_id}")
        items2 = await repo.get_ids_namespace(engine, sample_metadata_id, namespace)
        print(f"Found {len(items2.matches)} chunks with same metadata_id")
        for chunk in items2.matches[:3]:
            print(f"  chunk id={chunk.id}, metadata_id={chunk.metadata_id}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())