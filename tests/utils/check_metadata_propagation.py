#!/usr/bin/env python3
"""
Test metadata_id propagation in Neo4j - force simple document nodes.
"""
import asyncio

from tilellm.models.vector_store import Engine
from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository
from tilellm.modules.knowledge_graph.repository.repository import GraphRepository
from tilellm.modules.knowledge_graph.services.services import GraphService, GraphRAGService

# Patch GRAPHRAG_AVAILABLE to False to force simple document nodes
import tilellm.modules.knowledge_graph.services.services as services_module
services_module.GRAPHRAG_AVAILABLE = False

async def test_metadata_propagation():
    print("Connecting to Neo4j...")
    graph_repo = GraphRepository()
    if not graph_repo.verify_connection():
        print("Failed to connect to Neo4j")
        return
    
    graph_service = GraphService(repository=graph_repo)
    
    # Qdrant repository
    print("Initializing Qdrant repository...")
    qdrant_repo = QdrantRepository()
    
    # GraphRAG service with no LLM (will fall back to simple document nodes)
    graph_rag_service = GraphRAGService(graph_service=graph_service, vector_store_repository=qdrant_repo)
    
    # Engine configuration matching existing data
    engine = Engine(
        name="qdrant",
        type="serverless",
        deployment="local",
        host="localhost",
        port=6333,
        index_name="tilellm-hybrid"
    )
    
    namespace = "test_ns-mock"
    limit = 3
    
    print(f"Importing from namespace {namespace} with overwrite=True...")
    result = await graph_rag_service.import_from_vector_store(
        namespace=namespace,
        engine=engine,
        limit=limit,
        overwrite=True,
        llm=None  # No LLM -> simple document nodes
    )
    print(f"Import result: {result}")
    
    # Query nodes to check metadata_id
    print("\nQuerying nodes with metadata_id...")
    with graph_repo._get_session() as session:
        # Find all nodes in this namespace
        query = """
        MATCH (n)
        WHERE n.namespace = $namespace AND n.engine_name = $engine_name
        RETURN labels(n) as labels, n.metadata_id as metadata_id, n.chunk_id as chunk_id, n.text as text
        LIMIT 10
        """
        result = session.run(query, namespace=namespace, engine_name=engine.name)
        count = 0
        for record in result:
            print(f"Node: {record['labels']}, metadata_id={record['metadata_id']}, chunk_id={record['chunk_id']}")
            count += 1
        print(f"Total nodes found: {count}")
        
        # Check if any nodes missing metadata_id
        query2 = """
        MATCH (n)
        WHERE n.namespace = $namespace AND n.engine_name = $engine_name AND n.metadata_id IS NULL
        RETURN count(n) as missing
        """
        result2 = session.run(query2, namespace=namespace, engine_name=engine.name)
        missing = result2.single()["missing"]
        print(f"Nodes missing metadata_id: {missing}")
    
    graph_repo.close()
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(test_metadata_propagation())