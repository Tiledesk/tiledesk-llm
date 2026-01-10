#!/usr/bin/env python3
"""
Test metadata_id propagation in Neo4j.
"""
import asyncio
import sys
sys.path.insert(0, '..')

from tilellm.models.vector_store import Engine
from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository
from tilellm.modules.knowledge_graph.repository.repository import GraphRepository
from tilellm.modules.knowledge_graph.services.services import GraphService, GraphRAGService

async def test_metadata_propagation():
    # Neo4j connection (same as controllers)
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "n3o4j_Mammata"
    
    print("Connecting to Neo4j...")
    graph_repo = GraphRepository(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        max_connection_pool_size=50
    )
    if not graph_repo.verify_connection():
        print("Failed to connect to Neo4j")
        return
    
    graph_service = GraphService(repository=graph_repo)
    
    # Qdrant repository
    print("Initializing Qdrant repository...")
    qdrant_repo = QdrantRepository()
    
    # GraphRAG service with mock LLM that will cause GraphRAG extraction to fail, falling back to simple document nodes
    class MockLLM:
        async def invoke(self, *args, **kwargs):
            raise Exception("Mock LLM failing intentionally")
        async def ainvoke(self, *args, **kwargs):
            raise Exception("Mock LLM failing intentionally")
        def chat(self, *args, **kwargs):
            raise Exception("Mock LLM failing intentionally")
    
    mock_llm = MockLLM()
    graph_rag_service = GraphRAGService(graph_service=graph_service, vector_store_repository=qdrant_repo, llm=mock_llm)
    
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
    limit = 5
    
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