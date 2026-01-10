#!/usr/bin/env python3
"""
Test Neo4j connection and metadata propagation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tilellm.modules.knowledge_graph.repository.repository import GraphRepository

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "n3o4j_Mammata"

def main():
    print("Connecting to Neo4j...")
    repo = GraphRepository(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        max_connection_pool_size=50
    )
    
    if not repo.verify_connection():
        print("Connection failed")
        return
    
    print("Connection successful")
    
    # Get database info
    info = repo.get_database_info()
    print(f"Database info: {info}")
    
    # Query all nodes with metadata_id property
    with repo._get_session() as session:
        query = """
        MATCH (n)
        WHERE n.metadata_id IS NOT NULL
        RETURN labels(n) as labels, n.metadata_id as metadata_id, n.namespace as namespace, n.engine_name as engine_name, n.index_name as index_name, n.engine_type as engine_type
        LIMIT 20
        """
        result = session.run(query)
        print("\nNodes with metadata_id:")
        count = 0
        for record in result:
            print(f"  - {record['labels']}: metadata_id={record['metadata_id']}, namespace={record['namespace']}, engine={record['engine_name']}")
            count += 1
        print(f"Total: {count}")
        
        # Query all nodes to see total
        query2 = """
        MATCH (n)
        RETURN count(n) as total
        """
        result2 = session.run(query2)
        total = result2.single()["total"]
        print(f"\nTotal nodes in graph: {total}")
        
        # Query for Document nodes
        query3 = """
        MATCH (n:Document)
        RETURN n.metadata_id as metadata_id, n.chunk_id as chunk_id, n.namespace as namespace
        LIMIT 10
        """
        result3 = session.run(query3)
        print("\nDocument nodes:")
        for record in result3:
            print(f"  - metadata_id={record['metadata_id']}, chunk_id={record['chunk_id']}, namespace={record['namespace']}")
    
    repo.close()

if __name__ == "__main__":
    main()