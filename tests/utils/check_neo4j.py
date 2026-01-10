#!/usr/bin/env python3
"""
Check Neo4j nodes and metadata.
"""
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "n3o4j_Mammata"

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Check connection
        result = session.run("RETURN 1 as test")
        record = result.single()
        if record and record["test"] == 1:
            print("✅ Connected to Neo4j")
        else:
            print("❌ Connection failed")
            return
        
        # Get total nodes
        result = session.run("MATCH (n) RETURN count(n) as total")
        total = result.single()["total"]
        print(f"Total nodes: {total}")
        
        # Get nodes with metadata_id
        result = session.run("""
            MATCH (n)
            WHERE n.metadata_id IS NOT NULL
            RETURN labels(n) as labels, n.metadata_id as metadata_id, n.namespace as namespace, 
                   n.engine_name as engine_name, n.index_name as index_name, n.engine_type as engine_type,
                   n.chunk_id as chunk_id
            LIMIT 30
        """)
        print("\nNodes with metadata_id:")
        count = 0
        for record in result:
            labels = record["labels"]
            metadata_id = record["metadata_id"]
            namespace = record["namespace"]
            engine_name = record["engine_name"]
            chunk_id = record["chunk_id"]
            print(f"  - {labels}: metadata_id={metadata_id}, namespace={namespace}, engine={engine_name}, chunk_id={chunk_id}")
            count += 1
        print(f"Total with metadata_id: {count}")
        
        # Count nodes per label
        result = session.run("""
            MATCH (n)
            RETURN labels(n) as labels, count(*) as count
            ORDER BY count DESC
        """)
        print("\nLabel distribution:")
        for record in result:
            print(f"  {record['labels']}: {record['count']}")
        
        # Check for duplicate metadata_id per namespace
        result = session.run("""
            MATCH (n)
            WHERE n.metadata_id IS NOT NULL AND n.namespace IS NOT NULL
            RETURN n.namespace as namespace, n.metadata_id as metadata_id, count(*) as count
            ORDER BY count DESC
            LIMIT 20
        """)
        print("\nDuplicate metadata_id per namespace (top 20):")
        for record in result:
            if record["count"] > 1:
                print(f"  namespace={record['namespace']}, metadata_id={record['metadata_id']}, count={record['count']}")
        
        # Check for nodes with source_ids property (GraphRAG entities)
        result = session.run("""
            MATCH (n)
            WHERE n.source_ids IS NOT NULL
            RETURN labels(n) as labels, size(n.source_ids) as source_count, n.metadata_id as metadata_id
            LIMIT 10
        """)
        print("\nNodes with source_ids (GraphRAG entities):")
        for record in result:
            print(f"  - {record['labels']}: source_count={record['source_count']}, metadata_id={record['metadata_id']}")
    
    driver.close()

if __name__ == "__main__":
    main()