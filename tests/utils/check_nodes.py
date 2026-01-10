#!/usr/bin/env python3
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "n3o4j_Mammata"))

with driver.session() as session:
    # Get a few nodes with all properties
    result = session.run("""
        MATCH (n)
        RETURN labels(n) as labels, properties(n) as props
        LIMIT 10
    """)
    for i, record in enumerate(result):
        print(f"\nNode {i}: {record['labels']}")
        for k, v in record['props'].items():
            print(f"  {k}: {v}")
    
    # Check if any nodes have namespace, engine_name, index_name
    result = session.run("""
        MATCH (n)
        WHERE n.namespace IS NOT NULL OR n.engine_name IS NOT NULL OR n.index_name IS NOT NULL
        RETURN labels(n) as labels, n.namespace as namespace, n.engine_name as engine_name, n.index_name as index_name, n.engine_type as engine_type
        LIMIT 10
    """)
    print("\n=== Nodes with partition properties ===")
    for record in result:
        print(f"{record['labels']}: namespace={record['namespace']}, engine={record['engine_name']}, index={record['index_name']}, engine_type={record['engine_type']}")
    
    # Check for Document nodes
    result = session.run("""
        MATCH (n:Document)
        RETURN properties(n) as props
        LIMIT 5
    """)
    print("\n=== Document nodes ===")
    for record in result:
        print(record['props'])
    
driver.close()