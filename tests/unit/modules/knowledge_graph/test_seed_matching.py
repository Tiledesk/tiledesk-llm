#!/usr/bin/env python3
"""Test seed node matching with fixed query."""

import sys
import logging
sys.path.insert(0, '..')

from neo4j import GraphDatabase
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "n3o4j_Mammata")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

print("=== Testing seed node matching ===")

# Test with chunk IDs that should match
test_ids = [
    "13d6c3a8-d2e4-4c5c-8dc2-af6e8551f536",  # Should match (seen in earlier debug)
    "0f6062a9-cfb4-48b7-be96-e61ab9676b17",  # Should NOT match (from tilellm-hybrid)
    "9acf2283-5875-400b-ba3a-cca9b42186cf",  # Should match
    "non-existent-id"  # Should NOT match
]

for source_id in test_ids:
    print(f"\n--- Testing source_id: {source_id} ---")
    
    # Use the exact query from our fixed repository method
    query = """
    MATCH (n)
    WHERE n.source_ids IS NOT NULL AND $source_id IN n.source_ids
    RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
    LIMIT 10
    """
    
    with driver.session() as session:
        try:
            result = session.run(query, source_id=source_id)
            nodes = []
            for record in result:
                node_id = str(record["id"])
                label = record["labels"][0] if record["labels"] else "Unknown"
                props = dict(record["properties"])
                nodes.append({
                    "id": node_id,
                    "label": label,
                    "source_ids": props.get('source_ids', 'N/A')
                })
            
            print(f"Found {len(nodes)} nodes")
            for node in nodes:
                print(f"  Node: id={node['id']}, label={node['label']}, source_ids={node['source_ids']}")
                
        except Exception as e:
            print(f"Error: {e}")

driver.close()
print("\n=== Test completed ===")