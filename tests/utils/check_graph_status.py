#!/usr/bin/env python3
"""Test to check graph status and clustering."""

import sys
sys.path.insert(0, '..')

from tilellm.modules.knowledge_graph.repository.repository import GraphRepository

# Initialize repository (reads from config)
repo = GraphRepository()

print("=== Graph Database Status ===")

# 1. Check connection
if repo.verify_connection():
    print("✅ Connected to Neo4j")
else:
    print("❌ Connection failed")
    sys.exit(1)

# 2. Get basic stats
info = repo.get_database_info()
print(f"Nodes: {info['node_count']}")
print(f"Relationships: {info['relationship_count']}")

# 3. Check for CommunityReport nodes
community_reports = repo.search_community_reports("", limit=10)
print(f"\nCommunityReport nodes: {len(community_reports)}")
for i, report in enumerate(community_reports):
    print(f"  {i+1}. {report.properties.get('title', 'No title')}")

# 4. Check for any nodes with source_ids or chunk_id
print("\n=== Checking for vector store linked nodes ===")
# Simple query to find nodes with source_ids or chunk_id
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "n3o4j_Mammata"))

with driver.session() as session:
    # Find nodes with source_ids property
    result = session.run("""
        MATCH (n) 
        WHERE n.source_ids IS NOT NULL OR n.chunk_id IS NOT NULL
        RETURN labels(n) as labels, count(*) as count
    """)
    print("Nodes linked to vector store chunks:")
    for record in result:
        print(f"  Label {record['labels']}: {record['count']} nodes")

# 5. Check total node labels distribution
print("\n=== Node labels distribution ===")
with driver.session() as session:
    result = session.run("""
        MATCH (n)
        RETURN labels(n) as labels, count(*) as count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"  {record['labels']}: {record['count']}")

driver.close()
print("\n=== Test completed ===")