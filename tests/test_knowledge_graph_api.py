# tests/test_knowledge_graph_api.py

import json
import os
from fastapi.testclient import TestClient

# This test file covers the endpoints in 'tilellm/modules/knowledge_graph/controllers.py'

def test_get_health_check(client: TestClient):
    """
    Test the /knowledge-graph/health endpoint.
    
    This test is expected to fail with a 503 Service Unavailable if a Neo4j 
    instance isn't running, because the service will fail its connection check.
    This is the expected behavior for a health check endpoint.
    """
    response = client.get("/knowledge-graph/health")
    
    # The endpoint should either return 200 (if connected) or 503 (if not).
    # It should not return 404, which would indicate the route is not registered.
    assert response.status_code in [200, 503]

# More tests for other KG endpoints can be added here later.
# For example, creating a node:
#
# def test_create_node(client: TestClient):
#     """
#     Test POST /knowledge-graph/nodes
#     """
#     payload = {
#         "label": "Test",
#         "properties": {"name": "My Test Node"}
#     }
#     response = client.post("/knowledge-graph/nodes", json=payload)
#     assert response.status_code != 404
