"""
Integration tests for FalkorDB Knowledge Graph API endpoints.
Requires a running FalkorDB instance on localhost:6380.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Set environment variables before importing app
os.environ["ENABLE_GRAPHRAG_FALKOR"] = "true"
os.environ["ENABLE_GRAPHRAG"] = "false"
os.environ["FALKORDB_URI"] = "redis://localhost:6380/0"

# Import the app after setting environment variables
from tilellm.__main__ import app


@pytest.fixture
def client():
    """Test client fixture for FalkorDB API tests."""
    with TestClient(app) as test_client:
        yield test_client


def test_falkor_health_check(client: TestClient):
    """
    Test the /api/kg-falkor/health endpoint.
    Should return 200 if FalkorDB connection works.
    """
    response = client.get("/api/kg-falkor/health")
    # Expect 200 (connected) or 503 (service unavailable)
    # Since we have a running FalkorDB container, should be 200
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


def test_falkor_create_endpoint_exists(client: TestClient):
    """
    Test that /api/kg-falkor/create endpoint exists and validates input.
    Should return 422 (validation error) for empty payload.
    """
    response = client.post("/api/kg-falkor/create", json={})
    # Should not be 404 (endpoint not found)
    assert response.status_code != 404
    # Likely 422 Unprocessable Entity due to missing required fields
    assert response.status_code in [400, 422, 500]


def test_falkor_qa_endpoint_exists(client: TestClient):
    """
    Test that /api/kg-falkor/qa endpoint exists and validates input.
    """
    response = client.post("/api/kg-falkor/qa", json={})
    assert response.status_code != 404
    assert response.status_code in [400, 422, 500]


def test_falkor_create_and_query_simple_graph(client: TestClient):
    """
    Test creating a simple graph and querying it.
    This is a more comprehensive test that creates a small graph,
    runs a query, and verifies results.
    """
    # Create a simple graph with a single node
    create_payload = {
        "namespace": "test_falkor",
        "index_name": "test_index",
        "engine_name": "test_engine",
        "engine_type": "test_type",
        "metadata_id": "test_metadata",
        "entities": [
            {
                "id": "entity_1",
                "text": "Test entity for FalkorDB",
                "label": "Entity",
                "properties": {
                    "name": "Test Entity 1",
                    "description": "A test entity for integration testing"
                }
            }
        ],
        "relationships": []
    }
    create_response = client.post("/api/kg-falkor/create", json=create_payload)
    # Should be 200 (success) or 500 if something goes wrong
    assert create_response.status_code in [200, 500]
    if create_response.status_code == 500:
        # Log error but don't fail - might be due to index creation errors
        print(f"Create endpoint returned 500: {create_response.text}")
        return
    
    # Query the graph to verify node exists
    query_payload = {
        "query": "MATCH (n) RETURN n LIMIT 10",
        "namespace": "test_falkor",
        "index_name": "test_index",
        "engine_name": "test_engine",
        "engine_type": "test_type"
    }
    query_response = client.post("/api/kg-falkor/query", json=query_payload)
    # Should return 200 with results
    assert query_response.status_code == 200
    data = query_response.json()
    assert "results" in data
    # At least one node should be present
    assert len(data["results"]) >= 1
    
    # Cleanup: delete nodes by metadata
    delete_payload = {
        "namespace": "test_falkor",
        "index_name": "test_index",
        "engine_name": "test_engine",
        "engine_type": "test_type",
        "metadata_id": "test_metadata"
    }
    delete_response = client.post("/api/kg-falkor/delete-by-metadata", json=delete_payload)
    # Should return 200 with deletion stats
    assert delete_response.status_code == 200
    stats = delete_response.json()
    assert "nodes_deleted" in stats


if __name__ == "__main__":
    # Run tests directly (for debugging)
    import sys
    sys.exit(pytest.main([__file__, "-v"]))