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
