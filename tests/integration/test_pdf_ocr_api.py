# tests/integration/test_pdf_ocr_api.py

import json
import os
import uuid
import base64
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

# Enable PDF OCR module for tests
os.environ["ENABLE_PDF_OCR"] = "true"

# This test file covers the endpoints in 'tilellm/modules/pdf_ocr/controllers.py'


@pytest.fixture(autouse=True)
def setup_pdf_ocr_mocks(mocker):
    """Setup mocks for PDF OCR services before each test."""
    # Create mock services
    mock_taskiq = MagicMock()
    mock_taskiq.is_available.return_value = True
    mock_taskiq.submit = AsyncMock(return_value={
        "task_id": str(uuid.uuid4()),
        "doc_id": "test-doc-123",
        "message": "queued",
    })

    mock_job = MagicMock()
    mock_job.register_task = MagicMock()
    mock_job.get_status_response.return_value = None  # Default: job not found
    
    mock_redis = MagicMock()
    mock_redis.is_minio_available.return_value = False
    mock_redis.get_minio_error.return_value = "Not configured"
    mock_redis.submit = AsyncMock()
    
    # Mock the getter functions in controllers module
    mocker.patch('tilellm.modules.pdf_ocr.controllers.get_taskiq_service', return_value=mock_taskiq)
    mocker.patch('tilellm.modules.pdf_ocr.controllers.get_job_service', return_value=mock_job)
    mocker.patch('tilellm.modules.pdf_ocr.controllers.get_redis_queue_service', return_value=mock_redis)
    
    yield {
        'taskiq': mock_taskiq,
        'job': mock_job,
        'redis': mock_redis
    }


# --- Health Endpoint Tests ---

def test_get_pdf_health(client: TestClient, setup_pdf_ocr_mocks):
    """
    Test the /api/pdf/health endpoint with mocked services.
    """
    response = client.get("/api/pdf/health")

    # If the endpoint is reached and services can be "retrieved" (mocked), we get 200
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "pipelines" in data


# --- PDF Scrape Endpoint Tests - Docling Pipeline (URL only) ---

def test_post_pdf_scrape_docling_with_url(client: TestClient, setup_pdf_ocr_mocks):
    """
    Test the /api/pdf/scrape endpoint with Docling pipeline (use_docling=true) and URL.
    This is the recommended flow - no MinIO required.
    """
    mocks = setup_pdf_ocr_mocks
    
    # Payload with URL (required for Docling pipeline)
    # Must include all required fields from PDFScrapingRequest/ItemSingle
    payload = {
        "id": "test-doc-123",
        "file_name": "report.pdf",
        "file_content": "https://example.com/reports/test.pdf",
        "namespace": "test",
        "engine": {
            "name": "qdrant",
            "host": "localhost",
            "port": 6333,
            "deployment": "local"
        },
        "webhook_url": "https://example.com/webhook",
        "use_docling": True
    }

    response = client.post("/api/pdf/scrape", json=payload)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data

    mocks['taskiq'].submit.assert_called_once()
    mocks['job'].register_task.assert_called_once()


def test_post_pdf_scrape_docling_rejects_base64(client: TestClient, setup_pdf_ocr_mocks):
    """
    Test that the Docling pipeline rejects Base64 content (requires URL).
    """
    mocks = setup_pdf_ocr_mocks
    
    # Payload with Base64 (should be rejected for Docling pipeline)
    payload = {
        "id": "test-doc-456",
        "file_name": "report.pdf",
        "file_content": base64.b64encode(b"fake pdf content").decode('utf-8'),
        "namespace": "test",
        "engine": {
            "name": "qdrant",
            "host": "localhost",
            "port": 6333,
            "deployment": "local"
        },
        "webhook_url": "https://example.com/webhook",
        "use_docling": True
    }

    response = client.post("/api/pdf/scrape", json=payload)

    # Assertions - should return 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "URL" in data["detail"]
    
    # Taskiq submit should NOT be called
    mocks['taskiq'].submit.assert_not_called()


# --- PDF Scrape Endpoint Tests - Legacy Pipeline (requires MinIO) ---

def test_post_pdf_scrape_legacy_with_minio(client: TestClient, setup_pdf_ocr_mocks):
    """
    Test the /api/pdf/scrape endpoint with legacy pipeline (use_docling=false).
    Requires MinIO to be available.
    """
    mocks = setup_pdf_ocr_mocks
    mocks['redis'].is_minio_available.return_value = True
    
    # Payload with Base64 (allowed for legacy pipeline)
    payload = {
        "id": "test-doc-789",
        "file_name": "report.pdf",
        "file_content": base64.b64encode(b"fake pdf content").decode('utf-8'),
        "namespace": "test",
        "engine": {
            "name": "qdrant",
            "host": "localhost",
            "port": 6333,
            "deployment": "local"
        },
        "webhook_url": "https://example.com/webhook",
        "use_docling": False
    }

    response = client.post("/api/pdf/scrape", json=payload)

    # Assertions
    assert response.status_code == 200
    assert "job_id" in response.json()

    mocks['redis'].submit.assert_called_once()


def test_post_pdf_scrape_legacy_without_minio(client: TestClient, setup_pdf_ocr_mocks):
    """
    Test that legacy pipeline fails gracefully when MinIO is not available.
    """
    mocks = setup_pdf_ocr_mocks
    mocks['redis'].is_minio_available.return_value = False
    mocks['redis'].get_minio_error.return_value = "Connection refused"
    
    # Payload
    payload = {
        "id": "test-doc-000",
        "file_name": "report.pdf",
        "file_content": base64.b64encode(b"fake pdf content").decode('utf-8'),
        "namespace": "test",
        "engine": {
            "name": "qdrant",
            "host": "localhost",
            "port": 6333,
            "deployment": "local"
        },
        "webhook_url": "https://example.com/webhook",
        "use_docling": False
    }

    response = client.post("/api/pdf/scrape", json=payload)

    # Assertions - should return 503 Service Unavailable
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "MinIO" in data["detail"]
    
    # Submit should NOT be called
    mocks['redis'].submit.assert_not_called()


# --- PDF Status Endpoint Tests ---

def test_get_pdf_status_not_found(client: TestClient, setup_pdf_ocr_mocks):
    """
    Test the /api/pdf/status/{job_id} endpoint for a job that doesn't exist.
    """
    mocks = setup_pdf_ocr_mocks
    
    # Mock result_backend to return None (job not found)
    async def mock_get_status(job_id):
        return None
    
    mocks['job'].get_status_response = mock_get_status
    
    # Make request
    job_id = "non-existent-job-id"
    response = client.get(f"/api/pdf/status/{job_id}")

    # Assertions
    assert response.status_code == 404


def test_get_pdf_status_found_processing(client: TestClient, setup_pdf_ocr_mocks):
    """
    Test the /api/pdf/status/{job_id} endpoint for a job that is processing.
    """
    from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingStatusResponse
    
    mocks = setup_pdf_ocr_mocks
    
    # Mock result_backend to return not ready (still processing)
    async def mock_get_status(job_id):
        return PDFScrapingStatusResponse(
            job_id=job_id,
            status="processing",
            progress=50,
            message="Job is being processed by a worker.",
            result=None,
            error_message=None
        )
    
    mocks['job'].get_status_response = mock_get_status
    
    # Make request
    job_id = "test-job-processing"
    response = client.get(f"/api/pdf/status/{job_id}")

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "processing"


def test_get_pdf_status_found_completed(client: TestClient, setup_pdf_ocr_mocks):
    """
    Test the /api/pdf/status/{job_id} endpoint for a completed job.
    """
    from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingStatusResponse, PDFScrapingResponse, PDFPage
    
    mocks = setup_pdf_ocr_mocks
    
    # Mock result_backend to return completed result with proper structure
    async def mock_get_status(job_id):
        return PDFScrapingStatusResponse(
            job_id=job_id,
            status="completed",
            progress=100,
            message="Processing completed successfully.",
            result=PDFScrapingResponse(
                file_name="test.pdf",
                total_pages=1,
                pages=[],
                markdown_content="# Test"
            ),
            error_message=None
        )
    
    mocks['job'].get_status_response = mock_get_status
    
    # Make request
    job_id = "test-job-completed"
    response = client.get(f"/api/pdf/status/{job_id}")

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert data["result"] is not None
