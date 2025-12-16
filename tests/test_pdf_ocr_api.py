# tests/test_pdf_ocr_api.py

import json
import os
import uuid
from fastapi.testclient import TestClient

# This test file covers the endpoints in 'tilellm/modules/pdf_ocr/controllers.py'

def get_payload_path(filename: str) -> str:
    """Helper function to get the path of a payload file."""
    return os.path.join(os.path.dirname(__file__), 'payloads', filename)

def test_get_pdf_health(client: TestClient, mocker):
    """
    Test the /api/pdf/health endpoint with mocked services.
    """
    # Mock the service getters to prevent real instantiation
    mock_redis_q = mocker.patch('tilellm.modules.pdf_ocr.controllers.get_redis_queue_service')
    mock_job_s = mocker.patch('tilellm.modules.pdf_ocr.controllers.get_job_service')

    response = client.get("/api/pdf/health")

    # If the endpoint is reached and services can be "retrieved" (mocked), we get 200
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    mock_redis_q.assert_called_once()
    mock_job_s.assert_called_once()

def test_post_pdf_scrape(client: TestClient, mocker):
    """
    Test the /api/pdf/scrape endpoint with mocked services.
    """
    # 1. Mock the services that the controller depends on
    mock_job_service = mocker.MagicMock()
    mock_job_service.create_job.return_value = str(uuid.uuid4())
    mocker.patch('tilellm.modules.pdf_ocr.controllers.get_job_service', return_value=mock_job_service)

    mock_queue_service = mocker.MagicMock()
    mocker.patch('tilellm.modules.pdf_ocr.controllers.get_redis_queue_service', return_value=mock_queue_service)

    # 2. Prepare payload and make request
    payload_path = get_payload_path('pdf_scrape.json')
    with open(payload_path, 'r') as f:
        payload = json.load(f)
    
    response = client.post("/api/pdf/scrape", json=payload)
    
    # 3. Assertions
    assert response.status_code == 200
    assert "job_id" in response.json()
    
    # Verify that the service methods were called correctly
    mock_job_service.create_job.assert_called_once_with(file_name=payload['file_name'])
    mock_queue_service.submit.assert_called_once()

def test_get_pdf_status_not_found(client: TestClient, mocker):
    """
    Test the /api/pdf/status/{job_id} endpoint for a job that doesn't exist.
    """
    # 1. Mock the job service to simulate a job not being found
    mock_job_service = mocker.MagicMock()
    mock_job_service.get_status_response.return_value = None
    mocker.patch('tilellm.modules.pdf_ocr.controllers.get_job_service', return_value=mock_job_service)

    # 2. Make request
    job_id = "non-existent-job-id"
    response = client.get(f"/api/pdf/status/{job_id}")
    
    # 3. Assertions
    if response.status_code != 404:
        print("Unexpected response body:", response.json())
        
    assert response.status_code == 404
    mock_job_service.get_status_response.assert_called_once_with(job_id)
