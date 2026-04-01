import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from tilellm.__main__ import app
from tilellm.models.llm import ItemSingle

client = TestClient(app)

@pytest.fixture
def mock_engine():
    return {
        "index_name": "test-index",
        "apikey": "test-key",
        "host": "http://localhost",
        "port": 6333,
        "deployment": "local",
        "vector_store": "qdrant"
    }

@patch("tilellm.modules.ingestion.controllers.add_item")
@patch("tilellm.modules.ingestion.controllers.scrape_pdf")
def test_ingestion_router_to_standard(mock_pdf, mock_standard, mock_engine):
    """Test that ingestion routes to standard pipeline for non-PDF or PDF without OCR."""
    mock_standard.return_value = {"status": "success", "pipeline": "standard"}
    
    payload = {
        "id": "doc123",
        "source": "http://example.com",
        "type": "url",
        "engine": mock_engine,
        "use_ocr": False
    }
    
    response = client.post("/api/ingestion", json=payload)
    assert response.status_code == 200
    assert response.json()["pipeline"] == "standard"
    mock_standard.assert_called_once()
    mock_pdf.assert_not_called()

@patch("tilellm.modules.ingestion.controllers.add_item")
@patch("tilellm.modules.ingestion.controllers.scrape_pdf")
def test_ingestion_router_to_pdf_ocr(mock_pdf, mock_standard, mock_engine):
    """Test that ingestion routes to PDF OCR pipeline for PDF with OCR enabled."""
    mock_pdf.return_value = {"status": "success", "pipeline": "pdf_ocr"}
    
    payload = {
        "id": "doc123",
        "source": "http://example.com/doc.pdf",
        "type": "pdf",
        "engine": mock_engine,
        "use_ocr": True
    }
    
    response = client.post("/api/ingestion", json=payload)
    assert response.status_code == 200
    assert response.json()["pipeline"] == "pdf_ocr"
    mock_pdf.assert_called_once()
    mock_standard.assert_not_called()
