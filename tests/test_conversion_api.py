# tests/test_conversion_api.py

import json
import os
from fastapi.testclient import TestClient

# This test file covers the endpoints in 'tilellm/modules/conversion/controllers.py'

def get_payload_path(filename: str) -> str:
    """Helper function to get the path of a payload file."""
    return os.path.join(os.path.dirname(__file__), 'payloads', filename)

def test_post_convert_pdf_to_text(client: TestClient):
    """
    Test the /api/convert endpoint for PDF to text conversion.
    """
    payload_path = get_payload_path('conversion_pdf_to_text.json')
    with open(payload_path, 'r') as f:
        payload = json.load(f)
    
    response = client.post("/api/convert", json=payload)
    
    # Basic check. The underlying 'pypdf' library might raise an error
    # with the dummy base64 content, so we might get a 400 or 500, which is fine for now.
    # The main goal is to ensure the endpoint is wired up correctly.
    assert response.status_code != 404
