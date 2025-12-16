# tests/test_main_api.py

import json
import os
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Dict, Any

# This test file will cover the endpoints defined in 'tilellm/__main__.py'

# Helper function to get the path of a payload file
def get_payload_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'payloads', filename)

# Define a simple Pydantic model for structured output testing
class TestOutputSchema(BaseModel):
    name: str
    age: int
    is_student: bool = False
    courses: List[str] = []

# Helper function to create a structured payload
def _create_structured_payload(base_payload: Dict, output_schema: BaseModel, question_prefix: str = "") -> Dict:
    payload = base_payload.copy()
    payload['structured_output'] = True
    payload['output_schema'] = output_schema.model_json_schema()
    payload['question'] = f"{question_prefix}Return a JSON object with name 'John Doe', age 30, is_student true, and courses ['Math', 'Science']. The model should strictly adhere to the provided JSON schema."
    return payload

def test_get_root(client: TestClient):
    """
    Test the root endpoint (GET /).
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello from Tiledesk LLM python server!!"

def test_post_ask_with_memory(client: TestClient):
    """
    Test the /api/qa endpoint.
    This corresponds to post_ask_with_memory_main.
    """
    payload_path = get_payload_path('post_ask_with_memory.json')
    with open(payload_path, 'r') as f:
        payload = json.load(f)
    
    response = client.post("/api/qa", json=payload)
    
    # Basic check. A 500 error is likely if LLM keys are missing, which is expected.
    # The main goal here is to ensure the endpoint is reachable.
    assert response.status_code != 404

def test_post_ask_to_agent(client: TestClient):
    """
    Test the /api/agent endpoint.
    This corresponds to post_ask_to_agent_main.
    """
    payload_path = get_payload_path('post_ask_to_agent.json')
    with open(payload_path, 'r') as f:
        payload = json.load(f)

    response = client.post("/api/agent", json=payload)
    assert response.status_code != 404

def test_post_ask_to_llm(client: TestClient):
    """
    Test the /api/ask endpoint.
    This corresponds to post_ask_to_llm_main.
    """
    payload_path = get_payload_path('post_ask_to_llm.json')
    with open(payload_path, 'r') as f:
        payload = json.load(f)

    response = client.post("/api/ask", json=payload)
    assert response.status_code != 404

def test_post_ask_with_memory_chain(client: TestClient):
    """
    Test the /api/qachain endpoint.
    This corresponds to post_ask_with_memory_chain_main.
    """
    payload_path = get_payload_path('post_ask_with_memory_chain.json')
    with open(payload_path, 'r') as f:
        payload = json.load(f)

    response = client.post("/api/qachain", json=payload)
    assert response.status_code != 404

# New test for ask_to_llm with structured output
def test_post_ask_to_llm_structured_output(client: TestClient):
    """
    Test the /api/ask endpoint with structured output.
    """
    base_payload_path = get_payload_path('post_ask_to_llm.json')
    with open(base_payload_path, 'r') as f:
        base_payload = json.load(f)

    structured_payload = _create_structured_payload(
        base_payload,
        TestOutputSchema,
        question_prefix="For the following question, "
    )
    
    # Ensure a valid API key is present in the payload for the LLM to function
    # In a real scenario, this would be mocked or provided via env vars
    structured_payload['llm_key'] = os.getenv('OPENAI_API_KEY') # Replace with actual key or mock

    response = client.post("/api/ask", json=structured_payload)
    assert response.status_code == 200
    
    response_json = response.json()
    assert 'answer' in response_json
    assert isinstance(response_json['answer'], dict)
    
    # Assert the structure and some content
    answer = response_json['answer']
    assert 'name' in answer and isinstance(answer['name'], str)
    assert 'age' in answer and isinstance(answer['age'], int)
    assert 'is_student' in answer and isinstance(answer['is_student'], bool)
    assert 'courses' in answer and isinstance(answer['courses'], list)

    assert answer['name'] == 'John Doe'
    assert answer['age'] == 30
    assert answer['is_student'] == True
    assert set(answer['courses']) == set(['Math', 'Science'])


# New test for ask_reason_llm with structured output (using /api/thinking)
def test_post_ask_to_llm_reason_structured_output(client: TestClient):
    """
    Test the /api/thinking endpoint with structured output.
    """
    base_payload_path = get_payload_path('post_ask_to_llm.json') # Use a similar base payload
    with open(base_payload_path, 'r') as f:
        base_payload = json.load(f)

    # The /api/thinking endpoint also uses QuestionToLLM, so base_payload is suitable.
    structured_payload = _create_structured_payload(
        base_payload,
        TestOutputSchema,
        question_prefix="For the following reasoning task, "
    )
    structured_payload['llm_key'] = os.getenv('OPENAI_API_KEY') # Replace with actual key or mock

    response = client.post("/api/thinking", json=structured_payload)
    assert response.status_code == 200

    response_json = response.json()
    assert 'answer' in response_json
    assert isinstance(response_json['answer'], dict)

    answer = response_json['answer']
    assert 'name' in answer and isinstance(answer['name'], str)
    assert 'age' in answer and isinstance(answer['age'], int)
    assert 'is_student' in answer and isinstance(answer['is_student'], bool)
    assert 'courses' in answer and isinstance(answer['courses'], list)

    assert answer['name'] == 'John Doe'
    assert answer['age'] == 30
    assert answer['is_student'] == True
    assert set(answer['courses']) == set(['Math', 'Science'])


# New test for ask_with_memory (search_type='similarity') with structured output
def test_post_ask_with_memory_structured_output(client: TestClient):
    """
    Test the /api/qa endpoint (ask_with_memory) with structured output.
    """
    base_payload_path = get_payload_path('post_ask_with_memory.json')
    with open(base_payload_path, 'r') as f:
        base_payload = json.load(f)

    structured_payload = _create_structured_payload(
        base_payload,
        TestOutputSchema,
        question_prefix="Based on provided context, "
    )
    structured_payload['search_type'] = 'similarity' # Ensure ask_with_memory is called
    structured_payload['gptkey'] = os.getenv('OPENAI_API_KEY') # Replace with actual key or mock

    response = client.post("/api/qa", json=structured_payload)
    assert response.status_code == 200

    response_json = response.json()
    assert 'answer' in response_json
    assert isinstance(response_json['answer'], dict)

    answer = response_json['answer']
    assert 'name' in answer and isinstance(answer['name'], str)
    assert 'age' in answer and isinstance(answer['age'], int)
    assert 'is_student' in answer and isinstance(answer['is_student'], bool)
    assert 'courses' in answer and isinstance(answer['courses'], list)

    assert answer['name'] == 'John Doe'
    assert answer['age'] == 30
    assert answer['is_student'] == True
    assert set(answer['courses']) == set(['Math', 'Science'])


# New test for ask_hybrid_with_memory (search_type='hybrid') with structured output
def test_post_ask_hybrid_with_memory_structured_output(client: TestClient):
    """
    Test the /api/qa endpoint (ask_hybrid_with_memory) with structured output.
    """
    base_payload_path = get_payload_path('post_ask_with_memory.json')
    with open(base_payload_path, 'r') as f:
        base_payload = json.load(f)

    structured_payload = _create_structured_payload(
        base_payload,
        TestOutputSchema,
        question_prefix="Considering relevant documents, "
    )
    structured_payload['search_type'] = 'hybrid' # Ensure ask_hybrid_with_memory is called
    structured_payload['gptkey'] = os.getenv('OPENAI_API_KEY') # Replace with actual key or mock

    response = client.post("/api/qa", json=structured_payload)
    assert response.status_code == 200

    response_json = response.json()
    assert 'answer' in response_json
    assert isinstance(response_json['answer'], dict)

    answer = response_json['answer']
    assert 'name' in answer and isinstance(answer['name'], str)
    assert 'age' in answer and isinstance(answer['age'], int)
    assert 'is_student' in answer and isinstance(answer['is_student'], bool)
    assert 'courses' in answer and isinstance(answer['courses'], list)

    assert answer['name'] == 'John Doe'
    assert answer['age'] == 30
    assert answer['is_student'] == True
    assert set(answer['courses']) == set(['Math', 'Science'])