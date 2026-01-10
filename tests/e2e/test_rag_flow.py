#!/usr/bin/env python3
"""
End-to-end tests for RAG flow.
These tests simulate complete user interactions with mocked external services.
"""
import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient


class TestRAGFlow:
    """Test complete RAG flow with mocked dependencies."""
    
    def test_complete_rag_conversation(self, client: TestClient):
        """
        Test a complete RAG conversation flow:
        1. Initial question with hybrid search
        2. Follow-up question with chat history
        3. Structured output request
        """
        # Mock payloads directory path
        import os
        payloads_dir = os.path.join(os.path.dirname(__file__), '..', 'payloads')
        
        # Step 1: Initial question with hybrid search
        with open(os.path.join(payloads_dir, 'post_ask_with_memory.json'), 'r') as f:
            payload1 = json.load(f)
        
        # Modify payload for hybrid search
        payload1['search_type'] = 'hybrid'
        payload1['question'] = "What is artificial intelligence?"
        
        # Mock the hybrid search response
        mock_response1 = {
            "answer": "Artificial intelligence is the simulation of human intelligence processes by machines.",
            "chat_history_dict": {
                "0": {
                    "question": "What is artificial intelligence?",
                    "answer": "Artificial intelligence is the simulation of human intelligence processes by machines."
                }
            },
            "prompt_token_info": {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70
            }
        }
        
        # Since we're using a real test client with mocked dependencies in conftest.py,
        # we need to mock the specific controller functions
        with patch('tilellm.controller.controller.ask_hybrid_with_memory', 
                  AsyncMock(return_value=Mock(**mock_response1))):
            response1 = client.post("/api/qa", json=payload1)
            assert response1.status_code == 200
            
            response_data1 = response1.json()
            assert "answer" in response_data1
            assert "chat_history_dict" in response_data1
        
        # Step 2: Follow-up question using history from step 1
        payload2 = payload1.copy()
        payload2['question'] = "What are its main applications?"
        payload2['chat_history_dict'] = response_data1['chat_history_dict']
        
        mock_response2 = {
            "answer": "Main applications include natural language processing, computer vision, and robotics.",
            "chat_history_dict": {
                "0": payload2['chat_history_dict']["0"],
                "1": {
                    "question": "What are its main applications?",
                    "answer": "Main applications include natural language processing, computer vision, and robotics."
                }
            },
            "prompt_token_info": {
                "input_tokens": 60,
                "output_tokens": 25,
                "total_tokens": 85
            }
        }
        
        with patch('tilellm.controller.controller.ask_hybrid_with_memory',
                  AsyncMock(return_value=Mock(**mock_response2))):
            response2 = client.post("/api/qa", json=payload2)
            assert response2.status_code == 200
            
            response_data2 = response2.json()
            assert "applications" in response_data2['answer'].lower()
            assert len(response_data2['chat_history_dict']) == 2
    
    def test_conversion_and_rag_integration(self, client: TestClient):
        """
        Test integration between conversion module and RAG:
        1. Convert PDF to text
        2. Use converted text in RAG search
        """
        # Step 1: Convert PDF to text
        conversion_payload = {
            "file_name": "sample.pdf",
            "file_content": "JVBERi0xLjUK...",  # Minimal PDF base64
            "conversion_type": "pdf_to_text"
        }
        
        mock_conversion_response = [
            {
                "FileName": "sample.txt",
                "FileExt": "txt",
                "FileSize": 1000,
                "File": "base64_encoded_text",
                "FileContent": "This is extracted text from PDF about machine learning."
            }
        ]
        
        with patch('tilellm.modules.conversion.controllers.process_pdf_to_text',
                  return_value=mock_conversion_response):
            response1 = client.post("/api/convert", json=conversion_payload)
            assert response1.status_code == 200
            
            conversion_result = response1.json()
            assert len(conversion_result) == 1
            assert conversion_result[0]['FileExt'] == 'txt'
            
            extracted_text = conversion_result[0]['FileContent']
            assert "machine learning" in extracted_text.lower()
        
        # Step 2: Use extracted text in RAG question
        rag_payload = {
            "question": f"Based on this document: {extracted_text}. What is machine learning?",
            "llm": "gpt-4",
            "llm_key": "test-key",
            "stream": False,
            "search_type": "hybrid"
        }
        
        mock_rag_response = {
            "answer": "Machine learning is a subset of AI that enables systems to learn from data.",
            "chat_history_dict": {},
            "prompt_token_info": {
                "input_tokens": 40,
                "output_tokens": 15,
                "total_tokens": 55
            }
        }
        
        with patch('tilellm.controller.controller.ask_hybrid_with_memory',
                  AsyncMock(return_value=Mock(**mock_rag_response))):
            response2 = client.post("/api/qa", json=rag_payload)
            assert response2.status_code == 200
            
            rag_result = response2.json()
            assert "machine learning" in rag_result['answer'].lower()
    
    def test_structured_output_flow(self, client: TestClient):
        """
        Test flow with structured output requests.
        """
        from pydantic import BaseModel
        
        class PersonInfo(BaseModel):
            name: str
            age: int
            occupation: str
        
        # Create structured output request
        payload = {
            "question": "Extract person information: John Doe, 30 years old, software engineer",
            "llm": "gpt-4",
            "llm_key": "test-key",
            "stream": False,
            "structured_output": True,
            "output_schema": PersonInfo.model_json_schema()
        }
        
        mock_response = {
            "answer": {"name": "John Doe", "age": 30, "occupation": "software engineer"},
            "chat_history_dict": {},
            "prompt_token_info": {
                "input_tokens": 35,
                "output_tokens": 18,
                "total_tokens": 53
            }
        }
        
        with patch('tilellm.controller.controller.ask_to_llm',
                  AsyncMock(return_value=Mock(**mock_response))):
            response = client.post("/api/ask", json=payload)
            assert response.status_code == 200
            
            result = response.json()
            assert result['answer']['name'] == "John Doe"
            assert result['answer']['age'] == 30
            assert result['answer']['occupation'] == "software engineer"


class TestErrorFlows:
    """Test error scenarios in complete flows."""
    
    def test_rag_with_missing_dependencies(self, client: TestClient):
        """Test RAG flow when external dependencies are missing."""
        payload = {
            "question": "Test question",
            "llm": "gpt-4",
            "llm_key": None,  # Missing API key
            "search_type": "hybrid"
        }
        
        # Mock the controller to simulate missing dependency error
        with patch('tilellm.controller.controller.ask_hybrid_with_memory',
                  AsyncMock(side_effect=Exception("Missing API key"))):
            response = client.post("/api/qa", json=payload)
            # Should return error response
            assert response.status_code in [400, 500]
    
    def test_conversion_invalid_file(self, client: TestClient):
        """Test conversion with invalid file content."""
        payload = {
            "file_name": "invalid.pdf",
            "file_content": "not-a-valid-base64",
            "conversion_type": "pdf_to_text"
        }
        
        with patch('tilellm.modules.conversion.controllers.process_pdf_to_text',
                  side_effect=Exception("Invalid PDF content")):
            response = client.post("/api/convert", json=payload)
            assert response.status_code == 400