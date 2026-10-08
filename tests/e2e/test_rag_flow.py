#!/usr/bin/env python3
"""
End-to-end tests for RAG flow.
These tests simulate complete user interactions with mocked external services.

The controller functions are patched where /api/qa and /api/ask look them up
(tilellm.__main__ imports them by name: patching tilellm.controller.controller
had no effect), the payloads are valid QuestionAnswer / QuestionToLLM bodies,
and the mocked controllers return the real response models.
"""
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from tilellm.models.schemas.general_schemas import SimpleAnswer
from tilellm.models.schemas.retrieval_schemas import RetrievalResult


def _qa_payload(question, **extra):
    payload = {
        "question": question,
        "namespace": "e2e-ns",
        "engine": {"name": "qdrant"},
        "gptkey": "sk-test",
        "search_type": "hybrid",
        "use_cache": False,   # no Redis lookups in this flow
    }
    payload.update(extra)
    return payload


def _answer(answer, history):
    return RetrievalResult(answer=answer, success=True, namespace="e2e-ns", chat_history_dict=history)


class TestRAGFlow:
    """Test complete RAG flow with mocked dependencies."""

    def test_complete_rag_conversation(self, client: TestClient):
        """Two turns of hybrid RAG: the second sends the history of the first."""
        first_q, first_a = "What is artificial intelligence?", "Artificial intelligence is the simulation of human intelligence."
        history1 = {"0": {"question": first_q, "answer": first_a}}

        with patch("tilellm.__main__.ask_hybrid_with_memory",
                   AsyncMock(return_value=_answer(first_a, history1))) as hybrid:
            response1 = client.post("/api/qa", json=_qa_payload(first_q))

        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["answer"] == first_a and data1["chat_history_dict"]["0"]["question"] == first_q
        assert hybrid.await_args.args[0].namespace == "e2e-ns"

        second_q, second_a = "What are its main applications?", "Natural language processing and robotics."
        history2 = {**data1["chat_history_dict"], "1": {"question": second_q, "answer": second_a}}
        with patch("tilellm.__main__.ask_hybrid_with_memory",
                   AsyncMock(return_value=_answer(second_a, history2))) as hybrid:
            response2 = client.post(
                "/api/qa", json=_qa_payload(second_q, chat_history_dict=data1["chat_history_dict"]))

        assert response2.status_code == 200
        assert len(response2.json()["chat_history_dict"]) == 2
        # the history of turn 1 reached the controller on turn 2
        assert "0" in hybrid.await_args.args[0].chat_history_dict

    def test_conversion_and_rag_integration(self, client: TestClient):
        """Convert a PDF to text, then ask a RAG question about it."""
        conversion_payload = {
            "file_name": "sample.pdf",
            "file_content": "JVBERi0xLjUK...",  # Minimal PDF base64
            "conversion_type": "pdf_to_text",
        }
        mock_conversion_response = [{
            "FileName": "sample.txt", "FileExt": "txt", "FileSize": 1000, "File": "base64_encoded_text",
            "FileContent": "This is extracted text from PDF about machine learning.",
        }]

        with patch("tilellm.modules.conversion.controllers.process_pdf_to_text",
                   return_value=mock_conversion_response):
            response1 = client.post("/api/convert", json=conversion_payload)

        assert response1.status_code == 200
        extracted_text = response1.json()[0]["FileContent"]
        assert "machine learning" in extracted_text.lower()

        answer = "Machine learning is a subset of AI that enables systems to learn from data."
        with patch("tilellm.__main__.ask_hybrid_with_memory", AsyncMock(return_value=_answer(answer, {}))):
            response2 = client.post(
                "/api/qa", json=_qa_payload(f"Based on this document: {extracted_text}. What is machine learning?"))

        assert response2.status_code == 200
        assert "machine learning" in response2.json()["answer"].lower()

    def test_structured_output_flow(self, client: TestClient):
        """A structured-output request to /api/ask returns the structured answer."""
        from pydantic import BaseModel

        class PersonInfo(BaseModel):
            name: str
            age: int
            occupation: str

        payload = {
            "question": "Extract person information: Mario Esempio, 30 years old, software engineer",
            "llm": "openai",
            "model": "gpt-4o-mini",
            "llm_key": "test-key",
            "stream": False,
            "structured_output": True,
            "output_schema": PersonInfo.model_json_schema(),
        }
        structured = {"name": "Mario Esempio", "age": 30, "occupation": "software engineer"}

        with patch("tilellm.__main__.ask_to_llm",
                   AsyncMock(return_value=SimpleAnswer(answer=structured))) as ask:
            response = client.post("/api/ask", json=payload)

        assert response.status_code == 200
        assert response.json()["answer"] == structured
        assert ask.await_args.kwargs["question"].structured_output is True


class TestErrorFlows:
    """Test error scenarios in complete flows."""

    def test_rag_with_missing_dependencies(self, client: TestClient):
        """A failing dependency inside the controller becomes an error response, not a crash."""
        # the app turns it into an error response; TestClient would re-raise it
        lenient = TestClient(client.app, raise_server_exceptions=False)
        with patch("tilellm.__main__.ask_hybrid_with_memory",
                   AsyncMock(side_effect=Exception("Missing API key"))):
            response = lenient.post("/api/qa", json=_qa_payload("Test question"))

        assert response.status_code in [400, 500]

    def test_invalid_rag_request_is_rejected(self, client: TestClient):
        """A body without namespace/engine never reaches the controller (422)."""
        with patch("tilellm.__main__.ask_hybrid_with_memory", AsyncMock()) as hybrid:
            response = client.post("/api/qa", json={"question": "q", "llm_key": None})

        assert response.status_code == 422
        hybrid.assert_not_called()

    def test_conversion_invalid_file(self, client: TestClient):
        """Test conversion with invalid file content."""
        payload = {
            "file_name": "invalid.pdf",
            "file_content": "not-a-valid-base64",
            "conversion_type": "pdf_to_text",
        }

        with patch("tilellm.modules.conversion.controllers.process_pdf_to_text",
                   side_effect=Exception("Invalid PDF content")):
            response = client.post("/api/convert", json=payload)

        assert response.status_code == 400
