#!/usr/bin/env python3
"""
Unit tests for controller functions.
Test controller logic with mocked dependencies.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uuid

from tilellm.controller.controller import (
    ask_hybrid_with_memory,
    ask_reason_llm,
    ask_to_llm,
    ask_mcp_agent_llm,
    handle_agent_exception,
    _process_history_messages,
    _build_message_list,
    _format_history_as_text,
    _update_history,
    _extract_token_info
)
from tilellm.models import ChatEntry, QuestionToLLM, QuestionAnswer
from tilellm.models.schemas import PromptTokenInfo


class TestControllerFunctions:
    """Test core controller functions."""
    
    @pytest.mark.asyncio
    async def test_handle_agent_exception(self):
        """Test agent exception handler returns proper JSONResponse."""
        test_exception = Exception("Test error")
        result = handle_agent_exception(test_exception, "test context")
        
        assert isinstance(result, JSONResponse)
        assert result.status_code == 500
        content = result.body.decode('utf-8')
        assert "Test error" in content or "An error has occurred" in content
    
    def test_format_history_as_text_empty(self):
        """Test formatting empty history."""
        result = _format_history_as_text({})
        assert result == "No previous conversation."
    
    def test_format_history_as_text_with_entries(self):
        """Test formatting history with entries."""
        history = {
            "0": ChatEntry(question="Hello", answer="Hi there"),
            "1": ChatEntry(question="How are you?", answer="I'm good")
        }
        result = _format_history_as_text(history)
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result
        assert "User: How are you?" in result
        assert "Assistant: I'm good" in result
    
    def test_update_history_empty(self):
        """Test updating empty history."""
        result = _update_history({}, "New question", "New answer")
        assert isinstance(result, dict)
        assert len(result) == 1
        assert "0" in result
        assert result["0"].question == "New question"
        assert result["0"].answer == "New answer"
    
    def test_update_history_existing(self):
        """Test updating existing history."""
        existing = {
            "0": ChatEntry(question="Q1", answer="A1"),
            "1": ChatEntry(question="Q2", answer="A2")
        }
        result = _update_history(existing, "Q3", "A3")
        assert len(result) == 3
        assert result["2"].question == "Q3"
        assert result["2"].answer == "A3"
    
    def test_extract_token_info_with_metadata(self):
        """Test extracting token info from message with metadata."""
        mock_message = Mock()
        mock_message.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }
        
        result = _extract_token_info(mock_message)
        assert isinstance(result, PromptTokenInfo)
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
    
    def test_extract_token_info_no_metadata(self):
        """Test extracting token info from message without metadata."""
        mock_message = Mock()
        mock_message.usage_metadata = {}
        
        result = _extract_token_info(mock_message)
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0
    
    def test_build_message_list(self):
        """Test building message list from history."""
        history = {
            "0": ChatEntry(question="Q1", answer="A1"),
            "1": ChatEntry(question="Q2", answer="A2")
        }
        
        result = _build_message_list(history, ["0", "1"])
        assert len(result) == 4  # 2 human + 2 AI messages
        from langchain_core.messages import HumanMessage, AIMessage
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert result[0].content == "Q1"
        assert result[1].content == "A1"


class TestAskToLLM:
    """Test ask_to_llm controller function."""
    
    @pytest.mark.asyncio
    async def test_ask_to_llm_basic(self):
        """Test basic ask_to_llm without streaming."""
        question = QuestionToLLM(
            question="Test question",
            llm="gpt-4",
            llm_key="test-key",
            stream=False
        )
        
        mock_chat_model = AsyncMock()
        mock_message = Mock()
        mock_message.content = "Test answer"
        mock_message.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_message)
        
        # Mock the LLM cache to return our mock chat model
        with patch('tilellm.shared.utility.TimedCache.async_get', AsyncMock(return_value=mock_chat_model)):
            result = await ask_to_llm(question)
            
            assert isinstance(result, JSONResponse)
            assert result.status_code == 200
            content = json.loads(result.body.decode('utf-8'))
            assert content['answer'] == "Test answer"
            assert 'chat_history_dict' in content
    
    @pytest.mark.asyncio
    async def test_ask_to_llm_with_history(self):
        """Test ask_to_llm with chat history."""
        history = {
            "0": ChatEntry(question="Previous", answer="Previous answer")
        }
        question = QuestionToLLM(
            question="Current question",
            llm="gpt-4",
            llm_key="test-key",
            chat_history_dict=history,
            stream=False,
            contextualize_prompt=False
        )
        
        mock_chat_model = AsyncMock()
        mock_message = Mock()
        mock_message.content = "Current answer"
        mock_message.usage_metadata = {}
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_message)
        
        with patch('tilellm.shared.utility.TimedCache.async_get', AsyncMock(return_value=mock_chat_model)):
            result = await ask_to_llm(question)
            
            content = json.loads(result.body.decode('utf-8'))
            assert content['answer'] == "Current answer"
            # History should be updated
            assert len(content['chat_history_dict']) == 2
    
    @pytest.mark.asyncio
    async def test_ask_to_llm_streaming(self):
        """Test ask_to_llm with streaming."""
        question = QuestionToLLM(
            question="Test question",
            llm="gpt-4",
            llm_key="test-key",
            stream=True
        )
        
        mock_chat_model = AsyncMock()
        # Mock streaming response
        mock_chat_model.astream = AsyncMock(return_value=[])
        
        with patch('tilellm.controller.controller.inject_llm_async', lambda f: f):
            result = await ask_to_llm(question, chat_model=mock_chat_model)
            
            assert isinstance(result, StreamingResponse)
            assert result.media_type == "text/event-stream"
    
    @pytest.mark.asyncio
    async def test_ask_to_llm_structured_output(self):
        """Test ask_to_llm with structured output."""
        from pydantic import BaseModel
        
        class TestSchema(BaseModel):
            name: str
            age: int
        
        question = QuestionToLLM(
            question="Return test data",
            llm="gpt-4",
            llm_key="test-key",
            stream=False,
            structured_output=True,
            output_schema=TestSchema.model_json_schema()
        )
        
        mock_structured_llm = AsyncMock()
        mock_result = Mock()
        mock_result.model_dump = Mock(return_value={"name": "John", "age": 30})
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_result)
        
        mock_chat_model = AsyncMock()
        mock_chat_model.with_structured_output = Mock(return_value=mock_structured_llm)
        
        with patch('tilellm.controller.controller.inject_llm_async', lambda f: f):
            result = await ask_to_llm(question, chat_model=mock_chat_model)
            
            content = json.loads(result.body.decode('utf-8'))
            assert content['answer'] == {"name": "John", "age": 30}


class TestAskReasonLLM:
    """Test ask_reason_llm controller function."""
    
    @pytest.mark.asyncio
    async def test_ask_reason_llm_basic(self):
        """Test basic ask_reason_llm without streaming."""
        question = QuestionToLLM(
            question="Reasoning question",
            llm="deepseek",
            llm_key="test-key",
            stream=False
        )
        
        mock_chat_model = AsyncMock()
        mock_message = Mock()
        mock_message.content = "Reasoned answer"
        mock_message.usage_metadata = {}
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_message)
        
        # Mock get_reasoning_content function
        with patch('tilellm.controller.controller.inject_reason_llm_async', lambda f: f):
            with patch('tilellm.controller.controller.get_reasoning_content', 
                      return_value=(False, "Reasoned answer", "")):
                result = await ask_reason_llm(question, chat_model=mock_chat_model)
                
                assert isinstance(result, JSONResponse)
                content = json.loads(result.body.decode('utf-8'))
                assert content['answer'] == "Reasoned answer"
                assert content['reasoning_content'] == ""


class TestAskHybridWithMemory:
    """Test ask_hybrid_with_memory controller function."""
    
    @pytest.mark.asyncio
    async def test_ask_hybrid_with_memory_basic(self):
        """Test basic hybrid search with memory."""
        question = QuestionAnswer(
            question="Test question",
            top_k=5,
            reranking=False,
            chat_history_dict={}
        )
        
        # Mock dependencies
        mock_repo = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm_embeddings = Mock()
        
        # Setup mock returns
        mock_repo.initialize_embeddings_and_index = AsyncMock(return_value=(384, None, None))
        mock_repo.perform_hybrid_search = AsyncMock(return_value=[])
        
        # Mock controller_utils functions
        with patch('tilellm.controller.controller.inject_repo_async', lambda f: f):
            with patch('tilellm.controller.controller.inject_llm_chat_async', lambda f: f):
                with patch('tilellm.controller.controller.preprocess_chat_history', 
                          return_value=([], [])):
                    with patch('tilellm.controller.controller.fetch_question_vectors',
                              AsyncMock(return_value=(None, None))):
                        with patch('tilellm.controller.controller.retrieve_documents',
                                  return_value=Mock()):
                            with patch('tilellm.controller.controller.create_chains',
                                      AsyncMock(return_value=(Mock(), Mock(), Mock()))):
                                with patch('tilellm.controller.controller.generate_answer_with_history',
                                          AsyncMock(return_value=Mock())):
                                    result = await ask_hybrid_with_memory(
                                        question,
                                        repo=mock_repo,
                                        llm=mock_llm,
                                        llm_embeddings=mock_llm_embeddings
                                    )
                                    
                                    # Should return the mocked result from generate_answer_with_history
                                    assert result is not None


class TestProcessHistoryMessages:
    """Test _process_history_messages function."""
    
    @pytest.mark.asyncio
    async def test_process_history_no_history(self):
        """Test with no history."""
        mock_chat_model = AsyncMock()
        result = await _process_history_messages({}, None, False, mock_chat_model)
        assert result == []
    
    @pytest.mark.asyncio
    async def test_process_history_within_limit(self):
        """Test with history within message limit."""
        history = {
            "0": ChatEntry(question="Q1", answer="A1"),
            "1": ChatEntry(question="Q2", answer="A2")
        }
        mock_chat_model = AsyncMock()
        
        result = await _process_history_messages(history, 5, False, mock_chat_model)
        assert len(result) == 4  # 2 human + 2 AI messages
    
    @pytest.mark.asyncio
    async def test_process_history_exceeds_limit_no_summarize(self):
        """Test history exceeds limit without summarization."""
        history = {
            "0": ChatEntry(question="Q1", answer="A1"),
            "1": ChatEntry(question="Q2", answer="A2"),
            "2": ChatEntry(question="Q3", answer="A3")
        }
        mock_chat_model = AsyncMock()
        
        result = await _process_history_messages(history, 2, False, mock_chat_model)
        # Should only include last 2 turns (4 messages)
        assert len(result) == 4
    
    @pytest.mark.asyncio
    async def test_process_history_exceeds_limit_with_summarize(self):
        """Test history exceeds limit with summarization."""
        history = {
            "0": ChatEntry(question="Q1", answer="A1"),
            "1": ChatEntry(question="Q2", answer="A2"),
            "2": ChatEntry(question="Q3", answer="A3")
        }
        mock_chat_model = AsyncMock()
        mock_chat_model.ainvoke = AsyncMock(return_value=Mock(content="Summary of old conversation"))
        
        result = await _process_history_messages(history, 2, True, mock_chat_model)
        # Should include system message with summary + last 2 turns
        assert len(result) == 5  # system + 2 human + 2 AI