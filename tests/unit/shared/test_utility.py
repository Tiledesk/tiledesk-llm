#!/usr/bin/env python3
"""
Unit tests for shared utility functions.
"""
import pytest
import yaml
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
import hashlib

from tilellm.shared.utility import (
    get_service_config,
    _hash_api_key,
    inject_llm_async,
    inject_llm_chat_async,
    inject_repo_async,
    inject_reason_llm_async
)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_hash_api_key(self):
        """Test API key hashing."""
        test_key = "test-api-key-123"
        result = _hash_api_key(test_key)
        
        expected = hashlib.sha256(test_key.encode()).hexdigest()
        assert result == expected
    
    def test_hash_api_key_empty(self):
        """Test hashing empty API key."""
        result = _hash_api_key("")
        expected = hashlib.sha256("".encode()).hexdigest()
        assert result == expected
    
    @patch.dict(os.environ, {"TILELLM_PROFILE": "app-base"})
    def test_get_service_config_profile_base(self):
        """Test loading service config from TILELLM_PROFILE environment variable."""
        result = get_service_config()
        
        assert "services" in result
        assert result["services"]["task_executor"] is True
        assert result["services"]["graphrag"] is False
        assert result["services"]["graphrag_falkor"] is False
        assert result["services"]["pdf_ocr"] is False
        assert result["services"]["conversion"] is True
        assert result["services"]["tools_registry"] is True
    
    @patch.dict(os.environ, {"TILELLM_PROFILE": "app-graph"})
    def test_get_service_config_profile_graph(self):
        """Test loading service config with graphrag enabled."""
        result = get_service_config()
        
        assert "services" in result
        assert result["services"]["graphrag"] is True
        assert result["services"]["graphrag_falkor"] is False
        assert result["services"]["pdf_ocr"] is False
    
    @patch.dict(os.environ, {"TILELLM_PROFILE": "app-ocr"})
    def test_get_service_config_profile_ocr(self):
        """Test loading service config with PDF OCR enabled."""
        result = get_service_config()
        
        assert "services" in result
        assert result["services"]["pdf_ocr"] is True
        assert result["services"]["graphrag"] is True
        assert result["services"]["graphrag_falkor"] is False
    
    @patch.dict(os.environ, {"TILELLM_PROFILE": "app-all"})
    def test_get_service_config_profile_all(self):
        """Test loading service config with all modules enabled."""
        result = get_service_config()
        
        assert "services" in result
        assert result["services"]["task_executor"] is True
        assert result["services"]["graphrag"] is True
        assert result["services"]["graphrag_falkor"] is False
        assert result["services"]["pdf_ocr"] is True
        assert result["services"]["conversion"] is True
        assert result["services"]["tools_registry"] is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_service_config_no_profile(self):
        """Test loading service config with no profile (uses individual flags)."""
        result = get_service_config()
        
        assert "services" in result
        # Default values should be used
        assert result["services"]["task_executor"] is True
        assert result["services"]["graphrag"] is False
        assert result["services"]["graphrag_falkor"] is False
        assert result["services"]["pdf_ocr"] is False
    
    @patch.dict(os.environ, {"ENABLE_GRAPHRAG": "true", "ENABLE_PDF_OCR": "yes"})
    def test_get_service_config_individual_flags(self):
        """Test loading service config with individual enable flags."""
        result = get_service_config()
        
        assert "services" in result
        assert result["services"]["graphrag"] is True
        assert result["services"]["graphrag_falkor"] is False
        assert result["services"]["pdf_ocr"] is True
    
    def test_str_to_bool(self):
        """Test string to boolean conversion."""
        from tilellm.shared.utility import _str_to_bool
        
        assert _str_to_bool("true") is True
        assert _str_to_bool("True") is True
        assert _str_to_bool("TRUE") is True
        assert _str_to_bool("1") is True
        assert _str_to_bool("yes") is True
        assert _str_to_bool("on") is True
        
        assert _str_to_bool("false") is False
        assert _str_to_bool("False") is False
        assert _str_to_bool("0") is False
        assert _str_to_bool("no") is False
        assert _str_to_bool("off") is False
        assert _str_to_bool("") is False
        assert _str_to_bool("random") is False


class TestGoogleVertexAiRouting:
    """Vertex AI routing for the 'google' LLM provider (project/location on LlmEmbeddingModel)."""

    def test_apply_google_vertex_flag_sets_vertexai_when_project_present(self):
        from tilellm.shared.utility import _apply_google_vertex_flag

        client_config = {"model": "gemini-2.5-flash", "project": "poc-tiledesk-496310"}
        result = _apply_google_vertex_flag(client_config)

        assert result["vertexai"] is True

    def test_apply_google_vertex_flag_leaves_config_untouched_without_project(self):
        from tilellm.shared.utility import _apply_google_vertex_flag

        client_config = {"model": "gemini-2.5-flash"}
        result = _apply_google_vertex_flag(client_config)

        assert "vertexai" not in result

    def test_apply_google_vertex_flag_strips_project_and_location(self):
        """
        project/location must NOT reach ChatGoogleGenerativeAI: the google-genai
        SDK discards an env-var API key whenever they're explicit constructor
        args, forcing an unwanted ADC lookup (google.auth.exceptions.DefaultCredentialsError)
        at request time instead of using the supplied api_key.
        """
        from tilellm.shared.utility import _apply_google_vertex_flag

        client_config = {
            "model": "gemini-2.5-flash",
            "google_api_key": "AQ.Ab8RN6Jjqyi62sQ",
            "project": "poc-tiledesk-496310",
            "location": "europe-west8",
        }
        result = _apply_google_vertex_flag(client_config)

        assert result["vertexai"] is True
        assert "project" not in result
        assert "location" not in result
        assert result["google_api_key"] == "AQ.Ab8RN6Jjqyi62sQ"

    @pytest.mark.asyncio
    async def test_get_llm_config_for_client_propagates_project_and_location(self):
        from tilellm.shared.utility import _get_llm_config_for_client
        from tilellm.models.embedding import LlmEmbeddingModel
        from tilellm.models.base import LLMEmbeddingProviders
        from unittest.mock import MagicMock

        question = MagicMock()
        question.llm = "google"
        question.model = LlmEmbeddingModel(
            provider=LLMEmbeddingProviders.GOOGLE,
            name="gemini-2.5-flash",
            api_key="AQ.Ab8RN6Jjqyi62sQ",
            project="poc-tiledesk-496310",
            location="europe-west8",
        )

        client_config = await _get_llm_config_for_client(question, {})

        assert client_config["project"] == "poc-tiledesk-496310"
        assert client_config["location"] == "europe-west8"

    @pytest.mark.asyncio
    async def test_get_llm_config_for_client_omits_project_when_absent(self):
        from tilellm.shared.utility import _get_llm_config_for_client
        from tilellm.models.embedding import LlmEmbeddingModel
        from tilellm.models.base import LLMEmbeddingProviders

        from unittest.mock import MagicMock

        question = MagicMock()
        question.llm = "google"
        question.model = LlmEmbeddingModel(
            provider=LLMEmbeddingProviders.GOOGLE,
            name="gemini-2.5-flash",
            api_key="sk-ai-studio",
        )

        client_config = await _get_llm_config_for_client(question, {})

        assert "project" not in client_config
        assert "location" not in client_config


class TestInjectDecorators:
    """Test dependency injection decorators."""
    
    @pytest.mark.asyncio
    async def test_inject_llm_async(self):
        """Test LLM injection decorator."""
        mock_func = AsyncMock(return_value="result")
        
        # Apply decorator
        decorated = inject_llm_async(mock_func)
        
        # Call with question that has llm_key
        mock_question = Mock()
        mock_question.llm_key = "test-key"
        mock_question.llm = "gpt-4"
        
        # Mock the internal _get_llm function
        with patch('tilellm.shared.utility._get_llm', return_value="mocked-llm"):
            result = await decorated(mock_question)
            
            # Function should be called with injected llm
            mock_func.assert_called_once_with(mock_question, "mocked-llm")
            assert result == "result"
    
    @pytest.mark.asyncio
    async def test_inject_llm_async_no_key(self):
        """Test LLM injection with missing API key."""
        mock_func = AsyncMock()
        
        decorated = inject_llm_async(mock_func)
        
        mock_question = Mock()
        mock_question.llm_key = None
        mock_question.llm = "gpt-4"
        
        with patch('tilellm.shared.utility._get_llm', return_value=None):
            result = await decorated(mock_question)
            
            # Function should be called with None llm
            mock_func.assert_called_once_with(mock_question, None)
    
    @pytest.mark.asyncio
    async def test_inject_llm_chat_async(self):
        """Test LLM chat injection decorator."""
        mock_func = AsyncMock(return_value="result")
        
        decorated = inject_llm_chat_async(mock_func)
        
        mock_question = Mock()
        mock_question.llm_key = "test-key"
        mock_question.llm = "gpt-4"
        
        with patch('tilellm.shared.utility._get_llm_chat', return_value="mocked-chat-llm"):
            result = await decorated(mock_question)
            
            mock_func.assert_called_once_with(mock_question, "mocked-chat-llm")
            assert result == "result"
    
    @pytest.mark.asyncio
    async def test_inject_repo_async(self):
        """Test repository injection decorator."""
        mock_func = AsyncMock(return_value="result")
        
        decorated = inject_repo_async(mock_func)
        
        mock_question = Mock()
        mock_question.engine = Mock()
        
        with patch('tilellm.shared.utility._get_repo', AsyncMock(return_value="mocked-repo")):
            result = await decorated(mock_question)
            
            mock_func.assert_called_once_with(mock_question, "mocked-repo")
            assert result == "result"
    
    @pytest.mark.asyncio
    async def test_inject_reason_llm_async(self):
        """Test reasoning LLM injection decorator."""
        mock_func = AsyncMock(return_value="result")
        
        decorated = inject_reason_llm_async(mock_func)
        
        mock_question = Mock()
        mock_question.llm_key = "test-key"
        mock_question.llm = "deepseek"
        
        with patch('tilellm.shared.utility._get_reason_llm', return_value="mocked-reason-llm"):
            result = await decorated(mock_question)
            
            mock_func.assert_called_once_with(mock_question, "mocked-reason-llm")
            assert result == "result"


# Test other utility functions if needed
class TestOtherUtilities:
    """Test other utility functions."""
    
    def test_timed_cache(self):
        """Test TimedCache functionality."""
        from tilellm.shared.timed_cache import TimedCache
        
        cache = TimedCache(default_ttl=1.0)  # 1 second TTL
        
        # Set value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Set with custom TTL
        cache.set("key2", "value2", ttl=0.1)  # 0.1 second TTL
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        
        # key2 should be expired, key1 still valid
        assert cache.get("key2") is None
        assert cache.get("key1") == "value1"
        
        # Wait for key1 expiration
        time.sleep(0.9)
        assert cache.get("key1") is None
    
    def test_embedding_factory(self):
        """Test embedding factory creation."""
        from tilellm.shared.embedding_factory import EmbeddingFactory
        
        factory = EmbeddingFactory()
        
        # Test getting OpenAI embeddings
        with patch('langchain_openai.embeddings.OpenAIEmbeddings') as MockEmbeddings:
            mock_embeddings = Mock()
            MockEmbeddings.return_value = mock_embeddings
            
            result = factory.create_embeddings(
                provider="openai",
                api_key="test-key",
                model="text-embedding-ada-002"
            )
            
            assert result == mock_embeddings
            MockEmbeddings.assert_called_once_with(
                api_key="test-key",
                model="text-embedding-ada-002"
            )