"""
Unit tests for GraphRAG Extractor utility.
Tests entity and relationship extraction functionality.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from tilellm.modules.knowledge_graph_falkor.tools.graphrag_extractor import (
    GraphRAGExtractor, clean_str, split_string_by_multi_markers,
    handle_single_entity_extraction, handle_single_relationship_extraction,
    flat_uniq_list, GRAPH_EXTRACTION_PROMPT
)


class TestCleanStr:
    """Test string cleaning utility."""
    
    def test_clean_simple_string(self):
        """Test cleaning a simple string."""
        result = clean_str("  Hello World  ")
        assert result == "Hello World"
    
    def test_clean_with_html_entities(self):
        """Test cleaning HTML entities."""
        result = clean_str("&lt;div&gt;Test&lt;/div&gt;")
        assert "<" in result or "<div>" in result or result == "<div>Test</div>"
    
    def test_clean_non_string_input(self):
        """Test cleaning non-string input."""
        result = clean_str(123)
        assert result == "123"
    
    def test_clean_control_characters(self):
        """Test removing control characters."""
        result = clean_str("Hello\x00World\x01")
        assert "\x00" not in result
        assert "\x01" not in result


class TestSplitStringByMultiMarkers:
    """Test string splitting utility."""
    
    def test_split_with_single_marker(self):
        """Test splitting with one marker."""
        result = split_string_by_multi_markers("a,b,c", [","])
        assert result == ["a", "b", "c"]
    
    def test_split_with_multiple_markers(self):
        """Test splitting with multiple markers."""
        result = split_string_by_multi_markers("a,b;c|d", [",", ";", "|"])
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "d" in result
    
    def test_split_empty_markers(self):
        """Test splitting with empty markers list."""
        result = split_string_by_multi_markers("hello world", [])
        assert result == ["hello world"]
    
    def test_split_with_whitespace(self):
        """Test that whitespace is stripped."""
        result = split_string_by_multi_markers("  a  ,  b  ", [","])
        assert "a" in result
        assert "b" in result
        assert "  a  " not in result


class TestHandleSingleEntityExtraction:
    """Test entity extraction from record."""
    
    def test_valid_entity_extraction(self):
        """Test extracting valid entity."""
        record = ['"entity"', '"John Doe"', '"person"', '"A software engineer"']
        result = handle_single_entity_extraction(record, "chunk_1")
        
        assert result is not None
        assert result["entity_name"] == "JOHN DOE"
        assert result["entity_type"] == "PERSON"
        assert result["source_id"] == "chunk_1"
    
    def test_invalid_entity_record(self):
        """Test extracting from invalid record."""
        record = ['"relationship"', '"A"', '"B"']  # Wrong type
        result = handle_single_entity_extraction(record, "chunk_1")
        
        assert result is None
    
    def test_empty_entity_name(self):
        """Test extracting entity with empty name."""
        record = ['"entity"', '"  "', '"person"', '"desc"']
        result = handle_single_entity_extraction(record, "chunk_1")
        
        assert result is None
    
    def test_short_record(self):
        """Test extracting from too short record."""
        record = ['"entity"', '"John"']  # Missing type and description
        result = handle_single_entity_extraction(record, "chunk_1")
        
        assert result is None


class TestHandleSingleRelationshipExtraction:
    """Test relationship extraction from record."""
    
    def test_valid_relationship_extraction(self):
        """Test extracting valid relationship."""
        record = ['"relationship"', '"Alice"', '"Bob"', '"Friends"', '"social"', '0.8']
        result = handle_single_relationship_extraction(record, "chunk_1")
        
        assert result is not None
        assert result["src_id"] == "ALICE"
        assert result["tgt_id"] == "BOB"
        assert result["relationship_type"] == "FRIENDS"
        assert result["description"] == "social"
        assert result["source_id"] == "chunk_1"
    
    def test_relationship_with_weight(self):
        """Test extracting relationship with weight."""
        record = ['"relationship"', '"A"', '"B"', '"desc"', '"keywords"', '0.9']
        result = handle_single_relationship_extraction(record, "chunk_1")
        
        assert result is not None
        assert result["weight"] == 0.9
    
    def test_relationship_without_weight(self):
        """Test extracting relationship without explicit weight."""
        record = ['"relationship"', '"A"', '"B"', '"desc"']
        result = handle_single_relationship_extraction(record, "chunk_1")
        
        assert result is not None
        assert result["weight"] == 1.0  # Default weight
    
    def test_invalid_relationship_record(self):
        """Test extracting from invalid record."""
        record = ['"entity"', '"A"']  # Wrong type
        result = handle_single_relationship_extraction(record, "chunk_1")
        
        assert result is None
    
    def test_missing_source_or_target(self):
        """Test extracting with empty source/target."""
        record = ['"relationship"', '"  "', '"B"', '"desc"']
        result = handle_single_relationship_extraction(record, "chunk_1")
        
        assert result is None


class TestFlatUniqList:
    """Test flat unique list utility."""
    
    def test_flatten_simple_list(self):
        """Test flattening simple list values."""
        data = [
            {"tags": ["a", "b"]},
            {"tags": ["b", "c"]}
        ]
        result = flat_uniq_list(data, "tags")
        
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert len(result) == 3  # Unique values only
    
    def test_flatten_single_values(self):
        """Test flattening single (non-list) values."""
        data = [
            {"name": "Alice"},
            {"name": "Bob"}
        ]
        result = flat_uniq_list(data, "name")
        
        assert "Alice" in result
        assert "Bob" in result
    
    def test_flatten_with_none(self):
        """Test handling None values."""
        data = [
            {"tags": ["a"]},
            {"tags": None}
        ]
        result = flat_uniq_list(data, "tags")
        
        assert "a" in result


class TestGraphRAGExtractorInitialization:
    """Test GraphRAGExtractor initialization."""
    
    def test_extractor_init_with_defaults(self):
        """Test extractor with default parameters."""
        mock_llm = Mock()
        extractor = GraphRAGExtractor(mock_llm)
        
        assert extractor.llm == mock_llm
        assert extractor.language == "English"
        assert len(extractor.entity_types) > 0
    
    def test_extractor_init_with_custom_params(self):
        """Test extractor with custom parameters."""
        mock_llm = Mock()
        custom_types = ["company", "product"]
        extractor = GraphRAGExtractor(
            llm_invoker=mock_llm,
            language="Italian",
            entity_types=custom_types
        )
        
        assert extractor.language == "Italian"
        assert extractor.entity_types == custom_types


class TestGraphRAGExtractorExtractChunk:
    """Test GraphRAGExtractor chunk extraction."""
    
    @pytest.mark.asyncio
    async def test_extract_chunk_with_invoke(self):
        """Test chunk extraction with LLM invoke method."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
        ("entity","John","person","A person")
        ("entity","Mary","person","Another person")
        ("relationship","John","Mary","Friends","social",0.9)
        [COMPLETED]
        """
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        extractor = GraphRAGExtractor(mock_llm)
        entities, relationships, token_count = await extractor.extract_chunk(
            chunk_key="chunk_1",
            chunk_text="John and Mary are friends."
        )
        
        assert len(entities) == 2
        assert len(relationships) == 1
        assert entities["JOHN"][0]["entity_name"] == "JOHN"
        assert entities["MARY"][0]["entity_name"] == "MARY"
        assert relationships[("JOHN", "MARY")][0]["src_id"] == "JOHN"
        assert relationships[("JOHN", "MARY")][0]["tgt_id"] == "MARY"
        assert relationships[("JOHN", "MARY")][0]["relationship_type"] == "FRIENDS"
    
    @pytest.mark.asyncio
    async def test_extract_chunk_with_chat_method(self):
        """Test chunk extraction with LLM chat method."""
        mock_llm = Mock()
        mock_llm.invoke = None
        del mock_llm.invoke
        mock_response = """
        ("entity","Company A","organization","A company")
        [COMPLETED]
        """
        mock_llm.chat = AsyncMock(return_value=mock_response)
        mock_llm.ainvoke = None  # No ainvoke method
        
        extractor = GraphRAGExtractor(mock_llm)
        entities, relationships, token_count = await extractor.extract_chunk(
            chunk_key="chunk_1",
            chunk_text="Company A is a business."
        )
        
        assert len(entities) == 1
    
    @pytest.mark.asyncio
    async def test_extract_chunk_extraction_error(self):
        """Test handling extraction error."""
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        
        extractor = GraphRAGExtractor(mock_llm)
        
        entities, relationships, token_count = await extractor.extract_chunk("chunk_1", "Some text")
        assert len(entities) == 0
        assert len(relationships) == 0
        assert token_count == 0


class TestGraphRAGExtractorExtract:
    """Test GraphRAGExtractor full extraction."""
    
    @pytest.mark.asyncio
    async def test_extract_from_multiple_chunks(self):
        """Test extracting from multiple chunks."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
        ("entity","Alice","person","Person A")
        ("entity","Bob","person","Person B")
        ("relationship","Alice","Bob","Colleagues","work",0.8)
        [COMPLETED]
        """
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        extractor = GraphRAGExtractor(mock_llm)
        chunks = [
            {"id": "chunk_1", "text": "Alice works with Bob."},
            {"id": "chunk_2", "text": "They are colleagues."}
        ]
        
        entities, relationships = await extractor.extract("doc_1", chunks)
        
        assert len(entities) > 0
        assert len(relationships) > 0
    
    @pytest.mark.asyncio
    async def test_extract_with_empty_chunks(self):
        """Test extracting with empty chunk list."""
        mock_llm = Mock()
        extractor = GraphRAGExtractor(mock_llm)
        
        entities, relationships = await extractor.extract("doc_1", [])
        
        assert len(entities) == 0
        assert len(relationships) == 0
    
    @pytest.mark.asyncio
    async def test_extract_deduplication(self):
        """Test entity deduplication across chunks."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
        ("entity","Alice","person","Person A")
        [COMPLETED]
        """
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        extractor = GraphRAGExtractor(mock_llm)
        chunks = [
            {"id": "chunk_1", "text": "Alice is here."},
            {"id": "chunk_2", "text": "Alice is there."}
        ]
        
        entities, relationships = await extractor.extract("doc_1", chunks)
        
        # Should only have one Alice despite appearing in both chunks
        alice_entities = [e for e in entities if e["entity_name"] == "ALICE"]
        assert len(alice_entities) == 1


class TestGraphExtractionPrompt:
    """Test extraction prompt template."""
    
    def test_prompt_contains_required_variables(self):
        """Test that prompt contains required template variables."""
        assert "{entity_types}" in GRAPH_EXTRACTION_PROMPT
        assert "{input_text}" in GRAPH_EXTRACTION_PROMPT
        assert "{tuple_delimiter}" in GRAPH_EXTRACTION_PROMPT
        assert "{record_delimiter}" in GRAPH_EXTRACTION_PROMPT
        assert "{completion_delimiter}" in GRAPH_EXTRACTION_PROMPT
    
    def test_prompt_contains_examples(self):
        """Test that prompt contains examples section."""
        assert "-Examples-" in GRAPH_EXTRACTION_PROMPT
        assert "Example 1" in GRAPH_EXTRACTION_PROMPT
    
    def test_prompt_formatting(self):
        """Test prompt variable substitution."""
        formatted = GRAPH_EXTRACTION_PROMPT.format(
            entity_types="person, organization",
            input_text="Sample text",
            tuple_delimiter="|",
            record_delimiter="\n",
            completion_delimiter="[END]"
        )
        
        assert "person, organization" in formatted
        assert "Sample text" in formatted
        assert "|" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
