"""
Test script for regex_custom chunking strategy
Tests the handle_regex_custom_chunk function
"""
import sys
sys.path.insert(0, '/home/lor/sviluppo/tiledesk/tiledesk-llm')

import pytest
from unittest.mock import patch, AsyncMock
from langchain_core.documents import Document
from tilellm.tools.document_tools import handle_regex_custom_chunk


def test_regex_custom_chunk_basic():
    """Test basic regex chunking with page markers"""
    print("\n=== Test 1: Basic regex chunking ===")
    
    test_content = """========== INIZIO: pagina_1.md ==========
Contenuto della pagina 1.
Questa è la prima pagina del documento.
========== FINE: pagina_1.md ==========

========== INIZIO: pagina_2.md ==========
Contenuto della pagina 2.
Questa è la seconda pagina del documento.
========== FINE: pagina_2.md ==========

========== INIZIO: pagina_3.md ==========
Contenuto della pagina 3.
Ultima pagina del documento.
========== FINE: pagina_3.md =========="""

    chunk_regex = r"={10,}\s*INIZIO: pagina_(\d+)\.md\s*={10,}(.*?)\s*={10,}\s*FINE: pagina_\1\.md\s*={10,}"
    
    with patch('tilellm.tools.document_tools.requests.get') as mock_get:
        mock_response = mock_get.return_value
        mock_response.text = test_content
        mock_response.raise_for_status = lambda: None
        
        import asyncio
        documents = asyncio.run(handle_regex_custom_chunk(
            url="http://example.com/test.txt",
            chunk_regex=chunk_regex
        ))
    
    print(f"Number of documents: {len(documents)}")
    
    assert len(documents) == 3, f"Expected 3 documents, got {len(documents)}"
    
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: page_number={doc.metadata.get('page_number')}")
        print(f"  Content preview: {doc.page_content[:50]}...")
        
        assert doc.metadata.get('page_number') == i + 1, f"Expected page_number {i+1}"
        assert f"pagina {i+1}" in doc.page_content.lower() or f"pagina_{i+1}" in doc.page_content.lower()
    
    print("✓ Basic regex chunking works correctly")


def test_regex_custom_chunk_no_matches():
    """Test behavior when regex finds no matches"""
    print("\n=== Test 2: No matches ===")
    
    test_content = "This is just regular text without any page markers."
    
    chunk_regex = r"={10,}\s*INIZIO: pagina_(\d+)\.md\s*={10,}(.*?)\s*={10,}\s*FINE: pagina_\1\.md\s*={10,}"
    
    with patch('tilellm.tools.document_tools.requests.get') as mock_get:
        mock_response = mock_get.return_value
        mock_response.text = test_content
        mock_response.raise_for_status = lambda: None
        
        import asyncio
        documents = asyncio.run(handle_regex_custom_chunk(
            url="http://example.com/test.txt",
            chunk_regex=chunk_regex
        ))
    
    print(f"Number of documents: {len(documents)}")
    
    assert len(documents) == 1, "Should return 1 document when no matches"
    assert documents[0].page_content == test_content
    assert documents[0].metadata.get('page_number') is None
    
    print("✓ No matches fallback works correctly")


def test_regex_custom_chunk_empty_content():
    """Test behavior with empty page content"""
    print("\n=== Test 3: Empty content in some pages ===")
    
    test_content = """========== INIZIO: pagina_1.md ==========
Contenuto della pagina 1.
========== FINE: pagina_1.md ==========

========== INIZIO: pagina_2.md ==========

========== FINE: pagina_2.md ==========

========== INIZIO: pagina_3.md ==========
Contenuto della pagina 3.
========== FINE: pagina_3.md =========="""

    chunk_regex = r"={10,}\s*INIZIO: pagina_(\d+)\.md\s*={10,}(.*?)\s*={10,}\s*FINE: pagina_\1\.md\s*={10,}"
    
    with patch('tilellm.tools.document_tools.requests.get') as mock_get:
        mock_response = mock_get.return_value
        mock_response.text = test_content
        mock_response.raise_for_status = lambda: None
        
        import asyncio
        documents = asyncio.run(handle_regex_custom_chunk(
            url="http://example.com/test.txt",
            chunk_regex=chunk_regex
        ))
    
    print(f"Number of documents: {len(documents)}")
    
    assert len(documents) == 2, "Should skip empty pages"
    assert documents[0].metadata.get('page_number') == 1
    assert documents[1].metadata.get('page_number') == 3
    
    print("✓ Empty content handling works correctly")


def test_regex_custom_chunk_missing_regex():
    """Test that missing regex raises error"""
    print("\n=== Test 4: Missing regex parameter ===")
    
    import asyncio
    
    with pytest.raises(ValueError, match="chunk_regex is required"):
        asyncio.run(handle_regex_custom_chunk(
            url="http://example.com/test.txt",
            chunk_regex=None
        ))
    
    print("✓ Missing regex validation works correctly")


def test_regex_custom_chunk_metadata():
    """Test that metadata is correctly populated"""
    print("\n=== Test 5: Metadata population ===")
    
    test_content = """========== INIZIO: pagina_1.md ==========
Contenuto pagina 1
========== FINE: pagina_1.md =========="""

    chunk_regex = r"={10,}\s*INIZIO: pagina_(\d+)\.md\s*={10,}(.*?)\s*={10,}\s*FINE: pagina_\1\.md\s*={10,}"
    
    with patch('tilellm.tools.document_tools.requests.get') as mock_get:
        mock_response = mock_get.return_value
        mock_response.text = test_content
        mock_response.raise_for_status = lambda: None
        
        import asyncio
        documents = asyncio.run(handle_regex_custom_chunk(
            url="http://example.com/test.txt",
            chunk_regex=chunk_regex
        ))
    
    doc = documents[0]
    
    assert 'source' in doc.metadata
    assert doc.metadata['source'] == "http://example.com/test.txt"
    assert doc.metadata['type'] == 'regex_custom'
    assert doc.metadata['page_number'] == 1
    
    print(f"Metadata: {doc.metadata}")
    print("✓ Metadata population works correctly")


if __name__ == "__main__":
    test_regex_custom_chunk_basic()
    test_regex_custom_chunk_no_matches()
    test_regex_custom_chunk_empty_content()
    test_regex_custom_chunk_missing_regex()
    test_regex_custom_chunk_metadata()
    print("\n✅ All tests passed!")
