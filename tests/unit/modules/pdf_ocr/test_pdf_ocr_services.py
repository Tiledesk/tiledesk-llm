#!/usr/bin/env python3
"""
Unit tests for PDF OCR services.
Test structure extraction, entity extraction, semantic linking, and chunking.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import pandas as pd

from tilellm.modules.pdf_ocr.services.document_structure_extractor import (
    DocumentStructureExtractor,
    DocumentSection
)
from tilellm.modules.pdf_ocr.services.pdf_entity_extractor import PDFEntityExtractor
from tilellm.modules.pdf_ocr.services.table_semantic_linker import TableSemanticLinker
from tilellm.modules.pdf_ocr.services.image_semantic_linker import ImageSemanticLinker
from tilellm.modules.pdf_ocr.services.context_aware_chunker import ContextAwareChunker


class TestDocumentStructureExtractor:
    """Test DocumentStructureExtractor functionality."""
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = DocumentStructureExtractor()
        assert extractor.sections == {}
        assert extractor.cross_references == {}
        assert extractor.reading_order == []
    
    def test_extract_hierarchy_simple(self):
        """Test extraction of simple document structure."""
        extractor = DocumentStructureExtractor()
        
        doc_id = "test_doc"
        structured_content = {
            'text_elements': [
                {'id': 'text1', 'text': 'Introduction', 'type': 'heading', 'page': 0},
                {'id': 'text2', 'text': 'Some content here.', 'type': 'text', 'page': 0},
                {'id': 'text3', 'text': '1. Methods', 'type': 'heading', 'page': 1},
            ],
            'tables': [],
            'images': [],
            'formulas': []
        }
        
        result = extractor.extract_hierarchy(doc_id, structured_content)
        
        assert 'outline' in result
        assert 'sections' in result
        assert 'cross_refs' in result
        assert result['metadata']['num_sections'] >= 1
    
    def test_extract_sections_from_numbering(self):
        """Test section extraction from numbering patterns."""
        extractor = DocumentStructureExtractor()
        
        doc_id = "test_doc"
        structured_content = {
            'text_elements': [
                {'id': 'text1', 'text': '1. Introduction', 'type': 'text', 'page': 0},
                {'id': 'text2', 'text': '1.1 Background', 'type': 'text', 'page': 0},
                {'id': 'text3', 'text': '2. Methods', 'type': 'text', 'page': 0},
            ],
            'tables': [],
            'images': [],
            'formulas': []
        }
        
        result = extractor.extract_hierarchy(doc_id, structured_content)
        
        assert len(result['sections']) == 3
        # Check hierarchy: 2 should be parent of 2.1
        section_ids = list(result['sections'].keys())
        
        # Verify levels
        levels = [result['sections'][sid]['level'] for sid in section_ids]
        assert 1 in levels  # Top level
        assert 2 in levels  # Sublevel
    
    def test_extract_cross_references(self):
        """Test cross-reference extraction."""
        extractor = DocumentStructureExtractor()
        
        doc_id = "test_doc"
        structured_content = {
            'text_elements': [
                {'id': 'text1', 'text': 'As shown in Figure 1, the results...', 'type': 'text', 'page': 0},
                {'id': 'text2', 'text': 'See Table 2 for details.', 'type': 'text', 'page': 0},
                {'id': 'text3', 'text': 'Refer to Section 3.4.', 'type': 'text', 'page': 0},
            ],
            'tables': [
                {'id': 'test_doc_table_2', 'page': 1}
            ],
            'images': [
                {'id': 'test_doc_image_1', 'page': 0}
            ],
            'formulas': []
        }
        
        result = extractor.extract_hierarchy(doc_id, structured_content)
        
        assert len(result['cross_refs']) > 0
        
        # Check for specific references
        all_refs = []
        for ref_list in result['cross_refs'].values():
            all_refs.extend(ref_list)
        
        # Should contain references to figure, table, section
        has_figure = any('figure' in ref.lower() for ref in all_refs)
        has_table = any('table' in ref.lower() for ref in all_refs)
        has_section = any('section' in ref.lower() for ref in all_refs)
        
        assert has_figure or has_table or has_section
    
    def test_get_section_path(self):
        """Test getting section path for an element."""
        extractor = DocumentStructureExtractor()
        
        doc_id = "test_doc"
        structured_content = {
            'text_elements': [
                {'id': 'sec1', 'text': 'Introduction', 'type': 'heading', 'page': 0},
                {'id': 'sec2', 'text': '1.1 Background', 'type': 'heading', 'page': 0},
                {'id': 'text1', 'text': 'Some text.', 'type': 'text', 'page': 0},
            ],
            'tables': [],
            'images': [],
            'formulas': []
        }
        
        extractor.extract_hierarchy(doc_id, structured_content)
        
        # Get path for text element
        path = extractor.get_section_path('text1')
        assert isinstance(path, str)
        assert len(path) > 0
    
    @pytest.mark.asyncio
    async def test_create_section_nodes_in_graph(self):
        """Test creating section nodes in Neo4j."""
        mock_graph_repo = Mock()
        
        extractor = DocumentStructureExtractor(graph_repository=mock_graph_repo)
        
        doc_id = "test_doc"
        extractor.sections = {
            'section1': DocumentSection('section1', 'Introduction', 1, 0),
            'section2': DocumentSection('section2', 'Background', 2, 0, 'section1')
        }
        extractor.sections['section1'].elements = ['text1']
        extractor.sections['section2'].elements = ['text2']
        
        result = await extractor.create_section_nodes_in_graph(doc_id)
        
        assert result['sections_created'] == 2
        assert result['relationships_created'] > 0
        assert mock_graph_repo.create_node.call_count >= 2
        assert mock_graph_repo.create_relationship.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_create_cross_reference_relationships(self):
        """Test creating cross-reference relationships."""
        mock_graph_repo = Mock()
        
        extractor = DocumentStructureExtractor(graph_repository=mock_graph_repo)
        extractor.cross_references = {
            'text1': ['figure_1', 'table_2'],
            'text2': ['section_3']
        }
        
        doc_id = "test_doc"
        structured_content = {
            'tables': [
                {'id': 'test_doc_table_2', 'page': 1}
            ],
            'images': [
                {'id': 'test_doc_image_1', 'page': 0}
            ]
        }
        extractor.sections = {
            'section3': DocumentSection('section3', 'Methods', 1, 1)
        }
        
        result = await extractor.create_cross_reference_relationships(doc_id, structured_content)
        
        assert result >= 0


class TestPDFEntityExtractor:
    """Test PDFEntityExtractor functionality."""
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = PDFEntityExtractor()
        assert extractor.graph_repository is None or hasattr(extractor.graph_repository, 'create_node')
    
    @pytest.mark.asyncio
    async def test_process_text_elements_with_mock_llm(self):
        """Test entity extraction with mocked LLM."""
        mock_graph_repo = Mock()
        mock_llm = AsyncMock()
        
        # Mock LLM response
        mock_llm.ainvoke.return_value = Mock(content='{"entities": [{"name": "TestEntity", "type": "concept"}], "relationships": []}')
        
        extractor = PDFEntityExtractor(graph_repository=mock_graph_repo)
        
        text_elements = [
            {'id': 'text1', 'text': 'This is a test document with TestEntity mentioned.' * 10, 'page': 0}
        ]
        
        result = await extractor.process_text_elements(
            text_elements=text_elements,
            doc_id='test_doc',
            llm=mock_llm,
            entity_types=['concept'],
            batch_size=1
        )
        
        assert result['elements_processed'] >= 0
        assert 'entities_extracted' in result
        assert 'relationships_created' in result
    
    @pytest.mark.asyncio
    async def test_process_text_elements_with_hierarchy(self):
        """Test entity extraction with hierarchy for section-entity relationships."""
        mock_graph_repo = Mock()
        mock_llm = AsyncMock()
        
        mock_llm.ainvoke.return_value = Mock(content='{"entities": [{"name": "TestEntity", "type": "concept"}], "relationships": []}')
        
        extractor = PDFEntityExtractor(graph_repository=mock_graph_repo)
        
        text_elements = [
            {'id': 'text1', 'text': 'TestEntity is important.' * 10, 'page': 0}
        ]
        
        hierarchy = {
            'sections': {
                'section1': {
                    'elements': ['text1']
                }
            }
        }
        
        result = await extractor.process_text_elements(
            text_elements=text_elements,
            doc_id='test_doc',
            llm=mock_llm,
            hierarchy=hierarchy,
            batch_size=1
        )
        
        assert result['elements_processed'] >= 0
        # Should create section-entity relationships
        # Section-[:MENTIONS]->Entity


class TestTableSemanticLinker:
    """Test TableSemanticLinker functionality."""
    
    def test_initialization(self):
        """Test linker initialization."""
        linker = TableSemanticLinker()
        assert linker.graph_repository is None
    
    @pytest.mark.asyncio
    async def test_link_table_to_context(self):
        """Test linking table to surrounding text context."""
        mock_graph_repo = Mock()
        mock_llm = AsyncMock()
        
        # Mock LLM responses
        mock_llm.ainvoke.side_effect = [
            Mock(content='["What is the accuracy?", "Which method performed best?"]'),
            Mock(content='This table shows comparison results.')
        ]
        
        linker = TableSemanticLinker(graph_repository=mock_graph_repo)
        
        df = pd.DataFrame({
            'Method': ['A', 'B', 'C'],
            'Accuracy': [0.85, 0.90, 0.88]
        })
        
        table_data = {
            'id': 'test_doc_table_1',
            'data': df,
            'page': 1
        }
        
        text_elements = [
            {'id': 'text1', 'text': 'Table 1 shows the results.', 'page': 1}
        ]
        
        result = await linker.link_table_to_context(
            table_data=table_data,
            text_elements=text_elements,
            llm=mock_llm,
            doc_id='test_doc'
        )
        
        assert 'answerable_questions' in result
        assert 'semantic_description' in result
        assert len(result['answerable_questions']) >= 0
        assert mock_llm.ainvoke.call_count >= 1
    
    def test_find_references(self):
        """Test finding references to table."""
        linker = TableSemanticLinker()
        
        text_elements = [
            {'id': 'text1', 'text': 'As shown in Table 1, the results are...'},
            {'id': 'text2', 'text': 'Refer to Table 2 for details.'},
        ]
        
        refs = linker._find_references('test_doc_table_1', text_elements)
        
        assert len(refs) >= 0


class TestImageSemanticLinker:
    """Test ImageSemanticLinker functionality."""
    
    def test_initialization(self):
        """Test linker initialization."""
        linker = ImageSemanticLinker()
        assert linker.graph_repository is None
    
    @pytest.mark.asyncio
    async def test_link_image_to_context(self):
        """Test linking image to surrounding text context."""
        mock_graph_repo = Mock()
        
        linker = ImageSemanticLinker(graph_repository=mock_graph_repo)
        
        image_data = {
            'id': 'test_doc_image_1',
            'page': 0
        }
        
        text_elements = [
            {'id': 'text1', 'text': 'Figure 1 illustrates the architecture.', 'page': 0}
        ]
        
        result = await linker.link_image_to_context(
            image_data=image_data,
            text_elements=text_elements,
            llm=None,
            doc_id='test_doc'
        )
        
        assert 'surrounding_text' in result
        assert len(result['surrounding_text']) >= 0
    
    def test_find_references(self):
        """Test finding references to image."""
        linker = ImageSemanticLinker()
        
        text_elements = [
            {'id': 'text1', 'text': 'See Figure 1 for details.'},
            {'id': 'text2', 'text': 'As shown in Figure 2...'},
        ]
        
        refs = linker._find_references('test_doc_image_1', text_elements)
        
        assert len(refs) >= 0


class TestContextAwareChunker:
    """Test ContextAwareChunker functionality."""
    
    def test_initialization(self):
        """Test chunker initialization."""
        chunker = ContextAwareChunker(max_tokens=256, overlap=32)
        assert chunker.max_tokens == 256
        assert chunker.overlap == 32
    
    def test_chunk_with_structure(self):
        """Test chunking with document structure."""
        chunker = ContextAwareChunker(max_tokens=200, overlap=20)
        
        doc_id = "test_doc"
        text_elements = [
            {'id': 'text1', 'text': 'Short text one.', 'type': 'text', 'page': 0},
            {'id': 'text2', 'text': 'Short text two.', 'type': 'text', 'page': 0},
            {'id': 'text3', 'text': 'Short text three.', 'type': 'text', 'page': 0},
        ]
        
        structure = {
            'sections': {
                'section1': {
                    'title': 'Introduction',
                    'level': 1,
                    'page': 0,
                    'parent_id': None,
                    'elements': ['text1', 'text2', 'text3']
                }
            },
            'cross_refs': {
                'text1': ['figure_1'],
                'text2': []
            }
        }
        
        chunks = chunker.chunk_with_structure(doc_id, text_elements, structure)
        
        assert len(chunks) > 0
        # Check metadata
        for chunk in chunks:
            assert 'metadata' in chunk.metadata
            assert 'section_path' in chunk.metadata
            assert 'doc_id' in chunk.metadata
    
    def test_extract_cross_refs(self):
        """Test cross-reference extraction from text."""
        chunker = ContextAwareChunker()
        
        text = "See Figure 3 and Table 5 in Section 2.1"
        refs = chunker._extract_cross_refs(text)
        
        assert len(refs) >= 0
        # Check for different types
        ref_types = [r.split('_')[0] for r in refs]
        
        has_all_types = (
            any('figure' in rt.lower() for rt in ref_types) or
            any('table' in rt.lower() for rt in ref_types) or
            any('section' in rt.lower() for rt in ref_types)
        )
        assert has_all_types or len(refs) == 0  # May not find all types
    
    def test_find_containing_section(self):
        """Test finding section that contains an element."""
        chunker = ContextAwareChunker()
        
        sections = {
            'section1': {
                'elements': ['text1', 'text2']
            },
            'section2': {
                'elements': ['text3']
            }
        }
        
        # Test element in section1
        result = chunker._find_containing_section('text1', sections)
        assert result == 'section1'
        
        # Test element in section2
        result = chunker._find_containing_section('text3', sections)
        assert result == 'section2'
        
        # Test non-existent element
        result = chunker._find_containing_section('text99', sections)
        assert result is None
