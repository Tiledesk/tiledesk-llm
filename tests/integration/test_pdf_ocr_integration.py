#!/usr/bin/env python3
"""
Integration tests for PDF OCR module.
Test complete PDF processing workflow with GraphRAG integration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import os

from tilellm.modules.pdf_ocr.logic import (
    process_pdf_document_with_embeddings,
    _invoke_llm_chat
)
from tilellm.modules.pdf_ocr.services.docling_processor import ProductionDocumentProcessor
from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest


class TestPDFOCRIntegration:
    """Integration tests for complete PDF processing workflow."""
    
    @pytest.mark.asyncio
    async def test_process_pdf_document_with_embeddings_basic(self, mocker):
        """Test basic PDF processing with embeddings."""
        # Mock dependencies
        mock_repo = Mock()
        mock_repo.aadd_documents = AsyncMock()
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = Mock(content="Test description")
        
        mock_llm_embeddings = Mock()
        mock_llm_embeddings.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
        
        # Mock document processor
        mock_processor = Mock()
        mock_processor.process_document = AsyncMock(return_value={
            'text_elements': [
                {'id': 'text1', 'text': 'Test content', 'type': 'text', 'page': 0}
            ],
            'tables': [],
            'images': [],
            'formulas': [],
            'metadata': {'doc_id': 'test_doc', 'num_pages': 1},
            'hierarchy': None
        })
        mock_processor.close = AsyncMock()
        
        mocker.patch(
            'tilellm.modules.pdf_ocr.logic.ProductionDocumentProcessor',
            return_value=mock_processor
        )
        
        # Create request
        request = PDFScrapingRequest(
            id='test_doc',
            file_name='test.pdf',
            file_content='fake_content',
            use_docling=True,
            include_text=True,
            include_tables=False,
            include_images=False,
            extract_entities=False,
            namespace='test_ns',
            engine='qdrant'
        )
        
        # Process
        result = await process_pdf_document_with_embeddings(
            question=request,
            file_path='/fake/path.pdf',
            repo=mock_repo,
            llm=mock_llm,
            llm_embeddings=mock_llm_embeddings
        )
        
        # Assertions
        assert result['status'] == 'success'
        assert result['doc_id'] == 'test_doc'
        assert 'statistics' in result
        assert mock_repo.aadd_documents.called
    
    @pytest.mark.asyncio
    async def test_process_pdf_with_entity_extraction(self, mocker):
        """Test PDF processing with entity extraction enabled."""
        # Mock dependencies
        mock_repo = Mock()
        mock_repo.aadd_documents = AsyncMock()
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = Mock(
            content='{"entities": [{"name": "TestEntity", "type": "concept"}], "relationships": []}'
        )
        
        mock_llm_embeddings = Mock()
        mock_llm_embeddings.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
        
        # Mock graph extractor
        mock_entity_extractor = Mock()
        mock_entity_extractor.process_text_elements = AsyncMock(
            return_value={
                'entities_extracted': 1,
                'relationships_created': 0,
                'elements_processed': 1
            }
        )
        
        mocker.patch(
            'tilellm.modules.pdf_ocr.logic.PDFEntityExtractor',
            return_value=mock_entity_extractor
        )
        
        # Mock document processor
        mock_processor = Mock()
        mock_processor.process_document = AsyncMock(return_value={
            'text_elements': [
                {'id': 'text1', 'text': 'TestEntity is important.' * 10, 'type': 'text', 'page': 0}
            ],
            'tables': [],
            'images': [],
            'formulas': [],
            'metadata': {'doc_id': 'test_doc', 'num_pages': 1},
            'hierarchy': {
                'sections': {
                    'section1': {
                        'elements': ['text1']
                    }
                }
            }
        })
        mock_processor.close = AsyncMock()
        
        mocker.patch(
            'tilellm.modules.pdf_ocr.logic.ProductionDocumentProcessor',
            return_value=mock_processor
        )
        
        # Create request with entity extraction enabled
        request = PDFScrapingRequest(
            id='test_doc',
            file_name='test.pdf',
            file_content='fake_content',
            use_docling=True,
            include_text=True,
            include_tables=False,
            include_images=False,
            extract_entities=True,
            namespace='test_ns',
            engine='qdrant'
        )
        
        # Process
        result = await process_pdf_document_with_embeddings(
            question=request,
            file_path='/fake/path.pdf',
            repo=mock_repo,
            llm=mock_llm,
            llm_embeddings=mock_llm_embeddings
        )
        
        # Assertions
        assert result['status'] == 'success'
        assert mock_entity_extractor.process_text_elements.called
        # Check that hierarchy was passed to entity extractor
        call_args = mock_entity_extractor.process_text_elements.call_args
        assert 'hierarchy' in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_process_pdf_with_tables_and_images(self, mocker):
        """Test PDF processing with tables and images."""
        # Mock dependencies
        mock_repo = Mock()
        mock_repo.aadd_documents = AsyncMock()
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = Mock(content="Generated caption/description")
        
        mock_llm_embeddings = Mock()
        mock_llm_embeddings.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
        
        # Mock linkers
        mock_table_linker = Mock()
        mock_table_linker.link_table_to_context = AsyncMock(
            return_value={'answerable_questions': [], 'semantic_description': ''}
        )
        
        mock_image_linker = Mock()
        mock_image_linker.link_image_to_context = AsyncMock()
        
        mocker.patch(
            'tilellm.modules.pdf_ocr.logic.TableSemanticLinker',
            return_value=mock_table_linker
        )
        mocker.patch(
            'tilellm.modules.pdf_ocr.logic.ImageSemanticLinker',
            return_value=mock_image_linker
        )
        
        # Mock document processor
        mock_processor = Mock()
        mock_processor.process_document = AsyncMock(return_value={
            'text_elements': [
                {'id': 'text1', 'text': 'Some text.', 'type': 'text', 'page': 0}
            ],
            'tables': [
                {'id': 'table1', 'data': None, 'page': 1}  # Empty for test
            ],
            'images': [
                {'id': 'image1', 'image_data': None, 'page': 1}  # None for test
            ],
            'formulas': [],
            'metadata': {'doc_id': 'test_doc', 'num_pages': 1},
            'hierarchy': None
        })
        mock_processor.close = AsyncMock()
        
        mocker.patch(
            'tilellm.modules.pdf_ocr.logic.ProductionDocumentProcessor',
            return_value=mock_processor
        )
        
        # Create request with tables and images enabled
        request = PDFScrapingRequest(
            id='test_doc',
            file_name='test.pdf',
            file_content='fake_content',
            use_docling=True,
            include_text=True,
            include_tables=True,
            include_images=True,
            extract_entities=False,
            namespace='test_ns',
            engine='qdrant'
        )
        
        # Process
        result = await process_pdf_document_with_embeddings(
            question=request,
            file_path='/fake/path.pdf',
            repo=mock_repo,
            llm=mock_llm,
            llm_embeddings=mock_llm_embeddings
        )
        
        # Assertions
        assert result['status'] == 'success'
        assert result['statistics']['tables'] == 1
        assert result['statistics']['images'] == 1


class TestPDFOCRGraphIntegration:
    """Integration tests for GraphRAG in PDF OCR."""
    
    @pytest.mark.asyncio
    async def test_document_structure_creates_graph_nodes(self, mocker):
        """Test that document structure extraction creates graph nodes."""
        from tilellm.modules.pdf_ocr.services.document_structure_extractor import DocumentStructureExtractor
        
        # Mock graph repository
        mock_graph_repo = Mock()
        mock_graph_repo.create_node = Mock()
        mock_graph_repo.create_relationship = Mock()
        
        # Create extractor with mocked repository
        extractor = DocumentStructureExtractor(graph_repository=mock_graph_repo)
        
        # Mock document content
        doc_id = "test_doc"
        structured_content = {
            'text_elements': [
                {'id': 'text1', 'text': 'Introduction', 'type': 'heading', 'page': 0},
                {'id': 'text2', 'text': 'Content here.', 'type': 'text', 'page': 0},
            ],
            'tables': [],
            'images': [],
            'formulas': []
        }
        
        # Extract hierarchy
        hierarchy = extractor.extract_hierarchy(doc_id, structured_content)
        
        # Create section nodes
        await extractor.create_section_nodes_in_graph(doc_id)
        
        # Verify Document node created
        doc_node_calls = [
            call for call in mock_graph_repo.create_node.call_args_list
            if call.kwargs.get('label') == 'Document'
        ]
        assert len(doc_node_calls) >= 1
        
        # Verify Section node(s) created
        section_node_calls = [
            call for call in mock_graph_repo.create_node.call_args_list
            if call.kwargs.get('label') == 'Section'
        ]
        assert len(section_node_calls) >= 1
        
        # Verify relationships created
        assert mock_graph_repo.create_relationship.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_entity_extraction_creates_entity_nodes(self, mocker):
        """Test that entity extraction creates entity nodes and relationships."""
        from tilellm.modules.pdf_ocr.services.pdf_entity_extractor import PDFEntityExtractor
        
        # Mock graph repository
        mock_graph_repo = Mock()
        
        # Mock LLM
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = Mock(
            content='{"entities": [{"name": "TestEntity", "type": "concept"}], "relationships": []}'
        )
        
        # Create extractor
        extractor = PDFEntityExtractor(graph_repository=mock_graph_repo)
        
        # Process text elements
        text_elements = [
            {'id': 'text1', 'text': 'TestEntity is important.' * 10, 'page': 0}
        ]
        
        result = await extractor.process_text_elements(
            text_elements=text_elements,
            doc_id='test_doc',
            llm=mock_llm,
            entity_types=['concept'],
            batch_size=1
        )
        
        # Verify Entity node created
        entity_node_calls = [
            call for call in mock_graph_repo.create_node.call_args_list
            if call.kwargs.get('label') == 'Entity'
        ]
        assert len(entity_node_calls) >= 1
        
        # Verify MENTIONS relationship created (Paragraph -> Entity)
        mentions_rel_calls = [
            call for call in mock_graph_repo.create_relationship.call_args_list
            if call.kwargs.get('type') == 'MENTIONS'
        ]
        assert len(mentions_rel_calls) >= 1
    
    @pytest.mark.asyncio
    async def test_cross_references_create_references_relationships(self, mocker):
        """Test that cross-references create REFERENCES relationships."""
        from tilellm.modules.pdf_ocr.services.document_structure_extractor import DocumentStructureExtractor
        
        # Mock graph repository
        mock_graph_repo = Mock()
        mock_graph_repo.create_relationship = Mock()
        
        # Create extractor
        extractor = DocumentStructureExtractor(graph_repository=mock_graph_repo)
        
        # Set up cross-references
        extractor.cross_references = {
            'text1': ['figure_1', 'table_2']
        }
        
        # Mock content with matching elements
        doc_id = "test_doc"
        structured_content = {
            'tables': [
                {'id': 'test_doc_table_2', 'page': 1}
            ],
            'images': [
                {'id': 'test_doc_image_1', 'page': 0}
            ]
        }
        extractor.sections = {}
        
        # Create cross-reference relationships
        count = await extractor.create_cross_reference_relationships(doc_id, structured_content)
        
        # Verify REFERENCES relationships created
        ref_rel_calls = [
            call for call in mock_graph_repo.create_relationship.call_args_list
            if call.kwargs.get('type') == 'REFERENCES'
        ]
        assert len(ref_rel_calls) >= 0


class TestLLMHelper:
    """Test the _invoke_llm_chat helper function."""
    
    @pytest.mark.asyncio
    async def test_invoke_llm_chat_basic(self, mocker):
        """Test basic LLM invocation with helper."""
        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = Mock(content="Test response")
        mock_llm.ainvoke.return_value = mock_response
        
        # Test invocation
        result = await _invoke_llm_chat(
            system_prompt="You are a helpful assistant.",
            human_prompt="Hello!",
            llm=mock_llm
        )
        
        # Verify LLM was called with correct message format
        assert mock_llm.ainvoke.called
        call_args = mock_llm.ainvoke.call_args[0][0]
        
        # Check that messages were passed
        assert len(call_args) == 2
        assert call_args[0].content == "You are a helpful assistant."
        assert call_args[1].content == "Hello!"
        
        # Check result
        assert result == "Test response"
    
    @pytest.mark.asyncio
    async def test_invoke_llm_chat_error_handling(self, mocker):
        """Test LLM invocation error handling."""
        # Mock LLM that raises exception
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("LLM error")
        
        # Test invocation - should raise exception
        with pytest.raises(Exception):
            await _invoke_llm_chat(
                system_prompt="You are a helpful assistant.",
                human_prompt="Hello!",
                llm=mock_llm
            )
    
    @pytest.mark.asyncio
    async def test_invoke_llm_chat_with_no_llm(self):
        """Test LLM invocation with None LLM."""
        # Test with None LLM - should raise ValueError
        with pytest.raises(ValueError):
            await _invoke_llm_chat(
                system_prompt="You are a helpful assistant.",
                human_prompt="Hello!",
                llm=None
            )
