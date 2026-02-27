"""
Markdown Extractor Service for PDF OCR Module.

This service extracts high-quality Markdown from PDF documents using Docling,
enhances it with LLM-generated descriptions for images and tables,
and produces a structured Markdown document suitable for indexing.
"""

import logging
import asyncio
import base64
import io
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import ConversionResult
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class MarkdownElement:
    """Represents an element in the Markdown document."""
    element_type: str  # 'text', 'heading', 'table', 'image', 'formula', 'list'
    content: str
    page_number: int
    metadata: Dict[str, Any]
    order: int


class MarkdownExtractor:
    """
    Extracts enhanced Markdown from PDF documents.
    
    Features:
    - Full document Markdown extraction using Docling
    - LLM-enhanced image descriptions with visual analysis
    - LLM-enhanced table descriptions with semantic analysis
    - Structured output with proper Markdown formatting
    - Context-aware element ordering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._init_docling()
        
    def _init_docling(self):
        """Initialize Docling converter."""
        if not DOCLING_AVAILABLE:
            logger.warning("Docling not available. Markdown extraction will be limited.")
            self.pdf_converter = None
            return
            
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            
            self.pdf_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("MarkdownExtractor: Docling initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Docling: {e}")
            self.pdf_converter = None
    
    async def extract_markdown(
        self,
        file_path: str,
        doc_id: str,
        llm=None,
        include_images: bool = True,
        include_tables: bool = True,
        include_formulas: bool = True
    ) -> Dict[str, Any]:
        """
        Extract enhanced Markdown from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            doc_id: Document identifier
            llm: LLM instance for generating descriptions
            include_images: Whether to include image descriptions
            include_tables: Whether to include table descriptions
            include_formulas: Whether to include formula extraction
            
        Returns:
            Dict containing:
                - markdown: Complete Markdown string
                - elements: List of MarkdownElement objects
                - metadata: Document metadata
                - images: List of image data with descriptions
                - tables: List of table data with descriptions
        """
        if not self.pdf_converter:
            raise RuntimeError("Docling converter not initialized")
        
        logger.info(f"Starting Markdown extraction for document {doc_id}")
        
        # Convert PDF using Docling
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.pdf_converter.convert, file_path)
        
        # Extract structured elements
        elements = await self._extract_elements(result, doc_id)
        
        # Process images with LLM if available
        images_data = []
        if include_images and llm:
            images_data = await self._process_images_for_markdown(result, doc_id, llm)
        
        # Process tables with LLM if available
        tables_data = []
        if include_tables and llm:
            tables_data = await self._process_tables_for_markdown(result, doc_id, llm)
        
        # Generate final Markdown
        markdown_content = await self._generate_enhanced_markdown(
            elements=elements,
            images_data=images_data,
            tables_data=tables_data,
            result=result,
            doc_id=doc_id
        )
        
        # Extract metadata
        metadata = self._extract_metadata(result, doc_id)
        
        logger.info(f"Completed Markdown extraction for document {doc_id}")
        
        return {
            "markdown": markdown_content,
            "elements": elements,
            "metadata": metadata,
            "images": images_data,
            "tables": tables_data
        }
    
    async def _extract_elements(self, result: Any, doc_id: str) -> List[MarkdownElement]:
        """Extract structured elements from Docling result."""
        elements = []
        element_order = 0
        
        doc = getattr(result, 'document', None)
        
        if doc:
            logger.info("Using Docling v2 document-centric extraction for Markdown")
            
            # Extract headings
            headings = getattr(doc, 'headings', [])
            for item in headings:
                text = getattr(item, 'text', '')
                if not text:
                    continue
                    
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov:
                    prov = prov[0]
                page_no = getattr(prov, 'page_no', 1)
                
                level = getattr(item, 'level', 1)
                heading_markers = '#' * min(level, 6)
                
                elements.append(MarkdownElement(
                    element_type='heading',
                    content=f"{heading_markers} {text}",
                    page_number=page_no - 1,
                    metadata={'level': level, 'raw_text': text},
                    order=element_order
                ))
                element_order += 1
            
            # Extract text paragraphs
            texts = getattr(doc, 'texts', [])
            for item in texts:
                text = getattr(item, 'text', '')
                if not text:
                    continue
                    
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov:
                    prov = prov[0]
                page_no = getattr(prov, 'page_no', 1)
                
                elements.append(MarkdownElement(
                    element_type='text',
                    content=text,
                    page_number=page_no - 1,
                    metadata={'label': getattr(item, 'label', 'text')},
                    order=element_order
                ))
                element_order += 1
            
            # Extract lists
            lists = getattr(doc, 'lists', [])
            for item in lists:
                items = getattr(item, 'items', [])
                if not items:
                    continue
                    
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov:
                    prov = prov[0]
                page_no = getattr(prov, 'page_no', 1)
                
                list_content = '\n'.join([f"- {list_item}" for list_item in items])
                
                elements.append(MarkdownElement(
                    element_type='list',
                    content=list_content,
                    page_number=page_no - 1,
                    metadata={'num_items': len(items)},
                    order=element_order
                ))
                element_order += 1
        
        # Sort elements by page and reading order
        elements.sort(key=lambda x: (x.page_number, x.order))
        
        return elements
    
    async def _process_images_for_markdown(
        self,
        result: Any,
        doc_id: str,
        llm
    ) -> List[Dict[str, Any]]:
        """Process images and generate enhanced descriptions."""
        images_data = []
        
        doc = getattr(result, 'document', None)
        if not doc:
            return images_data
        
        pictures = getattr(doc, 'pictures', [])
        
        for idx, item in enumerate(pictures):
            try:
                # Get image data
                image_data = item.get_image(doc) if hasattr(item, 'get_image') else getattr(item, 'image', None)
                if not image_data:
                    continue
                
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov:
                    prov = prov[0]
                page_no = getattr(prov, 'page_no', 1)
                
                # Convert to bytes for LLM
                img_byte_arr = io.BytesIO()
                image_data.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                
                # Generate enhanced description using LLM
                description = await self._generate_image_description(img_bytes, llm, page_no)
                
                image_id = f"{doc_id}_img_{idx}"
                
                images_data.append({
                    'id': image_id,
                    'page': page_no - 1,
                    'image_data': image_data,
                    'description': description,
                    'alt_text': description[:100] + '...' if len(description) > 100 else description
                })
                
            except Exception as e:
                logger.error(f"Failed to process image {idx} for Markdown: {e}")
                continue
        
        logger.info(f"Processed {len(images_data)} images for Markdown")
        return images_data
    
    async def _generate_image_description(
        self,
        img_bytes: bytes,
        llm,
        page_no: int
    ) -> str:
        """Generate an enhanced description for an image using LLM."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            # Convert to base64
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Determine media type
            media_type = "image/png"
            if img_bytes.startswith(b'\x89PNG'):
                media_type = "image/png"
            elif img_bytes.startswith(b'\xff\xd8\xff'):
                media_type = "image/jpeg"
            
            # System prompt for detailed image analysis
            system_prompt = """You are an expert document analyst. Describe images from documents with rich detail.

Focus on:
1. Type of visual element (chart, graph, diagram, photograph, screenshot, etc.)
2. Key visual elements and their relationships
3. Any text, labels, numbers, or data visible in the image
4. Colors, patterns, or visual encoding used
5. The purpose or message the image conveys
6. Any trends, comparisons, or insights visible

Be specific and detailed. If it's a chart, describe what it measures. If it's a diagram, explain the flow or structure. If it's a photo, describe what's depicted.

Format your response as a well-structured paragraph suitable for embedding in a Markdown document."""
            
            # Build message with image
            llm_class_name = llm.__class__.__name__
            uses_openai_format = "OpenAI" in llm_class_name or "Google" in llm_class_name
            
            message_content = []
            message_content.append({
                "type": "text",
                "text": f"Analyze this image from page {page_no} of the document. Provide a comprehensive description."
            })
            
            if uses_openai_format:
                data_uri = f"data:{media_type};base64,{img_b64}"
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })
            else:
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64
                    }
                })
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message_content)
            ]
            
            response = await llm.ainvoke(messages)
            description = response.content if hasattr(response, 'content') else str(response)
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            return f"Image on page {page_no} (description generation failed)"
    
    async def _process_tables_for_markdown(
        self,
        result: Any,
        doc_id: str,
        llm
    ) -> List[Dict[str, Any]]:
        """Process tables and generate enhanced descriptions."""
        tables_data = []
        
        doc = getattr(result, 'document', None)
        if not doc:
            return tables_data
        
        tables = getattr(doc, 'tables', [])
        
        for idx, item in enumerate(tables):
            try:
                # Export to DataFrame
                df = item.export_to_dataframe(doc) if hasattr(item, 'export_to_dataframe') else pd.DataFrame()
                if df.empty:
                    continue
                
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov:
                    prov = prov[0]
                page_no = getattr(prov, 'page_no', 1)
                
                caption = getattr(item, 'caption', None) or ''
                
                # Generate enhanced description using LLM
                description = await self._generate_table_description(df, caption, llm, page_no)
                
                table_id = f"{doc_id}_tbl_{idx}"
                
                # Convert DataFrame to Markdown table
                md_table = df.to_markdown(index=False)
                
                tables_data.append({
                    'id': table_id,
                    'page': page_no - 1,
                    'dataframe': df,
                    'markdown_table': md_table,
                    'caption': caption,
                    'description': description,
                    'columns': list(df.columns),
                    'shape': df.shape
                })
                
            except Exception as e:
                logger.error(f"Failed to process table {idx} for Markdown: {e}")
                continue
        
        logger.info(f"Processed {len(tables_data)} tables for Markdown")
        return tables_data
    
    async def _generate_table_description(
        self,
        df: pd.DataFrame,
        caption: str,
        llm,
        page_no: int
    ) -> str:
        """Generate an enhanced semantic description for a table using LLM."""
        try:
            # Prepare table summary
            stats = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
            
            # Sample data
            sample_rows = df.head(3).to_dict('records')
            
            # Build prompt
            human_prompt = f"""Analyze this table from page {page_no} of a document and provide a comprehensive semantic description.

{caption if caption else 'No caption provided.'}

Table Structure:
- Rows: {stats['shape'][0]}, Columns: {stats['shape'][1]}
- Column names: {', '.join(stats['columns'])}
- Data types: {stats['dtypes']}

Sample Data (first 3 rows):
{sample_rows}

Provide a rich description that covers:
1. What kind of data this table contains and its purpose
2. The main categories or metrics represented
3. Relationships between columns (if any are apparent)
4. What insights or questions this table could answer
5. Any notable patterns, ranges, or data characteristics

Format as a well-structured paragraph suitable for a Markdown document."""
            
            system_prompt = """You are an expert data analyst. Describe tables from documents with semantic richness.

Focus on understanding what the data represents, not just technical details. Explain the meaning and significance of the table in the context of a document."""
            
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            description = response.content if hasattr(response, 'content') else str(response)
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Error generating table description: {e}")
            return f"Table with {len(df)} rows and {len(df.columns)} columns on page {page_no}"
    
    async def _generate_enhanced_markdown(
        self,
        elements: List[MarkdownElement],
        images_data: List[Dict[str, Any]],
        tables_data: List[Dict[str, Any]],
        result: Any,
        doc_id: str
    ) -> str:
        """Generate the final enhanced Markdown document."""
        markdown_parts = []
        
        # Add document header
        markdown_parts.append(f"# Document: {doc_id}\n")
        
        # Add metadata section
        doc = getattr(result, 'document', None)
        if doc:
            title = getattr(doc, 'title', None)
            if title:
                markdown_parts.append(f"**Title:** {title}\n")
        
        markdown_parts.append(f"**Document ID:** {doc_id}\n")
        markdown_parts.append("---\n")
        
        # Track which images and tables have been inserted
        inserted_images = set()
        inserted_tables = set()
        
        # Process elements page by page
        current_page = -1
        
        for element in elements:
            # Add page break if new page
            if element.page_number != current_page:
                current_page = element.page_number
                markdown_parts.append(f"\n## Page {current_page + 1}\n")
            
            # Add element content
            if element.element_type == 'heading':
                markdown_parts.append(f"\n{element.content}\n")
            
            elif element.element_type == 'text':
                markdown_parts.append(f"\n{element.content}\n")
            
            elif element.element_type == 'list':
                markdown_parts.append(f"\n{element.content}\n")
            
            # Check for images on this page that haven't been inserted
            page_images = [img for img in images_data 
                          if img['page'] == element.page_number and img['id'] not in inserted_images]
            
            for img in page_images:
                markdown_parts.append(f"\n### Image: {img['id']}\n")
                markdown_parts.append(f"**Location:** Page {img['page'] + 1}\n")
                markdown_parts.append(f"\n{img['description']}\n")
                markdown_parts.append(f"\n*Alt text: {img['alt_text']}*\n")
                inserted_images.add(img['id'])
            
            # Check for tables on this page that haven't been inserted
            page_tables = [tbl for tbl in tables_data 
                          if tbl['page'] == element.page_number and tbl['id'] not in inserted_tables]
            
            for tbl in page_tables:
                markdown_parts.append(f"\n### Table: {tbl['id']}\n")
                if tbl['caption']:
                    markdown_parts.append(f"**Caption:** {tbl['caption']}\n")
                markdown_parts.append(f"**Location:** Page {tbl['page'] + 1}\n")
                markdown_parts.append(f"\n{tbl['description']}\n")
                markdown_parts.append(f"\n**Table Data:**\n")
                markdown_parts.append(tbl['markdown_table'])
                markdown_parts.append("\n")
                inserted_tables.add(tbl['id'])
        
        # Add any remaining images not yet inserted
        remaining_images = [img for img in images_data if img['id'] not in inserted_images]
        if remaining_images:
            markdown_parts.append("\n## Additional Images\n")
            for img in remaining_images:
                markdown_parts.append(f"\n### Image: {img['id']}\n")
                markdown_parts.append(f"**Location:** Page {img['page'] + 1}\n")
                markdown_parts.append(f"\n{img['description']}\n")
        
        # Add any remaining tables not yet inserted
        remaining_tables = [tbl for tbl in tables_data if tbl['id'] not in inserted_tables]
        if remaining_tables:
            markdown_parts.append("\n## Additional Tables\n")
            for tbl in remaining_tables:
                markdown_parts.append(f"\n### Table: {tbl['id']}\n")
                if tbl['caption']:
                    markdown_parts.append(f"**Caption:** {tbl['caption']}\n")
                markdown_parts.append(f"**Location:** Page {tbl['page'] + 1}\n")
                markdown_parts.append(f"\n{tbl['description']}\n")
                markdown_parts.append(f"\n**Table Data:**\n")
                markdown_parts.append(tbl['markdown_table'])
                markdown_parts.append("\n")
        
        return '\n'.join(markdown_parts)
    
    def _extract_metadata(self, result: Any, doc_id: str) -> Dict[str, Any]:
        """Extract document metadata."""
        metadata = {
            'doc_id': doc_id,
            'num_pages': len(result.pages) if hasattr(result, 'pages') else 0
        }
        
        doc = getattr(result, 'document', None)
        if doc:
            title = getattr(doc, 'title', None)
            if title:
                metadata['title'] = title
        
        return metadata
