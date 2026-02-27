"""
LangGraph Agent for Markdown Extraction from PDFs.

This agent orchestrates the extraction of high-quality Markdown from PDF documents
using a multi-step workflow:
1. Extract document structure with Docling
2. Analyze images in parallel with vision LLM
3. Analyze tables in parallel with LLM
4. Assemble final Markdown document

Uses LangGraph for robust, observable, and maintainable workflow orchestration.
"""

import logging
import asyncio
import base64
import io
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum

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
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExtractionPhase(str, Enum):
    """Phases of the extraction workflow."""
    INIT = "init"
    EXTRACT_STRUCTURE = "extract_structure"
    ANALYZE_IMAGES = "analyze_images"
    ANALYZE_TABLES = "analyze_tables"
    ASSEMBLE_MARKDOWN = "assemble_markdown"
    COMPLETE = "complete"
    ERROR = "error"


# Reducer functions for state management
def _keep_last(left: Any, right: Any) -> Any:
    """Keep the right value (last written)."""
    return right

def _concat_lists(left: List, right: List) -> List:
    """Concatenate two lists."""
    return left + right

def _merge_dicts(left: Dict, right: Dict) -> Dict:
    """Merge two dictionaries (right values override left)."""
    result = left.copy()
    result.update(right)
    return result


class MarkdownExtractionState(TypedDict):
    """State for the Markdown extraction agent."""
    # Input - these are set once and read by parallel nodes
    file_path: Annotated[str, _keep_last]
    doc_id: Annotated[str, _keep_last]
    llm: Annotated[Any, _keep_last]  # LLM instance
    include_images: Annotated[bool, _keep_last]
    include_tables: Annotated[bool, _keep_last]
    include_formulas: Annotated[bool, _keep_last]
    
    # Processing state
    phase: Annotated[str, _keep_last]
    error_message: Annotated[Optional[str], _keep_last]
    
    # Extracted data - populated by extract_structure node
    docling_result: Annotated[Optional[Any], _keep_last]
    text_elements: Annotated[List[Dict[str, Any]], _concat_lists]
    images: Annotated[List[Dict[str, Any]], _concat_lists]
    tables: Annotated[List[Dict[str, Any]], _concat_lists]
    formulas: Annotated[List[Dict[str, Any]], _concat_lists]
    
    # Processed data - populated by parallel analysis nodes
    image_descriptions: Annotated[Dict[str, str], _merge_dicts]  # image_id -> description
    table_descriptions: Annotated[Dict[str, str], _merge_dicts]  # table_id -> description
    
    # Output - populated by assemble_markdown node
    markdown_content: Annotated[str, _keep_last]
    metadata: Annotated[Dict[str, Any], _merge_dicts]


@dataclass
class MarkdownElement:
    """Represents an element in the Markdown document."""
    element_type: str  # 'text', 'heading', 'table', 'image', 'formula', 'list'
    content: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    order: int = 0


class MarkdownExtractionAgent:
    """
    LangGraph agent for extracting enhanced Markdown from PDF documents.
    
    Workflow:
    1. extract_structure: Extract document structure using Docling
    2. analyze_images: Analyze images with vision LLM (parallel)
    3. analyze_tables: Analyze tables with LLM (parallel)
    4. assemble_markdown: Combine everything into final Markdown
    """
    
    def __init__(self):
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph is required for MarkdownExtractionAgent")
        
        if not DOCLING_AVAILABLE:
            raise RuntimeError("Docling is required for MarkdownExtractionAgent")
        
        self.graph = self._build_graph()
        logger.info("MarkdownExtractionAgent initialized with LangGraph workflow")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create state graph
        workflow = StateGraph(MarkdownExtractionState)
        
        # Add nodes
        workflow.add_node("extract_structure", self._extract_structure_node)
        workflow.add_node("analyze_images", self._analyze_images_node)
        workflow.add_node("analyze_tables", self._analyze_tables_node)
        workflow.add_node("assemble_markdown", self._assemble_markdown_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define edges
        workflow.add_edge(START, "extract_structure")
        
        # After structure extraction, analyze images and tables in parallel
        workflow.add_edge("extract_structure", "analyze_images")
        workflow.add_edge("extract_structure", "analyze_tables")
        
        # After both analyses are complete, assemble markdown
        workflow.add_edge("analyze_images", "assemble_markdown")
        workflow.add_edge("analyze_tables", "assemble_markdown")
        
        # Complete
        workflow.add_edge("assemble_markdown", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
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
        Extract Markdown from PDF using the LangGraph agent.
        
        Args:
            file_path: Path to the PDF file
            doc_id: Document identifier
            llm: LLM instance for generating descriptions
            include_images: Whether to include image descriptions
            include_tables: Whether to include table descriptions
            include_formulas: Whether to include formula extraction
            
        Returns:
            Dict containing markdown content, metadata, and processing info
        """
        # Initialize state
        initial_state: MarkdownExtractionState = {
            "file_path": file_path,
            "doc_id": doc_id,
            "llm": llm,
            "include_images": include_images,
            "include_tables": include_tables,
            "include_formulas": include_formulas,
            "phase": ExtractionPhase.INIT,
            "error_message": None,
            "docling_result": None,
            "text_elements": [],
            "images": [],
            "tables": [],
            "formulas": [],
            "image_descriptions": {},
            "table_descriptions": {},
            "markdown_content": "",
            "metadata": {}
        }
        
        try:
            # Execute the graph
            logger.info(f"Starting LangGraph extraction for document {doc_id}")
            final_state = await self.graph.ainvoke(initial_state)
            
            if final_state.get("error_message"):
                logger.error(f"Extraction failed: {final_state['error_message']}")
                raise RuntimeError(final_state["error_message"])
            
            logger.info(f"Completed LangGraph extraction for document {doc_id}")
            
            return {
                "markdown": final_state["markdown_content"],
                "metadata": final_state["metadata"],
                "images": final_state["images"],
                "tables": final_state["tables"],
                "image_descriptions": final_state["image_descriptions"],
                "table_descriptions": final_state["table_descriptions"]
            }
            
        except Exception as e:
            logger.error(f"LangGraph extraction failed: {e}", exc_info=True)
            raise
    
    async def _extract_structure_node(self, state: MarkdownExtractionState) -> MarkdownExtractionState:
        """Node: Extract document structure using Docling."""
        logger.info(f"[Node: extract_structure] Processing {state['doc_id']}")
        
        try:
            # Initialize Docling
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Convert document
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                converter.convert, 
                state["file_path"]
            )
            
            # Extract elements
            text_elements, images, tables, formulas = self._parse_docling_result(
                result, 
                state["doc_id"]
            )
            
            # Update state
            state["docling_result"] = result
            state["text_elements"] = text_elements
            state["images"] = images
            state["tables"] = tables
            state["formulas"] = formulas
            state["phase"] = ExtractionPhase.EXTRACT_STRUCTURE
            state["metadata"] = self._extract_metadata(result, state["doc_id"])
            
            logger.info(f"[Node: extract_structure] Extracted {len(text_elements)} texts, "
                       f"{len(images)} images, {len(tables)} tables")
            
        except Exception as e:
            logger.error(f"[Node: extract_structure] Error: {e}")
            state["error_message"] = f"Structure extraction failed: {str(e)}"
            state["phase"] = ExtractionPhase.ERROR
        
        return state
    
    async def _analyze_images_node(self, state: MarkdownExtractionState) -> MarkdownExtractionState:
        """Node: Analyze images with vision LLM."""
        logger.info(f"[Node: analyze_images] Processing {len(state['images'])} images")
        
        if not state["include_images"] or not state["images"] or not state["llm"]:
            logger.info("[Node: analyze_images] Skipping image analysis")
            state["phase"] = ExtractionPhase.ANALYZE_IMAGES
            return state
        
        try:
            # Process images in parallel
            tasks = [
                self._analyze_single_image(img, state["llm"])
                for img in state["images"]
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store descriptions
            for img, result in zip(state["images"], results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to analyze image {img['id']}: {result}")
                    state["image_descriptions"][img["id"]] = (
                        f"Image on page {img['page'] + 1} (analysis failed)"
                    )
                else:
                    state["image_descriptions"][img["id"]] = result
                    # Update image data with description
                    img["description"] = result
            
            state["phase"] = ExtractionPhase.ANALYZE_IMAGES
            logger.info(f"[Node: analyze_images] Analyzed {len(state['images'])} images")
            
        except Exception as e:
            logger.error(f"[Node: analyze_images] Error: {e}")
            # Don't fail the whole workflow, just log the error
            state["phase"] = ExtractionPhase.ANALYZE_IMAGES
        
        return state
    
    async def _analyze_tables_node(self, state: MarkdownExtractionState) -> MarkdownExtractionState:
        """Node: Analyze tables with LLM."""
        logger.info(f"[Node: analyze_tables] Processing {len(state['tables'])} tables")
        
        if not state["include_tables"] or not state["tables"] or not state["llm"]:
            logger.info("[Node: analyze_tables] Skipping table analysis")
            state["phase"] = ExtractionPhase.ANALYZE_TABLES
            return state
        
        try:
            # Process tables in parallel
            tasks = [
                self._analyze_single_table(table, state["llm"])
                for table in state["tables"]
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store descriptions
            for table, result in zip(state["tables"], results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to analyze table {table['id']}: {result}")
                    state["table_descriptions"][table["id"]] = (
                        f"Table on page {table['page'] + 1} with "
                        f"{table.get('shape', [0, 0])[0]} rows "
                        f"(analysis failed)"
                    )
                else:
                    state["table_descriptions"][table["id"]] = result
                    # Update table data with description
                    table["description"] = result
            
            state["phase"] = ExtractionPhase.ANALYZE_TABLES
            logger.info(f"[Node: analyze_tables] Analyzed {len(state['tables'])} tables")
            
        except Exception as e:
            logger.error(f"[Node: analyze_tables] Error: {e}")
            # Don't fail the whole workflow, just log the error
            state["phase"] = ExtractionPhase.ANALYZE_TABLES
        
        return state
    
    async def _assemble_markdown_node(self, state: MarkdownExtractionState) -> MarkdownExtractionState:
        """Node: Assemble final Markdown document."""
        logger.info(f"[Node: assemble_markdown] Assembling final document")
        
        try:
            # Build Markdown content
            markdown_parts = []
            
            # Document header
            markdown_parts.append(f"# Document: {state['doc_id']}\n")
            
            # Add metadata
            metadata = state["metadata"]
            if metadata.get("title"):
                markdown_parts.append(f"**Title:** {metadata['title']}\n")
            markdown_parts.append(f"**Document ID:** {state['doc_id']}\n")
            markdown_parts.append("---\n")
            
            # Build elements by page
            elements_by_page = self._organize_elements_by_page(state)
            
            for page_num in sorted(elements_by_page.keys()):
                page_elements = elements_by_page[page_num]
                
                markdown_parts.append(f"\n## Page {page_num + 1}\n")
                
                for element in page_elements:
                    if element.element_type == "heading":
                        markdown_parts.append(f"\n{element.content}\n")
                    elif element.element_type == "text":
                        markdown_parts.append(f"\n{element.content}\n")
                    elif element.element_type == "list":
                        markdown_parts.append(f"\n{element.content}\n")
                    elif element.element_type == "image":
                        img_id = element.metadata.get("image_id")
                        img_data = next((img for img in state["images"] if img["id"] == img_id), None)
                        if img_data:
                            markdown_parts.append(f"\n### Image: {img_id}\n")
                            markdown_parts.append(f"**Location:** Page {img_data['page'] + 1}\n")
                            markdown_parts.append(f"\n{img_data.get('description', 'No description available')}\n")
                    elif element.element_type == "table":
                        tbl_id = element.metadata.get("table_id")
                        tbl_data = next((tbl for tbl in state["tables"] if tbl["id"] == tbl_id), None)
                        if tbl_data:
                            markdown_parts.append(f"\n### Table: {tbl_id}\n")
                            if tbl_data.get("caption"):
                                markdown_parts.append(f"**Caption:** {tbl_data['caption']}\n")
                            markdown_parts.append(f"**Location:** Page {tbl_data['page'] + 1}\n")
                            markdown_parts.append(f"\n{tbl_data.get('description', 'No description available')}\n")
                            markdown_parts.append(f"\n**Table Data:**\n")
                            markdown_parts.append(tbl_data.get("markdown_table", ""))
                            markdown_parts.append("\n")
            
            # Add remaining images not yet inserted
            inserted_images = set()
            for element in sum(elements_by_page.values(), []):
                if element.element_type == "image":
                    inserted_images.add(element.metadata.get("image_id"))
            
            remaining_images = [
                img for img in state["images"] 
                if img["id"] not in inserted_images
            ]
            
            if remaining_images:
                markdown_parts.append("\n## Additional Images\n")
                for img in remaining_images:
                    markdown_parts.append(f"\n### Image: {img['id']}\n")
                    markdown_parts.append(f"**Location:** Page {img['page'] + 1}\n")
                    markdown_parts.append(f"\n{img.get('description', 'No description available')}\n")
            
            # Add remaining tables not yet inserted
            inserted_tables = set()
            for element in sum(elements_by_page.values(), []):
                if element.element_type == "table":
                    inserted_tables.add(element.metadata.get("table_id"))
            
            remaining_tables = [
                tbl for tbl in state["tables"] 
                if tbl["id"] not in inserted_tables
            ]
            
            if remaining_tables:
                markdown_parts.append("\n## Additional Tables\n")
                for tbl in remaining_tables:
                    markdown_parts.append(f"\n### Table: {tbl['id']}\n")
                    if tbl.get("caption"):
                        markdown_parts.append(f"**Caption:** {tbl['caption']}\n")
                    markdown_parts.append(f"**Location:** Page {tbl['page'] + 1}\n")
                    markdown_parts.append(f"\n{tbl.get('description', 'No description available')}\n")
                    markdown_parts.append(f"\n**Table Data:**\n")
                    markdown_parts.append(tbl.get("markdown_table", ""))
                    markdown_parts.append("\n")
            
            state["markdown_content"] = "\n".join(markdown_parts)
            state["phase"] = ExtractionPhase.COMPLETE
            
            logger.info(f"[Node: assemble_markdown] Assembled {len(state['markdown_content'])} characters")
            
        except Exception as e:
            logger.error(f"[Node: assemble_markdown] Error: {e}")
            state["error_message"] = f"Markdown assembly failed: {str(e)}"
            state["phase"] = ExtractionPhase.ERROR
        
        return state
    
    async def _handle_error_node(self, state: MarkdownExtractionState) -> MarkdownExtractionState:
        """Node: Handle errors gracefully."""
        logger.error(f"[Node: handle_error] Error occurred: {state.get('error_message')}")
        return state
    
    def _parse_docling_result(
        self, 
        result: Any, 
        doc_id: str
    ) -> tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Parse Docling result into structured elements."""
        text_elements = []
        images = []
        tables = []
        formulas = []
        
        doc = getattr(result, 'document', None)
        
        if doc:
            # Extract headings and texts
            headings = getattr(doc, 'headings', [])
            texts = getattr(doc, 'texts', [])
            lists = getattr(doc, 'lists', [])
            
            element_order = 0
            
            # Process headings
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
                
                text_elements.append({
                    'id': f"{doc_id}_heading_{element_order}",
                    'type': 'heading',
                    'text': f"{heading_markers} {text}",
                    'page': page_no - 1,
                    'order': element_order,
                    'level': level
                })
                element_order += 1
            
            # Process texts
            for item in texts:
                text = getattr(item, 'text', '')
                if not text:
                    continue
                
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov:
                    prov = prov[0]
                page_no = getattr(prov, 'page_no', 1)
                
                text_elements.append({
                    'id': f"{doc_id}_text_{element_order}",
                    'type': 'text',
                    'text': text,
                    'page': page_no - 1,
                    'order': element_order
                })
                element_order += 1
            
            # Process pictures (images)
            pictures = getattr(doc, 'pictures', [])
            for idx, item in enumerate(pictures):
                image_data = item.get_image(doc) if hasattr(item, 'get_image') else getattr(item, 'image', None)
                if not image_data:
                    continue
                
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov:
                    prov = prov[0]
                page_no = getattr(prov, 'page_no', 1)
                
                images.append({
                    'id': f"{doc_id}_img_{idx}",
                    'page': page_no - 1,
                    'image_data': image_data,
                    'order': element_order
                })
                element_order += 1
            
            # Process tables
            doc_tables = getattr(doc, 'tables', [])
            for idx, item in enumerate(doc_tables):
                try:
                    df = item.export_to_dataframe(doc) if hasattr(item, 'export_to_dataframe') else pd.DataFrame()
                    if df.empty:
                        continue
                    
                    prov = getattr(item, 'prov', None)
                    if isinstance(prov, list) and prov:
                        prov = prov[0]
                    page_no = getattr(prov, 'page_no', 1)
                    
                    caption = getattr(item, 'caption', None) or ''
                    
                    tables.append({
                        'id': f"{doc_id}_tbl_{idx}",
                        'page': page_no - 1,
                        'dataframe': df,
                        'markdown_table': df.to_markdown(index=False),
                        'caption': caption,
                        'columns': list(df.columns),
                        'shape': df.shape,
                        'order': element_order
                    })
                    element_order += 1
                except Exception as e:
                    logger.error(f"Failed to process table {idx}: {e}")
                    continue
        
        # Sort by order
        text_elements.sort(key=lambda x: x['order'])
        images.sort(key=lambda x: x['order'])
        tables.sort(key=lambda x: x['order'])
        
        return text_elements, images, tables, formulas
    
    async def _analyze_single_image(self, img: Dict[str, Any], llm) -> str:
        """Analyze a single image with vision LLM."""
        try:
            image_data = img.get('image_data')
            if not image_data:
                return f"Image on page {img['page'] + 1} (no data)"
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image_data.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Determine media type
            media_type = "image/png"
            if img_bytes.startswith(b'\x89PNG'):
                media_type = "image/png"
            elif img_bytes.startswith(b'\xff\xd8\xff'):
                media_type = "image/jpeg"
            
            # System prompt
            system_prompt = """You are an expert document analyst. Describe images from documents with rich detail.

Focus on:
1. Type of visual element (chart, graph, diagram, photograph, screenshot, etc.)
2. Key visual elements and their relationships
3. Any text, labels, numbers, or data visible in the image
4. Colors, patterns, or visual encoding used
5. The purpose or message the image conveys
6. Any trends, comparisons, or insights visible

Be specific and detailed. Format your response as a well-structured paragraph."""
            
            # Build message
            llm_class_name = llm.__class__.__name__
            uses_openai_format = "OpenAI" in llm_class_name or "Google" in llm_class_name
            
            message_content = []
            message_content.append({
                "type": "text",
                "text": f"Analyze this image from page {img['page'] + 1} of the document. Provide a comprehensive description."
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
            logger.error(f"Error analyzing image: {e}")
            return f"Image on page {img['page'] + 1} (analysis failed)"
    
    async def _analyze_single_table(self, table: Dict[str, Any], llm) -> str:
        """Analyze a single table with LLM."""
        try:
            df = table.get('dataframe')
            if df is None or df.empty:
                return f"Table on page {table['page'] + 1} (empty)"
            
            stats = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
            
            sample_rows = df.head(3).to_dict('records')
            caption = table.get('caption', '')
            
            human_prompt = f"""Analyze this table from page {table['page'] + 1} of a document and provide a comprehensive semantic description.

{caption if caption else 'No caption provided.'}

Table Structure:
- Rows: {stats['shape'][0]}, Columns: {stats['shape'][1]}
- Column names: {', '.join(stats['columns'])}
- Data types: {stats['dtypes']}

Sample Data (first 3 rows):
{sample_rows}

Provide a rich description covering:
1. What kind of data this table contains and its purpose
2. The main categories or metrics represented
3. Relationships between columns (if any are apparent)
4. What insights or questions this table could answer
5. Any notable patterns, ranges, or data characteristics

Format as a well-structured paragraph."""
            
            system_prompt = """You are an expert data analyst. Describe tables from documents with semantic richness.

Focus on understanding what the data represents, not just technical details. Explain the meaning and significance of the table in the context of a document."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            description = response.content if hasattr(response, 'content') else str(response)
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Error analyzing table: {e}")
            return f"Table with {len(df)} rows and {len(df.columns)} columns on page {table['page'] + 1}"
    
    def _organize_elements_by_page(
        self, 
        state: MarkdownExtractionState
    ) -> Dict[int, List[MarkdownElement]]:
        """Organize all elements by page for sequential output."""
        elements_by_page: Dict[int, List[MarkdownElement]] = {}
        
        # Add text elements
        for elem in state["text_elements"]:
            page = elem.get('page', 0)
            if page not in elements_by_page:
                elements_by_page[page] = []
            
            elem_type = elem.get('type', 'text')
            if elem_type == 'heading':
                elements_by_page[page].append(MarkdownElement(
                    element_type='heading',
                    content=elem['text'],
                    page_number=page,
                    order=elem.get('order', 0)
                ))
            else:
                elements_by_page[page].append(MarkdownElement(
                    element_type='text',
                    content=elem['text'],
                    page_number=page,
                    order=elem.get('order', 0)
                ))
        
        # Add images
        for img in state["images"]:
            page = img.get('page', 0)
            if page not in elements_by_page:
                elements_by_page[page] = []
            
            elements_by_page[page].append(MarkdownElement(
                element_type='image',
                content='',
                page_number=page,
                metadata={'image_id': img['id']},
                order=img.get('order', 0)
            ))
        
        # Add tables
        for tbl in state["tables"]:
            page = tbl.get('page', 0)
            if page not in elements_by_page:
                elements_by_page[page] = []
            
            elements_by_page[page].append(MarkdownElement(
                element_type='table',
                content='',
                page_number=page,
                metadata={'table_id': tbl['id']},
                order=tbl.get('order', 0)
            ))
        
        # Sort each page by order
        for page in elements_by_page:
            elements_by_page[page].sort(key=lambda x: x.order)
        
        return elements_by_page
    
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


# Convenience function for external use
async def extract_markdown_with_agent(
    file_path: str,
    doc_id: str,
    llm=None,
    include_images: bool = True,
    include_tables: bool = True,
    include_formulas: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to extract Markdown using the LangGraph agent.
    
    Args:
        file_path: Path to the PDF file
        doc_id: Document identifier
        llm: LLM instance for generating descriptions
        include_images: Whether to include image descriptions
        include_tables: Whether to include table descriptions
        include_formulas: Whether to include formula extraction
        
    Returns:
        Dict containing markdown content and metadata
    """
    agent = MarkdownExtractionAgent()
    return await agent.extract_markdown(
        file_path=file_path,
        doc_id=doc_id,
        llm=llm,
        include_images=include_images,
        include_tables=include_tables,
        include_formulas=include_formulas
    )
