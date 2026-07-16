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
from enum import Enum

# Silence noisy RapidOCR logs
logging.getLogger("RapidOCR").setLevel(logging.ERROR)

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
from tilellm.shared.llm_utils import extract_llm_text
from tilellm.modules.pdf_ocr.services.converter_registry import (
    ConverterResult,
    register_converter,
)

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


# Placeholder Docling inserts at page boundaries when serializing a segment;
# split_segment_pages turns it back into per-page bodies with global numbers.
PAGE_BREAK = "<!-- PAGE_BREAK -->"


def split_segment_pages(markdown: str, page_offset: int, page_break: str = PAGE_BREAK):
    """Split one segment's native markdown into (global_page_no, body) pairs.

    page_offset is the 0-based index of the segment's first page in the
    original document; page numbers returned are 1-based to match the
    "## Page N" contract the MarkdownChunker relies on.
    """
    return [
        (page_offset + i + 1, chunk.strip())
        for i, chunk in enumerate(markdown.split(page_break))
    ]


def assemble_markdown(doc_id, page_bodies, image_notes, table_notes) -> str:
    """Assemble the final document from native per-page markdown bodies.

    page_bodies: list of (page_no, markdown) already in reading order.
    image_notes / table_notes: list of (page_no, description) from the LLM
    enrichment; appended as trailing sections (Docling leaves image
    placeholders and renders tables but never the semantic description).
    """
    parts = [f"# Document: {doc_id}\n"]
    for page_no, body in page_bodies:
        parts.append(f"\n## Page {page_no}\n")
        parts.append(body)
    if image_notes:
        parts.append("\n## Image descriptions\n")
        for page_no, desc in image_notes:
            parts.append(f"\n**Image (page {page_no}):** {desc}\n")
    if table_notes:
        parts.append("\n## Table descriptions\n")
        for page_no, desc in table_notes:
            parts.append(f"\n**Table (page {page_no}):** {desc}\n")
    return "\n".join(parts)


def _segment_to_markdown(document) -> str:
    """Serialize one Docling segment to markdown, page breaks marked.

    Never raises: a serializer failure degrades to an empty body for that
    segment rather than killing the whole extraction.
    """
    try:
        return document.export_to_markdown(page_break_placeholder=PAGE_BREAK)
    except Exception as e:
        logger.warning(f"export_to_markdown failed for a segment, skipping body: {e}")
        return ""


def _native_to_page_bodies(text_elements: List[Dict], tables: List[Dict]):
    """Build (page_no, body) pairs from native (PyMuPDF) elements.

    The degraded native level has no DoclingDocument to serialize, so we
    stitch plain text + rendered tables per page. page numbers are 1-based.
    """
    from collections import defaultdict

    by_page: Dict[int, List[str]] = defaultdict(list)
    for el in text_elements:
        text = el.get("text", "")
        if text:
            by_page[el.get("page", 0)].append(text)
    for tbl in tables:
        md_table = tbl.get("markdown_table")
        if md_table:
            by_page[tbl.get("page", 0)].append(md_table)
    return [(p + 1, "\n\n".join(by_page[p])) for p in sorted(by_page)]


class MarkdownExtractionState(TypedDict):
    """State for the Markdown extraction agent."""
    # Input - these are set once and read by parallel nodes
    file_path: Annotated[str, _keep_last]
    doc_id: Annotated[str, _keep_last]
    llm: Annotated[Any, _keep_last]  # LLM instance
    include_images: Annotated[bool, _keep_last]
    include_tables: Annotated[bool, _keep_last]
    include_formulas: Annotated[bool, _keep_last]
    attempt: Annotated[int, _keep_last]  # 1-based; drives the conversion degradation ladder
    converter: Annotated[str, _keep_last]  # registry name of the PDF converter
    skip_ocr: Annotated[bool, _keep_last]  # converter hint: skip OCR (native-digital PDFs)
    converter_options: Annotated[Optional[Dict[str, Any]], _keep_last]  # per-request converter config

    # Processing state
    phase: Annotated[str, _keep_last]
    error_message: Annotated[Optional[str], _keep_last]
    
    # Extracted data - populated by extract_structure node
    docling_result: Annotated[Optional[Any], _keep_last]
    text_elements: Annotated[List[Dict[str, Any]], _keep_last]
    images: Annotated[List[Dict[str, Any]], _keep_last]
    tables: Annotated[List[Dict[str, Any]], _keep_last]
    formulas: Annotated[List[Dict[str, Any]], _keep_last]
    
    # Processed data - populated by parallel analysis nodes
    image_descriptions: Annotated[Dict[str, str], _merge_dicts]  # image_id -> description
    table_descriptions: Annotated[Dict[str, str], _merge_dicts]  # table_id -> description
    
    # Native per-page markdown bodies (page_no, body) from Docling's own
    # serializer — populated by extract_structure, consumed by assemble.
    page_bodies: Annotated[List[Any], _keep_last]

    # Output - populated by assemble_markdown node
    markdown_content: Annotated[str, _keep_last]
    metadata: Annotated[Dict[str, Any], _merge_dicts]


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
        
        # Define edges — sequential to keep state consistent.
        # Image and table analysis each already parallelize internally via asyncio.gather.
        workflow.add_edge(START, "extract_structure")
        workflow.add_edge("extract_structure", "analyze_images")
        workflow.add_edge("analyze_images", "analyze_tables")
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
        include_formulas: bool = True,
        attempt: int = 1,
        converter: str = "docling",
        skip_ocr: bool = False,
        converter_options: Optional[Dict[str, Any]] = None,
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
            attempt: 1-based attempt number; higher attempts select
                progressively cheaper conversion strategies (degradation ladder)
            converter: registry name of the PDF converter to use (default docling)
            skip_ocr: converter hint to skip OCR (native-digital PDFs)
            converter_options: per-request config merged into the converter's
                options (e.g. LightOnOCR endpoint_url/api_key/model)

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
            "attempt": attempt,
            "converter": converter,
            "skip_ocr": skip_ocr,
            "converter_options": converter_options,
            "phase": ExtractionPhase.INIT,
            "error_message": None,
            "docling_result": None,
            "text_elements": [],
            "images": [],
            "tables": [],
            "formulas": [],
            "image_descriptions": {},
            "table_descriptions": {},
            "page_bodies": [],
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
        """Node: convert the PDF via the selected pluggable converter.

        The converter (default "docling") returns engine-agnostic per-page
        markdown plus images/tables for downstream LLM enrichment. Swapping in
        MinerU/LightOnOCR/... is a registry entry, not a change here.
        """
        converter_name = state.get("converter") or "docling"
        logger.info(f"[Node: extract_structure] Processing {state['doc_id']} "
                    f"(attempt {state.get('attempt', 1)}, converter={converter_name})")

        try:
            from tilellm.modules.pdf_ocr.services.converter_registry import get_converter

            result = await get_converter(converter_name)(
                state["file_path"],
                state["doc_id"],
                attempt=state.get("attempt", 1),
                options={
                    "skip_ocr": state.get("skip_ocr", False),
                    **(state.get("converter_options") or {}),
                },
            )

            # The raw conversion objects stay out of agent state.
            state["docling_result"] = None
            state["text_elements"] = result.text_elements
            state["images"] = result.images
            state["tables"] = result.tables
            state["formulas"] = result.formulas
            state["page_bodies"] = result.page_bodies
            state["phase"] = ExtractionPhase.EXTRACT_STRUCTURE
            state["metadata"] = {
                "doc_id": state["doc_id"],
                "num_pages": result.num_pages,
                "extraction_quality": result.extraction_quality,
            }

            logger.info(f"[Node: extract_structure] Extracted {len(result.text_elements)} texts, "
                        f"{len(result.images)} images, {len(result.tables)} tables "
                        f"(quality={result.extraction_quality})")

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
        """Node: Assemble the final document from Docling's native per-page markdown.

        The structural body comes verbatim from Docling's own serializer
        (lists, inline formatting, tables, reading order). The LLM image/table
        descriptions — which Docling does not produce — are appended as
        trailing sections keyed by page.
        """
        logger.info("[Node: assemble_markdown] Assembling final document")

        try:
            img_desc = state.get("image_descriptions", {})
            tbl_desc = state.get("table_descriptions", {})
            image_notes = [
                (img.get("page", 0) + 1, img_desc[img["id"]])
                for img in state["images"]
                if img.get("id") in img_desc
            ]
            table_notes = [
                (tbl.get("page", 0) + 1, tbl_desc[tbl["id"]])
                for tbl in state["tables"]
                if tbl.get("id") in tbl_desc
            ]
            state["markdown_content"] = assemble_markdown(
                state["doc_id"],
                state.get("page_bodies", []),
                image_notes,
                table_notes,
            )
            state["phase"] = ExtractionPhase.COMPLETE

            logger.info(
                f"[Node: assemble_markdown] Assembled {len(state['markdown_content'])} characters"
            )

        except Exception as e:
            logger.error(f"[Node: assemble_markdown] Error: {e}")
            state["error_message"] = f"Markdown assembly failed: {str(e)}"
            state["phase"] = ExtractionPhase.ERROR

        return state

    async def _handle_error_node(self, state: MarkdownExtractionState) -> MarkdownExtractionState:
        """Node: Handle errors gracefully."""
        logger.error(f"[Node: handle_error] Error occurred: {state.get('error_message')}")
        return state
    
    @staticmethod
    def _parse_docling_result(
        result: Any,
        doc_id: str,
        page_offset: int = 0
    ) -> tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Parse Docling result into structured elements.

        Accepts either a ConversionResult (legacy) or a DoclingDocument directly.
        page_offset shifts page numbers when the document is a segment of a
        larger PDF (segmented conversion).
        """
        text_elements = []
        images = []
        tables = []
        formulas = []

        # ConversionResult has .document; a DoclingDocument is used as-is
        doc = getattr(result, 'document', None) or result

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
                    'page': page_no - 1 + page_offset,
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
                    'page': page_no - 1 + page_offset,
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
                    'page': page_no - 1 + page_offset,
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
                        'page': page_no - 1 + page_offset,
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

    @staticmethod
    def _native_result_to_elements(
        native: Dict[str, Any],
        doc_id: str
    ) -> tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Adapt a process_pdf_native result to the agent element shapes.

        Used by the last rung of the degradation ladder: full content coverage,
        no images, tables extracted by PyMuPDF heuristics.
        """
        text_elements: List[Dict] = []
        tables: List[Dict] = []
        order = 0

        for el in native.get("text_elements", []):
            text = el.get("text", "")
            if not text:
                continue
            text_elements.append({
                "id": el.get("id") or f"{doc_id}_text_{order}",
                "type": "text",
                "text": text,
                "page": el.get("page", 0),
                "order": order,
            })
            order += 1

        for idx, tbl in enumerate(native.get("tables", [])):
            df = tbl.get("data")
            if df is None or df.empty:
                continue
            tables.append({
                "id": tbl.get("id") or f"{doc_id}_tbl_{idx}",
                "page": tbl.get("page", 0),
                "dataframe": df,
                "markdown_table": df.to_markdown(index=False),
                "caption": tbl.get("caption") or "",
                "columns": list(df.columns),
                "shape": df.shape,
                "order": order,
            })
            order += 1

        return text_elements, [], tables, []

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
            description = extract_llm_text(response)
            
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
                'columns': [str(c) for c in df.columns.tolist()],
                'dtypes': df.dtypes.astype(str).to_dict()
            }
            
            if df.columns.duplicated().any():
                seen = {}
                new_cols = []
                for col in df.columns:
                    if col in seen:
                        seen[col] += 1
                        new_cols.append(f"{col}_{seen[col]}")
                    else:
                        seen[col] = 0
                        new_cols.append(col)
                df = df.copy()
                df.columns = new_cols
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
            description = extract_llm_text(response)
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Error analyzing table: {e}")
            return f"Table with {len(df)} rows and {len(df.columns)} columns on page {table['page'] + 1}"
    
async def docling_convert(file_path, doc_id, attempt: int = 1, options=None) -> ConverterResult:
    """Default PDF converter: Docling via the memory-bounded conversion pipeline.

    Produces engine-agnostic per-page native markdown plus images/tables for
    downstream LLM enrichment. Segmented heavy documents are merged with
    corrected page numbers and reading order. options={"skip_ocr": True}
    forces Docling to skip OCR (use for native-digital PDFs — faster and, on a
    clean text layer, more accurate than OCR-ing it).
    """
    from tilellm.modules.pdf_ocr.services.conversion_pipeline import run_conversion

    skip_ocr = bool(options and options.get("skip_ocr"))
    outcome = await run_conversion(
        file_path, doc_id, attempt=attempt,
        do_ocr_override=False if skip_ocr else None,
    )

    page_bodies: List[Any] = []
    if outcome.native_result is not None:
        # Degraded native level: PyMuPDF raw extraction, full coverage.
        text_elements, images, tables, formulas = (
            MarkdownExtractionAgent._native_result_to_elements(outcome.native_result, doc_id)
        )
        num_pages = outcome.native_result.get("metadata", {}).get(
            "num_pages", outcome.profile.num_pages
        )
        page_bodies = _native_to_page_bodies(text_elements, tables)
    else:
        # Docling levels: merge per-segment parses with corrected pages/order.
        text_elements, images, tables, formulas = [], [], [], []
        order_offset = 0
        for seg_idx, seg in enumerate(outcome.segments):
            page_bodies.extend(
                split_segment_pages(_segment_to_markdown(seg.document), seg.page_offset)
            )
            seg_texts, seg_imgs, seg_tbls, seg_forms = (
                MarkdownExtractionAgent._parse_docling_result(
                    seg.document, doc_id, page_offset=seg.page_offset
                )
            )
            seg_count = len(seg_texts) + len(seg_imgs) + len(seg_tbls)
            for coll in (seg_texts, seg_imgs, seg_tbls, seg_forms):
                for el in coll:
                    el["order"] = el.get("order", 0) + order_offset
                    if seg_idx > 0:
                        el["id"] = f"{el['id']}_s{seg_idx}"
            order_offset += seg_count
            text_elements.extend(seg_texts)
            images.extend(seg_imgs)
            tables.extend(seg_tbls)
            formulas.extend(seg_forms)
        num_pages = outcome.profile.num_pages

    return ConverterResult(
        page_bodies=page_bodies,
        images=images,
        tables=tables,
        text_elements=text_elements,
        formulas=formulas,
        num_pages=num_pages,
        extraction_quality=outcome.extraction_quality,
    )


register_converter("docling", docling_convert)


# Convenience function for external use
async def extract_markdown_with_agent(
    file_path: str,
    doc_id: str,
    llm=None,
    include_images: bool = True,
    include_tables: bool = True,
    include_formulas: bool = True,
    attempt: int = 1
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
        include_formulas=include_formulas,
        attempt=attempt
    )
