"""
Business Logic for PDF OCR Module.
Handles service initialization, dependency injection, and core operations.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
import asyncio

from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async, inject_llm_async
from tilellm.shared.llm_utils import extract_llm_text
from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest
from tilellm.tools.document_tools import _extract_file_name
from tilellm.models.chunk_metadata import CommonChunkMetadata
from tilellm.shared.situated_context import enrich_chunks_with_situated_context
from .services.docling_processor import get_or_create_processor, DocumentType
from .services.pdf_native_processor import process_pdf_native
from .services.pdf_classifier import classify_pdf
from .services.pdf_entity_extractor import PDFEntityExtractor
from .services.context_aware_chunker import ContextAwareChunker
from .services.table_semantic_linker import TableSemanticLinker
from .services.image_semantic_linker import ImageSemanticLinker
from .services.markdown_extraction_agent import MarkdownExtractionAgent
from .services.markdown_chunker import MarkdownChunker
from ...models.llm import TEIConfig

try:
    from tilellm.modules.knowledge_graph.repository.repository import GraphRepository
    from tilellm.modules.knowledge_graph.models import Node, Relationship
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False
    GraphRepository = None
    Node = None
    Relationship = None

logger = logging.getLogger(__name__)


def _normalize_date_str(val: str) -> str:
    """Convert DD/MM/YYYY to ISO YYYY-MM-DD; return unchanged for other formats."""
    import re as _re
    if isinstance(val, str) and _re.match(r'^\d{2}/\d{2}/\d{4}$', val):
        d, m, y = val.split('/')
        return f"{y}-{m}-{d}"
    return val


def _apply_additional_metadata(base: dict, additional) -> dict:
    """Merge additional_metadata dict into base, normalizing the 'date' key if present."""
    if not additional:
        return base
    for k, v in additional.items():
        if k == 'date' and isinstance(v, str):
            v = _normalize_date_str(v)
        base[k] = v
    return base


def _sc_needs_metadata_extraction(sc_config) -> bool:
    """Return True if the situated context config will extract structured metadata.

    This happens when either:
    - metadata_extraction_prompt is set explicitly, OR
    - the YAML profile has json_mode: true (e.g. pa_italiana)

    When metadata extraction is active, SC must run on ALL chunks — including
    tables with semantic descriptions and images with rich captions — because
    the metadata fields (act_type, topics, amount…) are not produced by
    TableSemanticLinker or image captioning.
    """
    if not sc_config or not sc_config.enable:
        return False
    if sc_config.metadata_extraction_prompt:
        return True
    if sc_config.profile:
        from tilellm.shared.situated_context import _load_profile_data
        data = _load_profile_data(sc_config.profile)
        return bool(data and data.get("json_mode", False))
    return False


def _bbox_center(bbox):
    if not bbox or len(bbox) < 4:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_distance(b1, b2):
    c1 = _bbox_center(b1)
    c2 = _bbox_center(b2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _compute_cross_modal_refs(text_elements: list, tables: list, images: list, max_surrounding_chars: int = 300):
    """
    Compute cross-modal references between text, table, and image elements.

    For each table/image:
      - sets 'surrounding_text': concatenated text of the closest text elements on the same page
      - sets 'ref_text_ids': IDs of those text elements

    For each text element:
      - sets 'ref_tables': list of table IDs on the same page sorted by bbox proximity
      - sets 'ref_images': list of image IDs on the same page sorted by bbox proximity
    """
    # Build page-keyed indexes
    text_by_page: Dict[int, list] = {}
    for el in text_elements:
        p = el.get('page', 0)
        text_by_page.setdefault(p, []).append(el)

    table_by_page: Dict[int, list] = {}
    for t in tables:
        p = t.get('page', 0)
        table_by_page.setdefault(p, []).append(t)

    image_by_page: Dict[int, list] = {}
    for img in images:
        p = img.get('page', 0)
        image_by_page.setdefault(p, []).append(img)

    # Surrounding text for tables
    for table in tables:
        page = table.get('page', 0)
        bbox = table.get('bbox')
        page_texts = text_by_page.get(page, [])
        sorted_texts = sorted(page_texts, key=lambda el: _bbox_distance(el.get('bbox'), bbox))
        surrounding = ' '.join(el.get('text', '') for el in sorted_texts[:3] if el.get('text'))
        table['surrounding_text'] = surrounding[:max_surrounding_chars]
        table['ref_text_ids'] = [el['id'] for el in sorted_texts[:3] if 'id' in el]

    # Surrounding text for images
    for image in images:
        page = image.get('page', 0)
        bbox = image.get('bbox')
        page_texts = text_by_page.get(page, [])
        sorted_texts = sorted(page_texts, key=lambda el: _bbox_distance(el.get('bbox'), bbox))
        surrounding = ' '.join(el.get('text', '') for el in sorted_texts[:3] if el.get('text'))
        image['surrounding_text'] = surrounding[:max_surrounding_chars]
        image['ref_text_ids'] = [el['id'] for el in sorted_texts[:3] if 'id' in el]

    # ref_tables / ref_images for text elements
    for el in text_elements:
        page = el.get('page', 0)
        bbox = el.get('bbox')
        page_tables = table_by_page.get(page, [])
        page_images = image_by_page.get(page, [])
        sorted_tables = sorted(page_tables, key=lambda t: _bbox_distance(t.get('bbox'), bbox))
        sorted_images = sorted(page_images, key=lambda i: _bbox_distance(i.get('bbox'), bbox))
        # Keep top-3 closest on the same page
        el['ref_tables'] = [t['id'] for t in sorted_tables[:3] if 'id' in t]
        el['ref_images'] = [i['id'] for i in sorted_images[:3] if 'id' in i]


async def _invoke_llm_chat(
    system_prompt: str,
    human_prompt: str,
    llm
) -> str:
    """
    Helper for invoking LLM chat with system and human messages.
    
    Args:
        system_prompt: System message content
        human_prompt: Human message content  
        llm: Injected LLM instance
    
    Returns:
        Response content as string
    """
    if llm is None:
        raise ValueError("LLM instance is required")
    
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    
    try:
        response = await llm.ainvoke(messages)
        return extract_llm_text(response)
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        raise


@inject_llm_chat_async
@inject_repo_async
async def process_pdf_document_with_embeddings(
    question: PDFScrapingRequest,
    bucket_name: Optional[str] = None,
    object_name: Optional[str] = None,
    file_path: Optional[str] = None,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process PDF document using Docling and add embeddings to vector store.
    
    If extract_md_simple=True, uses the LangGraph agent for Markdown extraction
    and structure-aware chunking instead of the default element-based processing.
    
    Args:
        question: PDFScrapingRequest with document configuration
        repo: Injected vector store repository
        llm: Injected LLM instance for captioning/description generation
        llm_embeddings: Injected embeddings model
        bucket_name: Optional MinIO bucket name
        object_name: Optional MinIO object name
        file_path: Optional local file path
    
    Returns:
        Dict with processing results including metadata and statistics
    """
    if repo is None:
        raise RuntimeError("Vector store repository not injected")
    
    if llm is None:
        raise ValueError("LLM configuration is required for image captioning and table descriptions")
    
    if question.engine is None:
        raise ValueError("Engine configuration is required for vector store access")
    
    # If extract_md_simple is enabled, use the LangGraph agent approach
    if getattr(question, 'extract_md_simple', False):
        logger.info(f"Using LangGraph Markdown extraction for document {question.id}")
        return await process_pdf_markdown_extraction(
            question=question,
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path,
            repo=repo,
            llm=llm,
            llm_embeddings=llm_embeddings,
            **kwargs
        )
    
    # Retrieve the module-level singleton processor (Docling models loaded once at first call).
    processor = await get_or_create_processor()

    strategy = getattr(question, 'strategy', None)

    try:
        # Process document from MinIO, file path, or direct content
        result = None
        temp_file_path = None

        if bucket_name and object_name:
            # MinIO path: always Docling (fast path not applicable without local file)
            result = await processor.process_from_minio(
                bucket_name=bucket_name,
                object_name=object_name,
                doc_id=question.id,
                doc_type=DocumentType.PDF
            )
        else:
            # Resolve local path: either provided directly or downloaded from URL
            if file_path:
                actual_path = file_path
            elif question.is_url():
                import tempfile
                import httpx
                logger.info(f"Downloading PDF from URL: {question.file_content}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(question.file_content, follow_redirects=True, timeout=60)
                        response.raise_for_status()
                        tmp_file.write(response.content)
                    temp_file_path = tmp_file.name
                actual_path = temp_file_path
            else:
                raise ValueError("Either bucket_name/object_name, file_path or a URL in file_content must be provided")

            # Route based on strategy
            if strategy == 'fast':
                logger.info(f"[strategy=fast] Using PyMuPDF native path for {question.id}")
                result = await process_pdf_native(actual_path, question.id)
            elif strategy == 'auto':
                classification = await asyncio.to_thread(classify_pdf, actual_path)
                doc_type = classification['doc_type']
                logger.info(f"[strategy=auto] PDF classified as '{doc_type}' (scan_ratio={classification['scan_ratio']:.2f})")
                if doc_type == 'native':
                    result = await process_pdf_native(actual_path, question.id)
                else:
                    result = await processor.process_document(
                        file_path=actual_path,
                        doc_id=question.id,
                        doc_type=DocumentType.PDF
                    )
            else:
                result = await processor.process_document(
                    file_path=actual_path,
                    doc_id=question.id,
                    doc_type=DocumentType.PDF
                )
        
        # Cleanup temp file if created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # Generate captions for images if include_images is True
        if question.include_images and result and 'images' in result:
            await _generate_image_captions(
                images=result['images'],
                llm=llm,
                doc_id=question.id,
                text_elements=result.get('text_elements')
            )

        # Release raw PIL image data regardless of include_images — bytes were already
        # uploaded to MinIO and captions generated; keeping PIL objects wastes RAM
        # (300 DPI A4 page ≈ 26 MB RGB each).
        if result and 'images' in result:
            for _img in result['images']:
                _img['image_data'] = None

        # Generate descriptions for tables if include_tables is True
        if question.include_tables and result and 'tables' in result:
            await _generate_table_descriptions(
                tables=result['tables'],
                llm=llm,
                doc_id=question.id,
                text_elements=result.get('text_elements')
            )
        
        # Compute cross-modal references (surrounding_text, ref_tables, ref_images)
        if result:
            _compute_cross_modal_refs(
                text_elements=result.get('text_elements', []),
                tables=result.get('tables', []),
                images=result.get('images', [])
            )

        # Extract global doc_context for situated context (Contextual Retrieval)
        doc_context = None
        if result and 'text_elements' in result:
            text_els = result.get('text_elements', [])
            # Combine first 10 text elements to get about 1000-2000 chars of context
            combined_text = " ".join(el.get('text', '')[:200] for el in text_els[:10])
            doc_context = combined_text[:1500]

        # Track whether we've already done the first delete for this doc
        _first_index_done = False
        _sc_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        def _accumulate_sc(usage: dict):
            if usage:
                _sc_tokens["input_tokens"] += usage.get("input_tokens", 0)
                _sc_tokens["output_tokens"] += usage.get("output_tokens", 0)
                _sc_tokens["total_tokens"] += usage.get("total_tokens", 0)

        # Index tables to vector store if requested
        if question.index_tables_to_vector_store and question.include_tables and result and 'tables' in result:
            namespace = question.namespace if question.namespace else "default"
            _accumulate_sc(await _index_tables_to_vector_store(
                repo=repo,
                llm_embeddings=llm_embeddings,
                tables=result['tables'],
                question=question,
                namespace=namespace,
                engine=question.engine,
                sparse_encoder=question.sparse_encoder,
                tags=question.tags if question.tags else None,
                skip_delete=_first_index_done,
                llm=llm,
                doc_context=doc_context
            ))
            _first_index_done = True

        # Index images to vector store if requested
        if question.index_images_to_vector_store and question.include_images and result and 'images' in result:
            namespace = question.namespace if question.namespace else "default"
            _accumulate_sc(await _index_images_to_vector_store(
                repo=repo,
                llm_embeddings=llm_embeddings,
                images=result['images'],
                question=question,
                namespace=namespace,
                engine=question.engine,
                sparse_encoder=question.sparse_encoder,
                tags=question.tags if question.tags else None,
                skip_delete=_first_index_done,
                llm=llm,
                doc_context=doc_context
            ))
            _first_index_done = True

        # Extract entities using GraphRAG if requested
        if question.extract_entities and result and 'text_elements' in result and KNOWLEDGE_GRAPH_AVAILABLE:
            try:
                entity_extractor = PDFEntityExtractor()
                entity_stats = await entity_extractor.process_text_elements(
                    text_elements=result['text_elements'],
                    doc_id=question.id,
                    llm=llm,
                    hierarchy=result.get('hierarchy')
                )
                logger.info(f"Entity extraction stats: {entity_stats}")
            except Exception as e:
                logger.error(f"Failed to extract entities: {e}")

        # After processing, index text chunks to vector store
        if question.include_text and result and 'text_elements' in result:
            namespace = question.namespace if question.namespace else "default"
            _accumulate_sc(await _index_text_chunks(
                repo=repo,
                llm_embeddings=llm_embeddings,
                text_elements=result['text_elements'],
                question=question,
                namespace=namespace,
                engine=question.engine,
                sparse_encoder=question.sparse_encoder,
                hierarchy=result.get('hierarchy'),
                tags=question.tags if question.tags else None,
                skip_delete=_first_index_done,
                llm=llm,
                doc_context=doc_context
            ))

        # Ensure result is not None (should be dict)
        if result is None:
            raise RuntimeError("Processor returned None result")

        metadata = result.get('metadata', {})
        text_elements = result.get('text_elements', [])
        tables = result.get('tables', [])
        images = result.get('images', [])
        formulas = result.get('formulas', [])

        sc_used = _sc_tokens["total_tokens"] > 0

        return {
            "status": "success",
            "doc_id": question.id,
            "metadata": metadata,
            "statistics": {
                "total_pages": metadata.get('num_pages', 0),
                "text_elements": len(text_elements),
                "tables": len(tables),
                "images": len(images),
                "formulas": len(formulas),
                **({"situated_context_tokens": _sc_tokens} if sc_used else {}),
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF document: {e}", exc_info=True)
        raise


@inject_llm_chat_async
async def generate_image_caption(
    image_data: bytes,
    context_text: str,
    llm=None,
    **kwargs
) -> str:
    """
    Generate caption for an image using vision-capable LLM.
    
    Args:
        image_data: Raw image bytes
        context_text: Surrounding text context
        llm: Injected LLM instance
    
    Returns:
        Generated caption string
    """
    if llm is None:
        raise ValueError("LLM configuration is required for image captioning")
    
    import base64
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Prepare system prompt
    system_prompt = """You are a helpful AI assistant specialized in analyzing images from documents.
Describe the image accurately and concisely. Focus on:
- What type of visual element it is (chart, diagram, photo, etc.)
- Key information or data it contains
- Any text within the image
- The visual style or presentation

Keep your description under 100 words. Be factual and objective."""
    
    # Convert image to base64
    img_b64 = base64.b64encode(image_data).decode('utf-8')
    
    # Determine media type from magic bytes
    media_type = "image/png"
    if image_data.startswith(b'\x89PNG'):
        media_type = "image/png"
    elif image_data.startswith(b'\xff\xd8\xff'):
        media_type = "image/jpeg"
    
    # Build message content
    prompt_text = f"Analyze this image from a document. Context: {context_text}"
    
    message_content = []
    message_content.append({"type": "text", "text": prompt_text})
    
    # Determine provider for correct format
    llm_class_name = llm.__class__.__name__
    uses_openai_format = "OpenAI" in llm_class_name or "Google" in llm_class_name
    
    if uses_openai_format:
        data_uri = f"data:{media_type};base64,{img_b64}"
        message_content.append({
            "type": "image_url",
            "image_url": {"url": data_uri}
        })
    else:
        # Anthropic format
        message_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": img_b64
            }
        })
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=message_content)]
    
    try:
        response = await llm.ainvoke(messages)
        caption = extract_llm_text(response)
        logger.info(f"Generated image caption: {caption[:100]}...")
        return caption.strip()
    except Exception as e:
        logger.error(f"Error generating image caption: {e}", exc_info=True)
        return "Image extracted from document (caption generation failed)"


@inject_llm_async
async def generate_table_description(
    table_data: Dict[str, Any],
    llm=None,
    **kwargs
) -> str:
    """
    Generate semantic description for a table using LLM.
    
    Args:
        table_data: Dict with table info including df, caption, surrounding text
        llm: Injected LLM instance
    
    Returns:
        Generated description string
    """
    if llm is None:
        raise ValueError("LLM configuration is required for table description")
    
    df = table_data.get('data')
    caption = table_data.get('caption', '')
    surrounding_text = table_data.get('surrounding_text', '')
    
    if df is None or df.empty:
        return "Empty table"
    
    # Prepare table statistics
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    # Sample rows
    sample_rows = df.head(5).to_dict('records')
    
    # Build prompt
    human_prompt = f"""Analyze this table and provide a concise but informative description.

 {"Caption: " + caption if caption else ""}

 {"Context: " + surrounding_text if surrounding_text else ""}

Table Statistics:
- Rows: {stats['shape'][0]}, Columns: {stats['shape'][1]}
- Column names: {', '.join(stats['columns'])}
- Column types: {stats['dtypes']}

Sample Data (first 5 rows):
{sample_rows}

Provide a description that covers:
1. What kind of data this table contains
2. Key metrics or categories
3. Potential questions this table could answer
4. Any notable patterns or relationships

Keep the description under 200 words and focus on semantic meaning, not just technical details."""
    
    system_prompt = """You are a helpful AI assistant specialized in analyzing tabular data from documents.
Extract and summarize the semantic meaning of tables accurately."""
    
    try:
        description = await _invoke_llm_chat(system_prompt, human_prompt, llm)
        logger.info(f"Generated table description: {description[:100]}...")
        return description.strip()
    except Exception as e:
        logger.error(f"Error generating table description: {e}", exc_info=True)
        return f"Table with {len(df)} rows and {len(df.columns)} columns"


async def _generate_image_captions(images: List[Dict[str, Any]], llm, doc_id: str, text_elements: Optional[List[Dict[str, Any]]] = None):
    """
    Generate captions for images and update graph nodes.
    Uses ImageSemanticLinker for better context.
    """
    if not images:
        return
    
    # Initialize graph repository if available
    graph_repo = None
    if KNOWLEDGE_GRAPH_AVAILABLE and GraphRepository is not None:
        try:
            graph_repo = GraphRepository()
        except Exception as e:
            logger.warning(f"Failed to initialize GraphRepository: {e}")
            graph_repo = None
            
    linker = ImageSemanticLinker(graph_repository=None)
    
    for image_data in images:
        image_id = image_data.get('id')
        if not image_id:
            continue
        
        pil_image = image_data.get('image_data')
        if not pil_image:
            continue
        
        # Convert PIL Image to bytes
        import io
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        try:
            if text_elements:
                await linker.link_image_to_context(
                    image_data=image_data,
                    text_elements=text_elements,
                    llm=llm,
                    doc_id=doc_id
                )
                
            context_text = image_data.get('surrounding_text', '')
            caption = await generate_image_caption(img_bytes, context_text, llm=llm)
            
            # Update image_data with generated caption
            image_data['caption'] = caption
            # PIL Image no longer needed — bytes were extracted above for LLM call
            image_data['image_data'] = None

            # Update graph node if repository available
            if graph_repo:
                graph_repo.update_node(
                    node_id=image_id,
                    properties={'caption': caption}
                )
                logger.debug(f"Updated graph node {image_id} with caption")
                
        except Exception as e:
            logger.error(f"Failed to generate caption for image {image_id}: {e}")
            # Keep placeholder caption
            image_data['caption'] = f"Image extracted from page {image_data.get('page', 'unknown')}"


async def _generate_table_descriptions(tables: List[Dict[str, Any]], llm, doc_id: str, text_elements: Optional[List[Dict[str, Any]]] = None):
    """Generate LLM descriptions for tables. Uses TableSemanticLinker when text context is available."""
    if not tables:
        return

    linker = TableSemanticLinker(graph_repository=None)

    for table_data in tables:
        table_id = table_data.get('id')
        if not table_id:
            continue

        try:
            if text_elements:
                await linker.link_table_to_context(
                    table_data=table_data,
                    text_elements=text_elements,
                    llm=llm,
                    doc_id=doc_id
                )
            else:
                description = await generate_table_description(table_data, llm=llm)
                table_data['description'] = description

            logger.debug(f"Updated table {table_id} with description")
                
        except Exception as e:
            logger.error(f"Failed to generate description for table {table_id}: {e}")
            # Keep placeholder description
            if not table_data.get('description'):
                table_data['description'] = f"Table with {len(table_data.get('data', []))} rows"


async def _index_text_chunks(
    repo,
    llm_embeddings,
    text_elements: list,
    question: PDFScrapingRequest,
    namespace: str,
    engine,
    sparse_encoder: Union[str, TEIConfig, None] = None,
    hierarchy: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    skip_delete: bool = False,
    llm=None,
    doc_context: Optional[str] = None
):
    """
    Index text chunks to vector store.
    Uses ContextAwareChunker if hierarchy is available.
    """
    doc_id = question.id
    resolved_file_name = question.file_name or _extract_file_name(question.source or "")
    if hierarchy:
        chunker = ContextAwareChunker()
        documents = chunker.chunk_with_structure(
            doc_id=doc_id,
            text_elements=text_elements,
            structure=hierarchy
        )
        for doc in documents:
            doc.metadata['namespace'] = namespace
            doc.metadata['file_name'] = resolved_file_name
            doc.metadata['file_content'] = question.file_content
            doc.metadata['source'] = resolved_file_name
            if tags:
                doc.metadata['tags'] = tags
    else:
        from langchain_core.documents import Document
        documents = []
        for element in text_elements:
            # Yield control back to event loop for heartbeat
            await asyncio.sleep(0)

            text = element.get('text', '')
            if not text or not text.strip():
                continue

            meta = CommonChunkMetadata(
                id=doc_id,
                metadata_id=doc_id,
                doc_id=doc_id,
                namespace=namespace,
                source=resolved_file_name,
                file_name=resolved_file_name,
                file_content=question.file_content,
                page=element.get('page', 1),
                heading_path=element.get('heading_path', ''),
                type=element.get('type', 'text'),
                chunk_type='text',
                ref_tables=str(element.get('ref_tables', [])),
                ref_images=str(element.get('ref_images', [])),
                date=question.doc_date or "",
                tags=tags if tags else None,
            ).to_metadata_dict()
            meta = _apply_additional_metadata(meta, getattr(question, 'additional_metadata', None))
            doc = Document(
                page_content=text,
                metadata=meta
            )
            documents.append(doc)
    
    if not documents:
        logger.warning(f"No text elements to index for document {doc_id}")
        return {}

    logger.info(f"Indexing {len(documents)} text chunks to vector store for doc {doc_id}")

    sc_token_usage = {}
    # Situated context enrichment (Contextual Retrieval, optional)
    sc_config = getattr(question, 'situated_context', None)
    if sc_config and sc_config.enable:
        try:
            from tilellm.shared.situated_context import build_llm_from_config
            # Use dedicated LLM for situated context if configured, otherwise fall back to DI LLM
            sc_llm = await build_llm_from_config(sc_config) or llm
            if sc_llm:
                sc_result = await enrich_chunks_with_situated_context(
                    documents, 
                    sc_llm, 
                    doc_context=doc_context,
                    profile=sc_config.profile,
                    custom_prompt=sc_config.custom_prompt,
                            metadata_extraction_prompt=sc_config.metadata_extraction_prompt
                )
                documents = sc_result.documents
                sc_token_usage = sc_result.token_usage
                logger.info(f"Situated context applied to {len(documents)} text chunks. Tokens: {sc_token_usage}")
        except Exception as sc_err:
            logger.warning(f"Situated context enrichment failed, continuing without: {sc_err}")

    # Use repository to index documents
    await repo.aadd_documents(
        engine=engine,
        documents=documents,
        namespace=namespace,
        embedding_model=llm_embeddings,
        sparse_encoder=sparse_encoder,
        metadata_id=doc_id,
        skip_delete=skip_delete
    )

    logger.info(f"Successfully indexed {len(documents)} text chunks")
    return sc_token_usage


async def _index_tables_to_vector_store(
    repo,
    llm_embeddings,
    tables: list,
    question: PDFScrapingRequest,
    namespace: str,
    engine,
    sparse_encoder: Union[str, TEIConfig, None] = None,
    tags: Optional[List[str]] = None,
    skip_delete: bool = False,
    llm=None,
    doc_context: Optional[str] = None
):
    """
    Index table semantic descriptions to vector store.
    """
    from langchain_core.documents import Document
    
    if not tables:
        return
    
    doc_id = question.id
    resolved_file_name = question.file_name or _extract_file_name(question.source or "")
    documents = []
    for table_data in tables:
        await asyncio.sleep(0)
        table_id = table_data.get('id')

        # Use semantic description if available, otherwise fallback to basic description
        semantic_desc = table_data.get('semantic_description') or table_data.get('description')

        if not semantic_desc:
            continue

        # Create document for vector store
        meta = CommonChunkMetadata(
            id=doc_id,
            metadata_id=doc_id,
            doc_id=doc_id,
            namespace=namespace,
            source=resolved_file_name,
            file_name=resolved_file_name,
            file_content=question.file_content,
            page=table_data.get('page', 1),
            type='table',
            element_type='table',
            chunk_type='table_description',
            table_id=table_id or '',
            answerable_questions=str(table_data.get('answerable_questions', [])),
            columns=str(table_data.get('columns', [])),
            col_names=", ".join(table_data.get('columns', [])),
            surrounding_text=table_data.get('surrounding_text', ''),
            parquet_path=table_data.get('parquet_path', ''),
            md_path=table_data.get('md_path', ''),
            date=question.doc_date or "",
            tags=tags if tags else None,
        ).to_metadata_dict()
        meta = _apply_additional_metadata(meta, getattr(question, 'additional_metadata', None))
        doc = Document(
            page_content=semantic_desc,
            metadata=meta
        )
        documents.append(doc)
    
    if not documents:
        logger.warning(f"No table descriptions to index for document {doc_id}")
        return {}

    logger.info(f"Indexing {len(documents)} table descriptions to vector store for doc {doc_id}")

    sc_token_usage = {}
    sc_config = getattr(question, 'situated_context', None)
    if sc_config and sc_config.enable:
        needs_meta = _sc_needs_metadata_extraction(sc_config)
        if needs_meta:
            # Profile uses json_mode (e.g. pa_italiana): metadata extraction (act_type,
            # topics, amount…) must run on ALL table chunks, including those that already
            # have a semantic description from TableSemanticLinker.
            docs_to_enrich = documents
        else:
            # Generic SC profile: only adds a contextual sentence. Skip tables that
            # already have a rich semantic description to avoid redundant noise.
            docs_to_enrich = []
            for i, doc in enumerate(documents):
                if not tables[i].get('semantic_description'):
                    docs_to_enrich.append(doc)
                else:
                    doc.metadata["has_situated_context"] = True

        if docs_to_enrich:
            try:
                from tilellm.shared.situated_context import build_llm_from_config
                sc_llm = await build_llm_from_config(sc_config) or llm
                if sc_llm:
                    sc_result = await enrich_chunks_with_situated_context(
                        docs_to_enrich,
                        sc_llm,
                        doc_context=doc_context,
                        profile=sc_config.profile,
                        custom_prompt=sc_config.custom_prompt,
                        metadata_extraction_prompt=sc_config.metadata_extraction_prompt,
                    )
                    sc_token_usage = sc_result.token_usage
                    logger.info(f"Situated context applied to {len(docs_to_enrich)} table chunks. Tokens: {sc_token_usage}")
            except Exception as sc_err:
                logger.warning(f"Situated context enrichment for tables failed, continuing without: {sc_err}")
        else:
            logger.info(f"Skipped situated context for {len(documents)} semantic table descriptions (no metadata extraction needed)")

    # Index to vector store
    await repo.aadd_documents(
        engine=engine,
        documents=documents,
        namespace=namespace,
        embedding_model=llm_embeddings,
        sparse_encoder=sparse_encoder,
        metadata_id=doc_id,
        skip_delete=skip_delete
    )

    logger.info(f"Successfully indexed {len(documents)} table descriptions")
    return sc_token_usage


async def _index_images_to_vector_store(
    repo,
    llm_embeddings,
    images: list,
    question: PDFScrapingRequest,
    namespace: str,
    engine,
    sparse_encoder: Union[str, TEIConfig, None] = None,
    tags: Optional[List[str]] = None,
    skip_delete: bool = False,
    llm=None,
    doc_context: Optional[str] = None
):
    """
    Index image captions to vector store.
    """
    from langchain_core.documents import Document
    
    if not images:
        return
    
    doc_id = question.id
    resolved_file_name = question.file_name or _extract_file_name(question.source or "")
    documents = []
    for image_data in images:
        await asyncio.sleep(0)
        image_id = image_data.get('id')
        caption = image_data.get('caption')

        if not caption:
            continue

        # Create document for vector store
        meta = CommonChunkMetadata(
            id=doc_id,
            metadata_id=doc_id,
            doc_id=doc_id,
            namespace=namespace,
            source=resolved_file_name,
            file_name=resolved_file_name,
            file_content=question.file_content,
            page=image_data.get('page', 1),
            type='image',
            element_type='image',
            chunk_type='image_caption',
            image_id=image_id or '',
            surrounding_text=image_data.get('surrounding_text', ''),
            path=image_data.get('path', ''),
            date=question.doc_date or "",
            tags=tags if tags else None,
        ).to_metadata_dict()
        meta = _apply_additional_metadata(meta, getattr(question, 'additional_metadata', None))
        doc = Document(
            page_content=caption,
            metadata=meta
        )
        documents.append(doc)
    
    if not documents:
        logger.warning(f"No image captions to index for document {doc_id}")
        return {}

    logger.info(f"Indexing {len(documents)} image captions to vector store for doc {doc_id}")

    sc_token_usage = {}
    sc_config = getattr(question, 'situated_context', None)
    if sc_config and sc_config.enable:
        needs_meta = _sc_needs_metadata_extraction(sc_config)
        if needs_meta:
            # Metadata extraction must run on all image chunks regardless of caption quality.
            docs_to_enrich = documents
        else:
            # Generic SC profile: skip images that already have a rich LLM-generated caption.
            docs_to_enrich = []
            for i, doc in enumerate(documents):
                caption_text = images[i].get('caption', '')
                if "Image extracted from page" in caption_text:
                    docs_to_enrich.append(doc)
                else:
                    doc.metadata["has_situated_context"] = True

        if docs_to_enrich:
            try:
                from tilellm.shared.situated_context import build_llm_from_config
                sc_llm = await build_llm_from_config(sc_config) or llm
                if sc_llm:
                    sc_result = await enrich_chunks_with_situated_context(
                        docs_to_enrich,
                        sc_llm,
                        doc_context=doc_context,
                        profile=sc_config.profile,
                        custom_prompt=sc_config.custom_prompt,
                        metadata_extraction_prompt=sc_config.metadata_extraction_prompt,
                    )
                    sc_token_usage = sc_result.token_usage
                    logger.info(f"Situated context applied to {len(docs_to_enrich)} image chunks. Tokens: {sc_token_usage}")
            except Exception as sc_err:
                logger.warning(f"Situated context enrichment for images failed, continuing without: {sc_err}")
        else:
            logger.info(f"Skipped situated context for {len(documents)} rich image captions (no metadata extraction needed)")
    
    # Index to vector store
    await repo.aadd_documents(
        engine=engine,
        documents=documents,
        namespace=namespace,
        embedding_model=llm_embeddings,
        sparse_encoder=sparse_encoder,
        metadata_id=doc_id,
        skip_delete=skip_delete
    )

    logger.info(f"Successfully indexed {len(documents)} image captions")
    return sc_token_usage


@inject_llm_chat_async
@inject_repo_async
async def process_pdf_markdown_extraction(
    question: PDFScrapingRequest,
    bucket_name: Optional[str] = None,
    object_name: Optional[str] = None,
    file_path: Optional[str] = None,
    repo=None,
    llm=None,
    llm_embeddings=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process PDF document using LangGraph agent for Markdown extraction and indexing.
    
    This is the agentic approach using LangGraph to orchestrate:
    1. Document structure extraction with Docling
    2. Parallel image analysis with vision LLM
    3. Parallel table analysis with LLM
    4. Markdown assembly
    5. Structure-aware chunking and indexing
    
    Args:
        question: PDFScrapingRequest with document configuration
        repo: Injected vector store repository
        llm: Injected LLM instance for analysis
        llm_embeddings: Injected embeddings model
        bucket_name: Optional MinIO bucket name
        object_name: Optional MinIO object name
        file_path: Optional local file path
    
    Returns:
        Dict with processing results including markdown content and statistics
    """
    if repo is None:
        raise RuntimeError("Vector store repository not injected")
    
    if llm is None:
        raise ValueError("LLM configuration is required for Markdown extraction")
    
    if question.engine is None:
        raise ValueError("Engine configuration is required for vector store access")
    
    doc_id = question.id
    namespace = question.namespace if question.namespace else "default"
    
    logger.info(f"Starting LangGraph Markdown extraction for document {doc_id}")
    
    try:
        # extract_md_simple does not use MinIO, Neo4j, or FalkorDB.
        # Input must be a local file path or a public URL.
        temp_file_path = None
        if file_path:
            file_path_to_use = file_path
        elif question.is_url():
            import tempfile
            import httpx
            logger.info(f"Downloading PDF from URL for Markdown extraction: {question.file_content}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                async with httpx.AsyncClient() as client:
                    response = await client.get(question.file_content, follow_redirects=True, timeout=60)
                    response.raise_for_status()
                    tmp_file.write(response.content)
                temp_file_path = tmp_file.name
            file_path_to_use = temp_file_path
        else:
            raise ValueError("extract_md_simple requires file_path or a public URL in file_content")
        
        # Initialize LangGraph agent
        agent = MarkdownExtractionAgent()
        
        # Extract Markdown using the agent
        extraction_result = await agent.extract_markdown(
            file_path=file_path_to_use,
            doc_id=doc_id,
            llm=llm,
            include_images=question.include_images,
            include_tables=question.include_tables,
            include_formulas=question.include_formulas
        )
        
        markdown_content = extraction_result["markdown"]
        images = extraction_result.get("images", [])
        tables = extraction_result.get("tables", [])
        metadata = extraction_result.get("metadata", {})
        
        logger.info(f"Extracted {len(markdown_content)} characters of Markdown with "
                   f"{len(images)} images and {len(tables)} tables")
        
        # Chunk Markdown using specialized chunker
        chunker = MarkdownChunker(
            chunk_size=question.chunk_size if hasattr(question, 'chunk_size') else 1000,
            chunk_overlap=question.chunk_overlap if hasattr(question, 'chunk_overlap') else 200,
            respect_headings=True,
            respect_tables=True,
            include_heading_context=True
        )
        
        # Build source metadata, including tags if provided
        source_metadata = {
            'id': doc_id,
            'metadata_id': doc_id,
            'doc_id': doc_id,
            'source_type': 'markdown_extraction',
            'source': question.file_name,
            'file_name': question.file_name,
            'file_content': question.file_content,
            'has_images': len(images) > 0,
            'has_tables': len(tables) > 0,
            'num_pages': metadata.get('num_pages', 0),
            'namespace': namespace,
            'date': question.doc_date or "",
        }
        if question.tags:
            source_metadata['tags'] = question.tags
        _apply_additional_metadata(source_metadata, getattr(question, 'additional_metadata', None))

        # Build page_line_map from "## Page N" markers inserted by MarkdownExtractionAgent.
        # Maps each line index → page number so each chunk gets an accurate 'page' field.
        import re as _re
        _page_line_map: Dict[int, int] = {}
        _current_page = 1
        for _ln, _line in enumerate(markdown_content.split('\n')):
            _m = _re.match(r'^##\s+Page\s+(\d+)', _line.strip())
            if _m:
                _current_page = int(_m.group(1))
            _page_line_map[_ln] = _current_page

        # Use semantic section-based chunking for better results
        documents = chunker.chunk_with_semantic_splitting(
            markdown_content=markdown_content,
            doc_id=doc_id,
            source_metadata=source_metadata
        )

        for doc in documents:
            await asyncio.sleep(0)
            doc.metadata['namespace'] = namespace
            if question.tags and 'tags' not in doc.metadata:
                doc.metadata['tags'] = question.tags
            # Assign page from page_line_map using chunk start_line
            start_line = doc.metadata.get('start_line', 0)
            doc.metadata['page'] = _page_line_map.get(start_line, 1)
        
        logger.info(f"Created {len(documents)} Markdown chunks for indexing")

        sc_token_usage = None
        # Situated context enrichment (Contextual Retrieval, optional)
        sc_config = getattr(question, 'situated_context', None)
        if sc_config and sc_config.enable:
            try:
                from tilellm.shared.situated_context import build_llm_from_config
                # Use dedicated LLM for situated context if configured, otherwise fall back to DI LLM
                sc_llm = await build_llm_from_config(sc_config) or llm
                if sc_llm:
                    # Global doc context for Markdown
                    doc_context = markdown_content[:1500]
                    sc_result = await enrich_chunks_with_situated_context(
                        documents,
                        sc_llm,
                        doc_context=doc_context,
                        profile=sc_config.profile,
                        custom_prompt=sc_config.custom_prompt,
                        metadata_extraction_prompt=sc_config.metadata_extraction_prompt,
                    )
                    documents = sc_result.documents
                    sc_token_usage = sc_result.token_usage
                    logger.info(f"Situated context applied to {len(documents)} Markdown chunks. Tokens: {sc_token_usage}")
            except Exception as sc_err:
                logger.warning(f"Situated context enrichment for Markdown failed: {sc_err}")

        # Index chunks to vector store
        if documents:
            await repo.aadd_documents(
                engine=question.engine,
                documents=documents,
                namespace=namespace,
                embedding_model=llm_embeddings,
                sparse_encoder=question.sparse_encoder,
                metadata_id=doc_id
            )
            logger.info(f"Successfully indexed {len(documents)} Markdown chunks")

        # Cleanup temp file if created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return {
            "status": "success",
            "doc_id": doc_id,
            "extraction_method": "langgraph_agent",
            "markdown_length": len(markdown_content),
            "num_chunks": len(documents),
            "num_images": len(images),
            "num_tables": len(tables),
            "metadata": metadata,
            "markdown_preview": markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content,
            **({"situated_context_tokens": sc_token_usage} if sc_token_usage else {}),
        }
        
    except Exception as e:
        logger.error(f"Error in LangGraph Markdown extraction: {e}", exc_info=True)
        raise
