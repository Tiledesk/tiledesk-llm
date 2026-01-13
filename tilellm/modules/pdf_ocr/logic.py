"""
Business Logic for PDF OCR Module.
Handles service initialization, dependency injection, and core operations.
"""

import logging
from typing import Dict, Any, Optional, List, Union

from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async, inject_llm_async
from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest
from .services.docling_processor import ProductionDocumentProcessor, DocumentType
from .services.pdf_entity_extractor import PDFEntityExtractor
from .services.context_aware_chunker import ContextAwareChunker
from .services.table_semantic_linker import TableSemanticLinker
from .services.image_semantic_linker import ImageSemanticLinker
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
        return response.content if hasattr(response, 'content') else str(response)
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
    
    # Initialize processor with the request config
    processor = ProductionDocumentProcessor(question.model_dump())
    
    try:
        # Process document from MinIO, file path, or direct content
        result = None
        if bucket_name and object_name:
            result = await processor.process_from_minio(
                bucket_name=bucket_name,
                object_name=object_name,
                doc_id=question.id,
                doc_type=DocumentType.PDF
            )
        elif file_path:
            result = await processor.process_document(
                file_path=file_path,
                doc_id=question.id,
                doc_type=DocumentType.PDF
            )
        else:
            raise ValueError("Either bucket_name/object_name or file_path must be provided")
        
        # Generate captions for images if include_images is True
        if question.include_images and result and 'images' in result:
            await _generate_image_captions(
                images=result['images'],
                llm=llm,
                doc_id=question.id,
                text_elements=result.get('text_elements')
            )
        
        # Generate descriptions for tables if include_tables is True
        if question.include_tables and result and 'tables' in result:
            await _generate_table_descriptions(
                tables=result['tables'],
                llm=llm,
                doc_id=question.id,
                text_elements=result.get('text_elements')
            )
        
        # Index tables to vector store if requested
        if question.index_tables_to_vector_store and question.include_tables and result and 'tables' in result:
            namespace = question.namespace or "default"
            await _index_tables_to_vector_store(
                repo=repo,
                llm_embeddings=llm_embeddings,
                tables=result['tables'],
                doc_id=question.id,
                namespace=namespace,
                engine=question.engine,
                sparse_encoder = question.sparse_encoder
            )
        
        # Index images to vector store if requested
        if question.index_images_to_vector_store and question.include_images and result and 'images' in result:
            namespace = question.namespace or "default"
            await _index_images_to_vector_store(
                repo=repo,
                llm_embeddings=llm_embeddings,
                images=result['images'],
                doc_id=question.id,
                namespace=namespace,
                engine=question.engine,
                sparse_encoder=question.sparse_encoder
            )
        
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
            namespace = question.namespace or "default"
            await _index_text_chunks(
                repo=repo,
                llm_embeddings=llm_embeddings,
                text_elements=result['text_elements'],
                doc_id=question.id,
                namespace=namespace,
                engine=question.engine,
                sparse_encoder=question.sparse_encoder,
                hierarchy=result.get('hierarchy')
            )
        
        # Ensure result is not None (should be dict)
        if result is None:
            raise RuntimeError("Processor returned None result")
        
        metadata = result.get('metadata', {})
        text_elements = result.get('text_elements', [])
        tables = result.get('tables', [])
        images = result.get('images', [])
        formulas = result.get('formulas', [])
        
        return {
            "status": "success",
            "doc_id": question.id,
            "metadata": metadata,
            "statistics": {
                "total_pages": metadata.get('num_pages', 0),
                "text_elements": len(text_elements),
                "tables": len(tables),
                "images": len(images),
                "formulas": len(formulas)
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF document: {e}", exc_info=True)
        raise
    finally:
        await processor.close()


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
        caption = response.content if hasattr(response, 'content') else str(response)
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
            
    linker = ImageSemanticLinker(graph_repository=graph_repo)
    
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
    """
    Generate descriptions for tables and update graph nodes.
    Uses TableSemanticLinker for better context and question generation.
    """
    if not tables:
        return
    
    # Initialize graph repository if available
    graph_repo = None
    if KNOWLEDGE_GRAPH_AVAILABLE and GraphRepository is not None:
        try:
            graph_repo = GraphRepository()
        except Exception as e:
            logger.warning(f"Failed to initialize GraphRepository: {e}")
            graph_repo = None
            
    linker = TableSemanticLinker(graph_repository=graph_repo)
    
    for table_data in tables:
        table_id = table_data.get('id')
        if not table_id:
            continue
        
        try:
            if text_elements:
                # Use advanced linker with context
                await linker.link_table_to_context(
                    table_data=table_data,
                    text_elements=text_elements,
                    llm=llm,
                    doc_id=doc_id
                )
                description = table_data.get('semantic_description')
            else:
                # Fallback to basic description
                description = await generate_table_description(table_data, llm=llm)
                table_data['description'] = description
                
                # Update graph node if repository available
                if graph_repo:
                    graph_repo.update_node(
                        node_id=table_id,
                        properties={'description': description}
                    )
            
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
    doc_id: str,
    namespace: str,
    engine,
    sparse_encoder: Union[str, TEIConfig, None] = None,
    hierarchy: Optional[Dict[str, Any]] = None
):
    """
    Index text chunks to vector store.
    Uses ContextAwareChunker if hierarchy is available.
    """
    if hierarchy:
        chunker = ContextAwareChunker()
        documents = chunker.chunk_with_structure(
            doc_id=doc_id,
            text_elements=text_elements,
            structure=hierarchy
        )
    else:
        from langchain_core.documents import Document
        documents = []
        for element in text_elements:
            text = element.get('text', '')
            if not text or not text.strip():
                continue
            
            doc = Document(
                page_content=text,
                metadata={
                    'doc_id': doc_id,
                    'page': element.get('page', 0),
                    'type': element.get('type', 'text'),
                    'chunk_type': 'text',
                    'source': f"docling_{doc_id}"
                }
            )
            documents.append(doc)
    
    if not documents:
        logger.warning(f"No text elements to index for document {doc_id}")
        return
    
    logger.info(f"Indexing {len(documents)} text chunks to vector store for doc {doc_id}")
    
    # Use repository to index documents
    await repo.aadd_documents(
        engine=engine,
        documents=documents,
        namespace=namespace,
        embedding_model=llm_embeddings,
        sparse_encoder=sparse_encoder  # Can be configured from request if needed
    )
    
    logger.info(f"Successfully indexed {len(documents)} text chunks")


async def _index_tables_to_vector_store(
    repo,
    llm_embeddings,
    tables: list,
    doc_id: str,
    namespace: str,
    engine,
    sparse_encoder: Union[str, TEIConfig, None] = None,
):
    """
    Index table semantic descriptions to vector store.
    """
    from langchain_core.documents import Document
    
    if not tables:
        return
    
    documents = []
    for table_data in tables:
        table_id = table_data.get('id')
        
        # Use semantic description if available, otherwise fallback to basic description
        semantic_desc = table_data.get('semantic_description') or table_data.get('description')
        
        if not semantic_desc:
            continue
        
        # Create document for vector store
        doc = Document(
            page_content=semantic_desc,
            metadata={
                'doc_id': doc_id,
                'table_id': table_id,
                'type': 'table',
                'chunk_type': 'table_description',
                'page': table_data.get('page', 0),
                'answerable_questions': table_data.get('answerable_questions', []),
                'columns': table_data.get('columns', []),
                'source': f"docling_{doc_id}"
            }
        )
        documents.append(doc)
    
    if not documents:
        logger.warning(f"No table descriptions to index for document {doc_id}")
        return
    
    logger.info(f"Indexing {len(documents)} table descriptions to vector store for doc {doc_id}")
    
    # Index to vector store
    await repo.aadd_documents(
        engine=engine,
        documents=documents,
        namespace=namespace,
        embedding_model=llm_embeddings,
        sparse_encoder=sparse_encoder
    )
    
    logger.info(f"Successfully indexed {len(documents)} table descriptions")


async def _index_images_to_vector_store(
    repo,
    llm_embeddings,
    images: list,
    doc_id: str,
    namespace: str,
    engine,
    sparse_encoder: Union[str, TEIConfig, None] = None
):
    """
    Index image captions to vector store.
    """
    from langchain_core.documents import Document
    
    if not images:
        return
    
    documents = []
    for image_data in images:
        image_id = image_data.get('id')
        caption = image_data.get('caption')
        
        if not caption:
            continue
        
        # Create document for vector store
        doc = Document(
            page_content=caption,
            metadata={
                'doc_id': doc_id,
                'image_id': image_id,
                'type': 'image',
                'chunk_type': 'image_caption',
                'page': image_data.get('page', 0),
                'surrounding_text': image_data.get('surrounding_text', ''),
                'path': image_data.get('path'),
                'source': f"docling_{doc_id}"
            }
        )
        documents.append(doc)
    
    if not documents:
        logger.warning(f"No image captions to index for document {doc_id}")
        return
    
    logger.info(f"Indexing {len(documents)} image captions to vector store for doc {doc_id}")
    
    # Index to vector store
    await repo.aadd_documents(
        engine=engine,
        documents=documents,
        namespace=namespace,
        embedding_model=llm_embeddings,
        sparse_encoder=sparse_encoder
    )
    
    logger.info(f"Successfully indexed {len(documents)} image captions")
