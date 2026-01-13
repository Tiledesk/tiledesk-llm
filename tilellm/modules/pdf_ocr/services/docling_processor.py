import logging
import asyncio
import json
import os
from io import BytesIO
from enum import Enum
from typing import Optional, Dict, Any

import pandas as pd
from minio import Minio
import duckdb

# Docling imports
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Project imports
from tilellm.shared.embedding_factory import AsyncEmbeddingFactory
from tilellm.modules.pdf_ocr.services.document_structure_extractor import DocumentStructureExtractor
from tilellm.shared.utility import get_service_config
from tilellm.modules.knowledge_graph.services.minio_storage import get_minio_storage_service

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"

class ProductionDocumentProcessor:
    """
    Production-grade document processor with Docling integration,
    Multimodal RAG support (Text, Tables, Images), and Knowledge Graph linking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 1. Initialize Docling
        self._init_docling()
        
        # 2. Initialize Storage Backends
        self._init_minio()
        self._init_duckdb()
        self._init_neo4j()
        
        # 3. Initialize Models (Lazy loaded usually, but we set up factories)
        self.embedding_factory = AsyncEmbeddingFactory()
        
        # 4. Initialize Structure Extractor with graph repository
        self.structure_extractor = DocumentStructureExtractor(
            graph_repository=self.graph_repository
        )
        
        # Redis for cache (e.g., CLIP embeddings)
        self._init_redis()

    def _init_docling(self):
        if not DOCLING_AVAILABLE:
            logger.warning("Docling not available. PDF parsing will be limited.")
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
            logger.info("Docling initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Docling: {e}")
            self.pdf_converter = None

    def _init_minio(self):
        try:
            self.minio_service = get_minio_storage_service()
            self.minio_bucket_tables = self.minio_service.bucket_tables
            self.minio_bucket_images = self.minio_service.bucket_images
            logger.info("Successfully initialized MinIOStorageService in ProductionDocumentProcessor")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO service: {e}")
            self.minio_service = None

    def _init_duckdb(self):
        # In-memory or persistent DuckDB
        # For this module, we might use a persistent path if mounted, or memory
        self.duckdb_path = os.getenv("DUCKDB_PATH", ":memory:") 
        try:
            self.duckdb_conn = duckdb.connect(self.duckdb_path)
            # Create metadata table
            self.duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS document_tables_metadata (
                    table_id VARCHAR PRIMARY KEY,
                    doc_id VARCHAR,
                    page_num INTEGER,
                    description TEXT,
                    schema_json JSON,
                    parquet_path VARCHAR,
                    row_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")

    def _init_neo4j(self):
        try:
            from tilellm.modules.knowledge_graph.repository.repository import GraphRepository
            self.graph_repository = GraphRepository()
            if not self.graph_repository.verify_connection():
                raise ConnectionError("Failed to verify connection to Neo4j database.")
            logger.info("Successfully connected to Neo4j via GraphRepository")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            self.graph_repository = None

    def _init_redis(self):
        import redis.asyncio as redis
        config = get_service_config()
        redis_conf = config.get("redis", {})
        
        self.redis_host = redis_conf.get("host", "localhost")
        self.redis_port = int(redis_conf.get("port", 6379))
        self.redis_db = int(redis_conf.get("db", 0))
        
        self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, db=self.redis_db)

    async def process_document(
        self, 
        file_path: str, 
        doc_id: str,
        doc_type: DocumentType = DocumentType.PDF
    ):
        """
        Main entry point for processing a document.
        """
        logger.info(f"Starting processing for document {doc_id} ({doc_type})")
        
        try:
            if doc_type == DocumentType.PDF:
                return await self._process_pdf_docling(file_path, doc_id)
            else:
                logger.warning(f"Unsupported document type: {doc_type}, falling back or skipping.")
                return None
        except Exception as e:
            logger.error(f"Processing failed for {doc_id}: {e}", exc_info=True)
            raise

    async def process_from_minio(
        self,
        bucket_name: str,
        object_name: str,
        doc_id: str,
        doc_type: DocumentType = DocumentType.PDF
    ):
        """
        Download file from MinIO and process it.
        """
        if not self.minio_service:
            raise RuntimeError("MinIO service not initialized")
            
        import tempfile
        
        try:
            # Create a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                # Download from MinIO
                response = self.minio_service.get_object(bucket_name, object_name)
                try:
                    for d in response.stream(32*1024):
                        tmp_file.write(d)
                finally:
                    response.close()
                    response.release_conn()
                
                tmp_path = tmp_file.name
            
            logger.info(f"Downloaded {object_name} to {tmp_path}")
            
            try:
                # Process
                return await self.process_document(tmp_path, doc_id, doc_type)
            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            logger.error(f"Failed to process from MinIO: {e}", exc_info=True)
            raise

    async def _process_pdf_docling(self, file_path: str, doc_id: str):
        if not self.pdf_converter:
            raise RuntimeError("Docling converter not initialized")

        # 0. Create Document node once at the beginning
        doc_node_id = None
        if self.graph_repository:
            try:
                from tilellm.modules.knowledge_graph.models import Node
                
                # Check if exists
                existing = self.graph_repository.find_nodes_by_property("Document", "id", doc_id)
                if existing:
                    doc_node_id = existing[0].id
                    logger.debug(f"Document node {doc_id} already exists with Neo4j ID {doc_node_id}")
                else:
                    doc_node = Node(
                        id=doc_id,
                        label="Document",
                        properties={"id": doc_id}
                    )
                    created_doc = self.graph_repository.create_node(doc_node)
                    doc_node_id = created_doc.id
                    logger.debug(f"Created Document node {doc_id} with ID {doc_node_id}")
            except Exception as e:
                logger.warning(f"Error checking/creating Document node: {e}")

        # 1. Convert with Docling (blocking call, run in executor)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.pdf_converter.convert, file_path)
        
        # 2. Extract structured content
        structured_content = {
            'text_elements': [],
            'tables': [],
            'images': [],
            'formulas': [],
            'metadata': {
                'doc_id': doc_id,
                'num_pages': len(result.pages)
            }
        }
        
        # robust extraction logic handling both Docling v2 (document-centric) and v1 (page-centric)
        doc = getattr(result, 'document', None)
        extracted_something = False

        if doc:
            logger.info("Using Docling v2 document-centric extraction")
            
            # TEXTS
            texts = getattr(doc, 'texts', [])
            for item in texts:
                text_content = getattr(item, 'text', '')
                if not text_content: continue
                
                # Provenance
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov: prov = prov[0]
                
                page_no = getattr(prov, 'page_no', 1)
                bbox = getattr(prov, 'bbox', None)
                if bbox and hasattr(bbox, 'as_tuple'): bbox = bbox.as_tuple()
                
                structured_content['text_elements'].append({
                    'id': f"{doc_id}_text_{getattr(item, 'self_ref', len(structured_content['text_elements']))}",
                    'text': text_content,
                    'page': page_no - 1,
                    'bbox': bbox,
                    'type': getattr(item, 'label', 'text')
                })
                extracted_something = True

            # TABLES
            tables = getattr(doc, 'tables', [])
            for item in tables:
                df = item.export_to_dataframe(doc) if hasattr(item, 'export_to_dataframe') else pd.DataFrame()
                
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov: prov = prov[0]
                page_no = getattr(prov, 'page_no', 1)
                bbox = getattr(prov, 'bbox', None)
                if bbox and hasattr(bbox, 'as_tuple'): bbox = bbox.as_tuple()

                structured_content['tables'].append({
                    'id': f"{doc_id}_table_{getattr(item, 'self_ref', len(structured_content['tables']))}",
                    'data': df,
                    'page': page_no - 1,
                    'bbox': bbox,
                    'caption': getattr(item, 'caption', None)
                })
                extracted_something = True

            # PICTURES
            pictures = getattr(doc, 'pictures', [])
            for item in pictures:
                image_data = item.get_image(doc) if hasattr(item, 'get_image') else getattr(item, 'image', None)
                
                prov = getattr(item, 'prov', None)
                if isinstance(prov, list) and prov: prov = prov[0]
                page_no = getattr(prov, 'page_no', 1)
                bbox = getattr(prov, 'bbox', None)
                if bbox and hasattr(bbox, 'as_tuple'): bbox = bbox.as_tuple()

                structured_content['images'].append({
                    'id': f"{doc_id}_image_{getattr(item, 'self_ref', len(structured_content['images']))}",
                    'image_data': image_data,
                    'page': page_no - 1,
                    'bbox': bbox
                })
                extracted_something = True

        # Fallback to page-based iteration if nothing extracted
        if not extracted_something:
            logger.info("Falling back to Docling page-centric extraction")
            for page_idx, page in enumerate(result.pages):
                 if hasattr(page, 'elements'):
                     elements = page.elements
                 else:
                     elements = []

                 for element in elements:
                    bbox = element.bbox if hasattr(element, 'bbox') else None
                    label = getattr(element, 'label', '').lower()
                    
                    if label == 'text':
                        structured_content['text_elements'].append({
                            'id': f"{doc_id}_text_{page_idx}_{getattr(element, 'id', 0)}",
                            'text': getattr(element, 'text', ''),
                            'page': page_idx,
                            'bbox': bbox,
                            'type': getattr(element, 'type', 'text')
                        })
                    elif label == 'table':
                        df = element.export_to_dataframe() if hasattr(element, 'export_to_dataframe') else pd.DataFrame()
                        structured_content['tables'].append({
                            'id': f"{doc_id}_table_{page_idx}_{getattr(element, 'id', 0)}",
                            'data': df,
                            'page': page_idx,
                            'bbox': bbox,
                            'caption': getattr(element, 'text', None)
                        })
                    elif label == 'picture':
                        structured_content['images'].append({
                            'id': f"{doc_id}_image_{page_idx}_{getattr(element, 'id', 0)}",
                            'image_data': getattr(element, 'image', None),
                            'page': page_idx,
                            'bbox': bbox
                        })
                    elif label == 'formula':
                        structured_content['formulas'].append({
                            'id': f"{doc_id}_formula_{page_idx}_{getattr(element, 'id', 0)}",
                            'latex': getattr(element, 'latex', ''),
                            'text': getattr(element, 'text', ''),
                            'page': page_idx,
                            'bbox': bbox
                        })

        # 3. Process batches
        await asyncio.gather(
            self._process_tables_batch(doc_id, structured_content['tables']),
            self._process_images_batch(doc_id, structured_content['images']),
            self._process_text_batch(doc_id, structured_content['text_elements'])
        )
        
        # 4. Build Spatial Graph
        await self._build_spatial_graph(doc_id, structured_content)
        
        # 5. Extract hierarchical structure
        try:
            hierarchy = self.structure_extractor.extract_hierarchy(doc_id, structured_content)
            structured_content['hierarchy'] = hierarchy
            logger.info(f"Successfully extracted hierarchy for {doc_id} with {hierarchy['metadata']['num_sections']} sections")
            
            # 6. Create Section nodes in Neo4j
            if self.graph_repository:
                await self.structure_extractor.create_section_nodes_in_graph(doc_id, doc_neo4j_id=doc_node_id)
                await self.structure_extractor.create_cross_reference_relationships(doc_id, structured_content)
        except Exception as e:
            logger.error(f"Failed to extract hierarchy for {doc_id}: {e}")
            structured_content['hierarchy'] = None

        return structured_content

    async def _process_tables_batch(self, doc_id: str, tables: list):
        tasks = [self._process_single_table(doc_id, table) for table in tables]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_table(self, doc_id: str, table_data: dict):
        table_id = table_data['id']
        df = table_data['data']
        
        if df.empty:
            return
            
        # Clean DF
        df = df.astype(str) # Simplify for parquet/storage
        
        # Save to Parquet MinIO
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, engine='pyarrow', compression='zstd', index=False)
        
        parquet_path = f"{doc_id}/tables/{table_id}.parquet"
        
        if self.minio_service:
            self.minio_service.upload_data(
                bucket_name=self.minio_bucket_tables,
                object_name=parquet_path,
                data=parquet_buffer.getvalue(),
                content_type='application/parquet'
            )
            
        # Register in DuckDB
        safe_table_name = f"tbl_{table_id.replace('-', '_')}"
        self.duckdb_conn.register(safe_table_name, df)
        
        # Get surrounding context for description
        surrounding_text = await self._get_surrounding_text(
            doc_id,
            table_data['page'],
            table_data['bbox']
        )
        
        # Store description for later use by logic layer
        table_data['surrounding_text'] = surrounding_text
        table_data['parquet_path'] = parquet_path
        table_data['duckdb_table'] = safe_table_name
        
        # Store Metadata
        description = f"Table from page {table_data['page']}. Columns: {', '.join(df.columns)}"
        
        self.duckdb_conn.execute(
            "INSERT INTO document_tables_metadata (table_id, doc_id, page_num, description, schema_json, parquet_path, row_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, current_timestamp)",
            [table_id, doc_id, table_data['page'], description, json.dumps({'columns': list(df.columns)}), parquet_path, len(df)]
        )
        
        # Neo4J Node via GraphRepository
        if self.graph_repository:
            try:
                from tilellm.modules.knowledge_graph.models import Node, Relationship
                
                table_node = Node(
                    id=table_id,
                    label="Table",
                    properties={
                        "doc_id": doc_id,
                        "page": table_data['page'],
                        "description": description,
                        "parquet_path": parquet_path,
                        "row_count": len(df),
                        "columns": list(df.columns)
                    }
                )
                
                created_table = self.graph_repository.create_node(table_node)
                table_data['neo4j_id'] = created_table.id
                
                # Find document node to link
                doc_nodes = self.graph_repository.find_nodes_by_property("Document", "id", doc_id)
                if doc_nodes:
                    doc_neo4j_id = doc_nodes[0].id
                    
                    # Create relationship with correct format
                    rel = Relationship(
                        source_id=doc_neo4j_id,
                        target_id=created_table.id,
                        type="CONTAINS_TABLE"
                    )
                    self.graph_repository.create_relationship(rel)
                else:
                    logger.warning(f"Could not find Document node {doc_id} to link table {table_id}")
                
            except Exception as e:
                logger.error(f"Error creating graph nodes for table: {e}")

    async def _process_images_batch(self, doc_id: str, images: list):
        tasks = [self._process_single_image(doc_id, img) for img in images]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_text_batch(self, doc_id: str, text_elements: list):
        """
        Create Text nodes for each text element and link to Document.
        """
        if not self.graph_repository or not text_elements:
            return

        # Find document node to link
        doc_neo4j_id = None
        try:
            doc_nodes = self.graph_repository.find_nodes_by_property("Document", "id", doc_id)
            if doc_nodes:
                doc_neo4j_id = doc_nodes[0].id
        except Exception as e:
            logger.error(f"Error looking up document node: {e}")
            return

        if not doc_neo4j_id:
            logger.warning(f"Skipping text node creation: Document {doc_id} not found in graph")
            return

        from tilellm.modules.knowledge_graph.models import Node, Relationship

        for element in text_elements:
            try:
                text_id = element.get('id')
                text_content = element.get('text', '')
                
                if not text_id: 
                    continue
                    
                text_node = Node(
                    id=text_id,
                    label="Text",
                    properties={
                        "doc_id": doc_id,
                        "text": text_content[:1000], # Truncate for graph property
                        "page": element.get('page', 0),
                        "type": element.get('type', 'text')
                    }
                )
                
                created_node = self.graph_repository.create_node(text_node)
                element['neo4j_id'] = created_node.id
                
                # Link to Document
                rel = Relationship(
                    source_id=doc_neo4j_id,
                    target_id=created_node.id,
                    type="CONTAINS_TEXT"
                )
                self.graph_repository.create_relationship(rel)
                
            except Exception as e:
                logger.error(f"Error creating text node {element.get('id')}: {e}")

    async def _process_single_image(self, doc_id: str, image_data: dict):
        image_id = image_data['id']
        image = image_data['image_data']
        
        if not image:
            return
            
        # Save PNG to MinIO
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG', optimize=True)
        image_path = f"{doc_id}/images/{image_id}.png"
        
        if self.minio_service:
            self.minio_service.upload_data(
                bucket_name=self.minio_bucket_images,
                object_name=image_path,
                data=img_buffer.getvalue(),
                content_type='image/png'
            )
        
        # Get surrounding context for captioning
        _surrounding_text = await self._get_surrounding_text(
            doc_id,
            image_data['page'],
            image_data['bbox']
        )
        
        # Captioning will be done via logic layer (stub for now)
        caption = f"Image extracted from page {image_data['page']}"
        
        # Store caption in metadata for later use
        image_data['caption'] = caption
        image_data['path'] = image_path
        
        # Neo4J Node via GraphRepository
        if self.graph_repository:
            try:
                from tilellm.modules.knowledge_graph.models import Node, Relationship
                
                image_node = Node(
                    id=image_id,
                    label="Image",
                    properties={
                        "doc_id": doc_id,
                        "page": image_data['page'],
                        "caption": caption,
                        "path": image_path
                    }
                )
                
                created_image = self.graph_repository.create_node(image_node)
                image_data['neo4j_id'] = created_image.id
                
                # Find document node to link
                doc_nodes = self.graph_repository.find_nodes_by_property("Document", "id", doc_id)
                if doc_nodes:
                    doc_neo4j_id = doc_nodes[0].id
                
                    # Create relationship with correct format
                    rel = Relationship(
                        source_id=doc_neo4j_id,
                        target_id=created_image.id,
                        type="CONTAINS_IMAGE"
                    )
                    self.graph_repository.create_relationship(rel)
                else:
                    logger.warning(f"Could not find Document node {doc_id} to link image {image_id}")
                
            except Exception as e:
                logger.error(f"Error creating graph nodes for image: {e}")
    
    async def _get_surrounding_text(self, doc_id: str, page: int, bbox: Any) -> str:
        """
        Get surrounding text context for an element based on bbox and page.
        """
        # This is typically called during initial processing before full structure is built.
        # It's a placeholder until we have all elements to compare distances.
        return f"Element on page {page}"

    async def _build_spatial_graph(self, doc_id: str, content: dict):
        """
        Build spatial and sequential relationships between document elements.
        
        Relationships:
        - FOLLOWS: Sequential reading order
        - NEAR_TO: Spatial proximity on the same page
        - CONTAINS: Document structure (handled by Hierarchy in logic)
        """
        if not self.graph_repository:
            return
            
        logger.info(f"Building spatial graph for document {doc_id}")
        
        try:
            from tilellm.modules.knowledge_graph.models import Node, Relationship
            
            # 1. Collect all elements across all pages
            all_elements = []
            all_elements.extend(content.get('text_elements', []))
            all_elements.extend(content.get('tables', []))
            all_elements.extend(content.get('images', []))
            all_elements.extend(content.get('formulas', []))
            
            if not all_elements:
                return

            # Sort by page and then by vertical position (y-coord of bbox)
            # bbox is [x1, y1, x2, y2]
            all_elements.sort(key=lambda x: (x.get('page', 0), (x.get('bbox') or [0, 0, 0, 0])[1]))
            
            # 2. Create FOLLOWS relationships (sequential order)
            for i in range(len(all_elements) - 1):
                source = all_elements[i]
                target = all_elements[i+1]
                
                source_nid = source.get('neo4j_id')
                target_nid = target.get('neo4j_id')
                
                if source_nid and target_nid:
                    rel = Relationship(
                        source_id=source_nid,
                        target_id=target_nid,
                        type="FOLLOWS",
                        properties={"doc_id": doc_id}
                    )
                    self.graph_repository.create_relationship(rel)

            # 3. Create NEAR_TO relationships (spatial proximity on same page)
            # We group by page to limit comparisons
            from collections import defaultdict
            elements_by_page = defaultdict(list)
            for el in all_elements:
                elements_by_page[el.get('page', 0)].append(el)
                
            for page, page_elements in elements_by_page.items():
                for i in range(len(page_elements)):
                    for j in range(i + 1, len(page_elements)):
                        el1 = page_elements[i]
                        el2 = page_elements[j]
                        
                        nid1 = el1.get('neo4j_id')
                        nid2 = el2.get('neo4j_id')
                        
                        if not nid1 or not nid2:
                            continue
                        
                        # Simple Euclidean distance between centers of bboxes
                        dist = self._calculate_bbox_distance(
                            el1.get('bbox'), 
                            el2.get('bbox')
                        )
                        
                        # Threshold for "near" (arbitrary, could be tuned)
                        # Assuming coordinates are normalized 0-1000 or points
                        if dist < 200: 
                            rel = Relationship(
                                source_id=nid1,
                                target_id=nid2,
                                type="NEAR_TO",
                                properties={"distance": float(dist), "doc_id": doc_id}
                            )
                            self.graph_repository.create_relationship(rel)
            
        except Exception as e:
            logger.error(f"Error building spatial graph for {doc_id}: {e}")

    def _calculate_bbox_distance(self, bbox1: Any, bbox2: Any) -> float:
        """Calculate distance between centers of two bboxes [x1, y1, x2, y2]."""
        if not bbox1 or not bbox2:
            return float('inf')
            
        c1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        c2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        import math
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    async def close(self):
        # GraphRepository uses static driver, no need to close here
        if self.redis_client:
            await self.redis_client.aclose()

