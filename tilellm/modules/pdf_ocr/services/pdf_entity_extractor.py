"""
PDF Entity Extractor
Extracts semantic entities and relationships from PDF elements using GraphRAG.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable

# Initialize logger first
logger = logging.getLogger(__name__)

# Import Knowledge Graph with graceful fallback
try:
    from tilellm.modules.knowledge_graph.tools.graphrag_extractor import extract_entities as graphrag_extract_entities
    from tilellm.modules.knowledge_graph.repository.repository import GraphRepository
    from tilellm.modules.knowledge_graph.models import Node, Relationship
    GRAPHRAG_AVAILABLE = True
    GRAPHRAG_IMPORT_ERROR = None
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    GRAPHRAG_IMPORT_ERROR = str(e)
    GraphRepository = None
    Node = None
    Relationship = None
    logger.warning(f"Knowledge Graph module not available: {e}")
    
    # Define stub functions if not available
    async def stub_extract_entities(*args, **kwargs):
        return [], []
    
    # Use actual or stub
    if GRAPHRAG_AVAILABLE:
        extract_entities = graphrag_extract_entities
    else:
        extract_entities = stub_extract_entities


class PDFEntityExtractor:
    """
    Extracts semantic entities and relationships from PDF elements using GraphRAG.
    
    This class bridges the PDF OCR module with the Knowledge Graph module's
    extraction capabilities.
    """

    def __init__(self, graph_repository: Optional[Any] = None):
        self.graph_repository = graph_repository
        if not self.graph_repository and GRAPHRAG_AVAILABLE:
            try:
                self.graph_repository = GraphRepository()
            except Exception as e:
                logger.warning(f"Could not initialize GraphRepository in PDFEntityExtractor: {e}")

    async def process_text_elements(
        self,
        text_elements: List[Dict[str, Any]],
        doc_id: str,
        llm: Any,
        entity_types: Optional[List[str]] = None,
        batch_size: int = 5,
        hierarchy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from text elements and create nodes/relationships in Neo4j.
        
        Args:
            text_elements: List of text elements with 'text' and 'id'
            doc_id: Document ID
            llm: LLM instance for extraction
            entity_types: Optional list of entity types to extract
            batch_size: Number of elements to process in parallel
            hierarchy: Optional document hierarchy for section-entity relationships
            
        Returns:
            Dict with extraction statistics
        """
        if not GRAPHRAG_AVAILABLE:
            logger.error(f"Knowledge Graph module or GraphRAG extractor not available. Import error: {GRAPHRAG_IMPORT_ERROR}")
            return {"status": "error", "message": f"GraphRAG not available: {GRAPHRAG_IMPORT_ERROR}"}

        if not self.graph_repository:
            logger.error("Graph repository not available")
            return {"status": "error", "message": "Graph repository not available"}

        if entity_types is None:
            entity_types = ["person", "organization", "concept", "method", "metric", "location", "date"]

        stats = {
            "entities_extracted": 0,
            "relationships_created": 0,
            "elements_processed": 0
        }

        logger.info(f"Extracting entities from {len(text_elements)} elements in document {doc_id}")

        # Build element to section mapping if hierarchy is provided
        element_to_section = {}
        if hierarchy:
            sections = hierarchy.get('sections', {})
            for section_id, section_data in sections.items():
                for element_id in section_data.get('elements', []):
                    element_to_section[element_id] = section_id

        # Filter valid elements
        valid_elements = [
            elem for elem in text_elements
            if elem.get('text', '').strip() and len(elem.get('text', '')) >= 50
        ]

        # Process in batches
        for i in range(0, len(valid_elements), batch_size):
            batch = valid_elements[i:i + batch_size]
            
            # Process batch in parallel
            batch_tasks = [
                self._process_single_element(elem, doc_id, llm, entity_types, element_to_section)
                for elem in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Aggregate stats
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch processing: {result}")
                    continue
                
                if result:
                    stats["entities_extracted"] += result.get("entities_extracted", 0)
                    stats["relationships_created"] += result.get("relationships_created", 0)
                    stats["elements_processed"] += 1

        logger.info(f"Entity extraction completed: {stats}")
        return stats

    async def _process_single_element(
        self,
        element: Dict[str, Any],
        doc_id: str,
        llm: Any,
        entity_types: List[str],
        element_to_section: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, int]]:
        """
        Process a single text element to extract entities.
        
        Returns:
            Dict with "entities_extracted" and "relationships_created" counts, or None on error
        """
        text = element.get('text', '').strip()
        element_id = element.get('id')
        
        if not element_id:
            return None
        
        try:
            # 1. Extract entities with LLM via GraphRAG tool
            entities, relationships = await extract_entities(
                text=text,
                llm=llm,
                entity_types=entity_types
            )
            
            stats = {"entities_extracted": 0, "relationships_created": 0}
            
            # 2. Create entity nodes and link to source paragraph
            for entity in entities:
                entity_name = entity.get('name')
                if not entity_name:
                    continue
                    
                entity_type = entity.get('type', 'concept')
                entity_node_id = f"entity_{entity_name.lower().replace(' ', '_')}"
                
                entity_node = Node(
                    id=entity_node_id,
                    label="Entity",
                    properties={
                        "name": entity_name,
                        "type": entity_type,
                        "description": entity.get('description', ''),
                        "last_doc_id": doc_id
                    }
                )
                
                self.graph_repository.create_node(entity_node)
                stats["entities_extracted"] += 1
                
                # Link source element -> entity
                rel = Relationship(
                    source_id=element_id,
                    target_id=entity_node_id,
                    type="MENTIONS",
                    properties={"doc_id": doc_id}
                )
                self.graph_repository.create_relationship(rel)
            
            # 3. Create relationships between entities
            for rel_data in relationships:
                source_name = rel_data.get('source')
                target_name = rel_data.get('target')
                
                if not source_name or not target_name:
                    continue
                    
                source_id = f"entity_{source_name.lower().replace(' ', '_')}"
                target_id = f"entity_{target_name.lower().replace(' ', '_')}"
                
                relationship = Relationship(
                    source_id=source_id,
                    target_id=target_id,
                    type="RELATES_TO",
                    properties={
                        "description": rel_data.get('description', ''),
                        "strength": rel_data.get('strength', 1),
                        "doc_id": doc_id,
                        "source_element": element_id
                    }
                )
                self.graph_repository.create_relationship(relationship)
                stats["relationships_created"] += 1
            
            # 4. Create Section -[:MENTIONS]-> Entity relationships
            if element_to_section:
                section_id = element_to_section.get(element_id)
                if section_id:
                    for entity_name in [entity.get('name') for entity in entities if entity.get('name')]:
                        entity_id = f"entity_{entity_name.lower().replace(' ', '_')}"
                        section_entity = Relationship(
                            source_id=section_id,
                            target_id=entity_id,
                            type="MENTIONS",
                            properties={"doc_id": doc_id}
                        )
                        self.graph_repository.create_relationship(section_entity)
                        stats["relationships_created"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error extracting entities from element {element_id}: {e}")
            return None
