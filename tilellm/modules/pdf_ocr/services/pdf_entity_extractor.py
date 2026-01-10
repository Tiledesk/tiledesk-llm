"""
PDF Entity Extractor
Extracts semantic entities and relationships from PDF elements using GraphRAG.
"""

import logging
from typing import Dict, Any, List, Optional

try:
    from tilellm.modules.knowledge_graph.tools.graphrag_extractor import extract_entities
    from tilellm.modules.knowledge_graph.repository.repository import GraphRepository
    from tilellm.modules.knowledge_graph.models import Node, Relationship
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False

logger = logging.getLogger(__name__)

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
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from text elements and create nodes/relationships in Neo4j.
        
        Args:
            text_elements: List of text elements with 'text' and 'id'
            doc_id: Document ID
            llm: LLM instance for extraction
            entity_types: Optional list of entity types to extract
            
        Returns:
            Dict with extraction statistics
        """
        if not GRAPHRAG_AVAILABLE:
            logger.error("Knowledge Graph module or GraphRAG extractor not available")
            return {"status": "error", "message": "GraphRAG not available"}

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

        for element in text_elements:
            text = element.get('text', '').strip()
            element_id = element.get('id')
            
            if not text or len(text) < 50: # Skip very short snippets
                continue
                
            try:
                # 1. Extract entities with LLM via GraphRAG tool
                entities, relationships = await extract_entities(
                    text=text,
                    llm=llm,
                    entity_types=entity_types
                )
                
                # 2. Create entity nodes and link to source paragraph
                for entity in entities:
                    entity_name = entity.get('name')
                    if not entity_name:
                        continue
                        
                    entity_type = entity.get('type', 'concept')
                    # Use a global-ish ID or document-scoped ID? 
                    # For now, let's use a name-based ID to allow cross-document linking if desired,
                    # but typically GraphRAG uses name + type as key.
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
                
                stats["elements_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error extracting entities from element {element_id}: {e}")

        return stats
