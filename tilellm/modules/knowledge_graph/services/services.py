"""
Service layer for Knowledge Graph operations.
Handles business logic and orchestrates repository operations.
"""

import json
import logging
from typing import Optional, List, Dict, Any, cast
from datetime import datetime
from ..models import Node, NodeUpdate, Relationship, RelationshipUpdate
from ..repository import GraphRepository
from tilellm.models import Engine
from tilellm.models.schemas import RepositoryItems
from tilellm.tools.reranker import TileReranker
from langchain_core.documents import Document
from .clustering import ClusterService  # type: ignore

GRAPHRAG_AVAILABLE = False
GraphRAGExtractor = None

logger = logging.getLogger(__name__)

try:
    from tilellm.modules.knowledge_graph.tools.graphrag_extractor import GraphRAGExtractor
    GRAPHRAG_AVAILABLE = True
    logger.info("GraphRAG extractor available")
except ImportError as e:
    logger.warning(f"GraphRAG extractor not available: {e}")
    logger.debug(f"Import error details: {e}", exc_info=True)


class GraphService:
    """
    Service layer for knowledge graph operations.
    Provides business logic on top of the repository layer.
    """

    def __init__(self, repository: Optional[GraphRepository] = None):
        """
        Initialize the service with a repository.

        Args:
            repository: GraphRepository instance. If None, needs to be set later.
        """
        self.repository = repository

    def set_repository(self, repository: GraphRepository):
        """Set the repository instance"""
        self.repository = repository

    def _ensure_repository(self):
        """Ensure repository is initialized"""
        if not self.repository:
            raise RuntimeError("Repository not initialized. Call set_repository first.")

    def _get_repository(self) -> GraphRepository:
        """Get the repository instance, ensuring it's initialized"""
        self._ensure_repository()
        return cast(GraphRepository, self.repository)

    # ==================== NODE OPERATIONS ====================

    def create_node(self, node: Node, namespace: Optional[str] = None, index_name: Optional[str] = None) -> Node:
        """
        Create a new node with business logic validation.

        Args:
            node: Node to create
            namespace: Optional namespace to partition the graph
            index_name: Optional index_name (collection name) to partition the graph

        Returns:
            Created node with ID

        Raises:
            ValueError: If validation fails
        """
        self._ensure_repository()

        # Business logic validation
        if not node.label:
            raise ValueError("Node must have a label.")

        if not node.label[0].isupper():
            logger.warning(f"Node label '{node.label}' should start with uppercase for Neo4j convention.")

        logger.info(f"Creating node with label: {node.label}, namespace: {namespace}, index_name: {index_name}")
        return self._get_repository().create_node(node, namespace=namespace, index_name=index_name)

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node if found, None otherwise
        """
        self._ensure_repository()
        logger.info(f"Retrieving node with ID: {node_id}")
        return self._get_repository().find_node_by_id(node_id)

    def get_nodes_by_label(self, label: str, limit: int = 100, namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
        """
        Get all nodes with a specific label.

        Args:
            label: Node label
            limit: Maximum number of nodes to return
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes

        Returns:
            List of nodes
        """
        self._ensure_repository()
        logger.info(f"Retrieving nodes with label: {label}, limit: {limit}, namespace: {namespace}, index_name: {index_name}")
        return self._get_repository().find_nodes_by_label(label, limit, namespace=namespace, index_name=index_name)

    def search_nodes(self, label: str, property_key: str, property_value: Any, limit: int = 100, namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
        """
        Search nodes by property value.

        Args:
            label: Node label
            property_key: Property key to search
            property_value: Property value to match
            limit: Maximum number of nodes to return
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes

        Returns:
            List of matching nodes
        """
        self._ensure_repository()
        logger.info(f"Searching nodes with {label}.{property_key} = {property_value}, namespace: {namespace}, index_name: {index_name}")
        return self._get_repository().find_nodes_by_property(label, property_key, property_value, limit, namespace=namespace, index_name=index_name)

    def update_node(self, node_id: str, node_update: NodeUpdate) -> Optional[Node]:
        """
        Update a node's properties.

        Args:
            node_id: Node ID
            node_update: Update data

        Returns:
            Updated node if found, None otherwise

        Raises:
            ValueError: If validation fails
        """
        self._ensure_repository()

        # Check if node exists
        existing_node = self._get_repository().find_node_by_id(node_id)
        if not existing_node:
            logger.warning(f"Node with ID {node_id} not found for update")
            return None

        # Validate label if provided
        if node_update.label and not node_update.label[0].isupper():
            logger.warning(f"Node label '{node_update.label}' should start with uppercase for Neo4j convention.")

        logger.info(f"Updating node with ID: {node_id}")
        return self._get_repository().update_node(
            node_id=node_id,
            label=node_update.label,
            properties=node_update.properties
        )

    def delete_node(self, node_id: str, detach: bool = True) -> bool:
        """
        Delete a node.

        Args:
            node_id: Node ID
            detach: If True, also delete all relationships

        Returns:
            True if deleted, False if not found
        """
        self._ensure_repository()
        logger.info(f"Deleting node with ID: {node_id}, detach: {detach}")

        # Check if node exists before deletion
        existing_node = self._get_repository().find_node_by_id(node_id)
        if not existing_node:
            logger.warning(f"Node with ID {node_id} not found for deletion")
            return False

        return self._get_repository().delete_node(node_id, detach)

    # ==================== RELATIONSHIP OPERATIONS ====================

    def create_relationship(self, relationship: Relationship, namespace: Optional[str] = None, index_name: Optional[str] = None) -> Relationship:
        """
        Create a relationship between nodes.

        Args:
            relationship: Relationship to create
            namespace: Optional namespace to partition the graph
            index_name: Optional index_name (collection name) to partition the graph

        Returns:
            Created relationship with ID

        Raises:
            ValueError: If validation fails
            RuntimeError: If source or target nodes don't exist
        """
        self._ensure_repository()

        # Business logic validation
        if not relationship.type:
            raise ValueError("Relationship must have a type.")

        if not relationship.type.isupper():
            logger.warning(f"Relationship type '{relationship.type}' should be uppercase for Neo4j convention.")

        # Verify source and target nodes exist
        source_node = self._get_repository().find_node_by_id(relationship.source_id)
        if not source_node:
            raise ValueError(f"Source node with ID {relationship.source_id} not found.")

        target_node = self._get_repository().find_node_by_id(relationship.target_id)
        if not target_node:
            raise ValueError(f"Target node with ID {relationship.target_id} not found.")

        logger.info(f"Creating relationship: {relationship.source_id} -[{relationship.type}]-> {relationship.target_id}, namespace: {namespace}, index_name: {index_name}")
        return self._get_repository().create_relationship(relationship, namespace=namespace, index_name=index_name)

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """
        Get a relationship by ID.

        Args:
            relationship_id: Relationship ID

        Returns:
            Relationship if found, None otherwise
        """
        self._ensure_repository()
        logger.info(f"Retrieving relationship with ID: {relationship_id}")
        return self._get_repository().find_relationship_by_id(relationship_id)

    def get_node_relationships(self, node_id: str, direction: str = "both") -> List[Relationship]:
        """
        Get all relationships for a node.

        Args:
            node_id: Node ID
            direction: "incoming", "outgoing", or "both"

        Returns:
            List of relationships

        Raises:
            ValueError: If direction is invalid
        """
        self._ensure_repository()

        if direction not in ["incoming", "outgoing", "both"]:
            raise ValueError(f"Invalid direction: {direction}. Must be 'incoming', 'outgoing', or 'both'.")

        logger.info(f"Retrieving {direction} relationships for node: {node_id}")
        return self._get_repository().find_relationships_by_node(node_id, direction)

    def update_relationship(self, relationship_id: str, relationship_update: RelationshipUpdate) -> Optional[Relationship]:
        """
        Update a relationship.

        Args:
            relationship_id: Relationship ID
            relationship_update: Update data

        Returns:
            Updated relationship if found, None otherwise

        Raises:
            ValueError: If validation fails
        """
        self._ensure_repository()

        # Check if relationship exists
        existing_rel = self._get_repository().find_relationship_by_id(relationship_id)
        if not existing_rel:
            logger.warning(f"Relationship with ID {relationship_id} not found for update")
            return None

        # Validate type if provided
        if relationship_update.type and not relationship_update.type.isupper():
            logger.warning(f"Relationship type '{relationship_update.type}' should be uppercase for Neo4j convention.")

        logger.info(f"Updating relationship with ID: {relationship_id}")
        return self._get_repository().update_relationship(
            relationship_id=relationship_id,
            rel_type=relationship_update.type,
            properties=relationship_update.properties
        )

    def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a relationship.

        Args:
            relationship_id: Relationship ID

        Returns:
            True if deleted, False if not found
        """
        self._ensure_repository()
        logger.info(f"Deleting relationship with ID: {relationship_id}")

        # Check if relationship exists before deletion
        existing_rel = self._get_repository().find_relationship_by_id(relationship_id)
        if not existing_rel:
            logger.warning(f"Relationship with ID {relationship_id} not found for deletion")
            return False

        return self._get_repository().delete_relationship(relationship_id)

    # ==================== UTILITY OPERATIONS ====================

    def verify_connection(self) -> bool:
        """
        Verify the connection to Neo4j.

        Returns:
            True if connection is successful
        """
        self._ensure_repository()
        logger.info("Verifying Neo4j connection")
        return self._get_repository().verify_connection()

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with node count, relationship count, etc.
        """
        self._ensure_repository()
        logger.info("Retrieving database statistics")
        return self._get_repository().get_database_info()


class GraphRAGService:
    """
    Service for GraphRAG operations.
    Integrates Neo4j knowledge graph with vector store and LLM for extraction and QA.
    """

    def __init__(self, graph_service: GraphService, llm=None, vector_store_repository=None):
        """
        Initialize GraphRAG service.

        Args:
            graph_service: GraphService instance for Neo4j operations
            llm: Optional LLM for extraction and QA (if None, will need to be set later)
            vector_store_repository: Optional vector store repository for retrieving documents
        """
        self.graph_service = graph_service
        self.llm = llm
        self.vector_store_repository = vector_store_repository

    def set_llm(self, llm):
        """Set the LLM instance"""
        self.llm = llm

    def set_vector_store_repository(self, repo):
        """Set the vector store repository"""
        self.vector_store_repository = repo

    def _ensure_llm(self):
        """Ensure LLM is initialized"""
        if not self.llm:
            raise RuntimeError("LLM not initialized. Call set_llm first.")

    def _ensure_vector_store(self):
        """Ensure vector store repository is initialized"""
        if not self.vector_store_repository:
            raise RuntimeError("Vector store repository not initialized. Call set_vector_store_repository first.")

    async def import_from_vector_store(self, namespace: str, vector_store_repo=None, engine: Optional[Engine] = None, limit: int = 100, llm=None, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Import documents from a vector store namespace into the knowledge graph.

        Steps:
        1. Retrieve all chunks from the vector store namespace
        2. Extract entities and relationships using GraphRAG extractor
        3. Store them as nodes and relationships in Neo4j

        Args:
            namespace: Namespace/collection name in vector store
            vector_store_repo: Vector store repository instance (if None, uses self.vector_store_repository)
            engine: Engine configuration for vector store (required if repo doesn't store engine)
            limit: Maximum number of chunks to process
            llm: LLM instance for extraction (if None, uses self.llm)
            index_name: Optional index_name for graph partition (defaults to engine.index_name)

        Returns:
            Dictionary with import statistics
        """
        # Use provided repo or fall back to instance repo
        repo = vector_store_repo or self.vector_store_repository
        if repo is None:
            raise RuntimeError("Vector store repository not provided and not set in service.")
        
        if engine is None:
            raise ValueError("Engine configuration is required for vector store retrieval.")
        
        # Determine which LLM to use (required for GraphRAG extraction)
        llm_to_use = llm or self.llm
        if llm_to_use is None:
            logger.error("LLM not provided - GraphRAG extraction requires LLM for entity and relationship extraction.")
            raise ValueError("LLM configuration is required for GraphRAG extraction. Provide LLM instance with chat/invoke method.")
        self.graph_service._ensure_repository()
        index_name = index_name if index_name is not None else (engine.index_name if engine else None)

        logger.info(f"Importing from vector store namespace: {namespace}, limit: {limit}, index_name: {index_name}")

        try:
            # Get all chunks from the namespace
            logger.info(f"Retrieving chunks from vector store namespace: {namespace}")
            result: RepositoryItems = await repo.get_all_obj_namespace(engine=engine, namespace=namespace,with_text=True)

            chunks = result.matches
            if not chunks:
                logger.warning(f"No chunks found in namespace: {namespace}")
                return {
                    "namespace": namespace,
                    "chunks_processed": 0,
                    "nodes_created": 0,
                    "relationships_created": 0,
                    "status": "empty"
                }
            
            # Limit the number of chunks to process
            chunks_to_process = chunks[:limit]
            logger.info(f"Processing {len(chunks_to_process)} chunks (limit: {limit})")
            
            # Extract text and IDs from chunks
            chunks_data = []

            for chunk in chunks_to_process:
                chunk_id = getattr(chunk, 'id', 'unknown')
                chunk_text = getattr(chunk, 'text', '')
                if chunk_text:
                    logger.info(f"Chunk {chunk_id} has text content... append")
                    chunks_data.append({"id": chunk_id, "text": chunk_text})
                else:
                    logger.warning(f"Chunk {chunk_id} has no text content, skipping")
            
            if not chunks_data:
                logger.warning("No text content found in any chunks")
                return {
                    "namespace": namespace,
                    "chunks_processed": len(chunks_to_process),
                    "nodes_created": 0,
                    "relationships_created": 0,
                    "status": "no_text"
                }
            
            nodes_created = 0
            relationships_created = 0
            
            # Extract entities and relationships using GraphRAG if LLM is available
            logger.info(f"GraphRAG extraction check - llm_to_use: {llm_to_use}, GRAPHRAG_AVAILABLE: {GRAPHRAG_AVAILABLE}, llm_type: {type(llm_to_use) if llm_to_use else None}")
            if llm_to_use is not None and GRAPHRAG_AVAILABLE:
                try:
                    logger.info("Starting GraphRAG extraction with LLM")
                    if GraphRAGExtractor is None:
                        logger.warning("GraphRAG extractor not available despite flag")
                        raise ImportError("GraphRAG extractor not available")
                    assert GraphRAGExtractor is not None
                    extractor = GraphRAGExtractor(llm_to_use)
                    entities, relationships = await extractor.extract(
                        doc_id=namespace,
                        chunks=chunks_data
                    )
                    
                    logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
                    
                    # Create nodes for extracted entities
                    entity_node_map = {}  # entity_name -> node_id
                    for entity in entities:
                        try:
                            entity_name = entity.get("entity_name", "").strip()
                            entity_type = entity.get("entity_type", "ENTITY").strip()
                            description = entity.get("description", "")
                            source_ids = entity.get("source_id", [])
                            
                            if not entity_name:
                                continue
                            
                            # Check if entity already exists (by name and type)
                            # For simplicity, create new node each time (could be deduplicated later)
                            node = Node(
                                label=entity_type,
                                properties={
                                    "name": entity_name,
                                    "description": description,
                                    "source_ids": source_ids if source_ids else [],
                                    "entity_type": entity_type,
                                    "import_timestamp": datetime.now().isoformat()
                                }
                            )
                            created_node = self.graph_service.create_node(node, namespace=namespace, index_name=index_name)
                            if created_node.id is None:
                                logger.error("Created entity node has no ID, skipping")
                                continue
                            
                            entity_node_map[entity_name] = created_node.id
                            nodes_created += 1
                            
                        except Exception as e:
                            logger.error(f"Error creating entity node {entity.get('entity_name', 'unknown')}: {e}")
                            continue
                    
                    # Create relationships between entities
                    for rel in relationships:
                        try:
                            source_name = rel.get("src_id", "").strip()
                            target_name = rel.get("tgt_id", "").strip()
                            rel_type = "RELATED_TO"  # Could be derived from description
                            weight = rel.get("weight", 1.0)
                            description = rel.get("description", "")
                            source_ids = rel.get("source_id", [])
                            
                            if not source_name or not target_name:
                                continue
                            
                            source_node_id = entity_node_map.get(source_name)
                            target_node_id = entity_node_map.get(target_name)
                            
                            if not source_node_id or not target_node_id:
                                logger.debug(f"Missing source or target node for relationship {source_name} -> {target_name}")
                                continue
                            
                            relationship = Relationship(
                                source_id=source_node_id,
                                target_id=target_node_id,
                                type=rel_type,
                                properties={
                                    "weight": weight,
                                    "description": description,
                                    "source_ids": source_ids if source_ids else [],
                                    "source_entity": source_name,
                                    "target_entity": target_name
                                }
                            )
                            self.graph_service.create_relationship(relationship, namespace=namespace, index_name=index_name)
                            relationships_created += 1
                            
                        except Exception as e:
                            logger.error(f"Error creating relationship {rel.get('src_id', 'unknown')} -> {rel.get('tgt_id', 'unknown')}: {e}")
                            continue
                    
                    logger.info(f"Created {nodes_created} entity nodes and {relationships_created} relationships")
                    
                except Exception as e:
                    logger.error(f"GraphRAG extraction failed: {e}")
                    # Fall back to simple document nodes
                    logger.info("Falling back to simple document nodes")
                    return await self._create_simple_document_nodes(chunks_to_process, namespace, index_name=index_name)
            else:
                # No LLM available, create simple document nodes
                logger.info("LLM not available for GraphRAG, creating simple document nodes")
                return await self._create_simple_document_nodes(chunks_to_process, namespace, index_name=index_name)
            
            return {
                "namespace": namespace,
                "chunks_processed": len(chunks_to_process),
                "nodes_created": nodes_created,
                "relationships_created": relationships_created,
                "status": "success" if nodes_created > 0 else "no_entities"
            }
            
        except Exception as e:
            logger.error(f"Error importing from vector store: {e}")
            raise

    async def _create_simple_document_nodes(self, chunks, namespace: str, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create simple Document nodes for chunks when GraphRAG extraction is not possible.
        
        Args:
            chunks: List of chunk objects from vector store
            namespace: Namespace/collection name
            
        Returns:
            Dictionary with import statistics
        """
        nodes_created = 0
        for chunk in chunks:
                chunk_id = getattr(chunk, 'id', 'unknown')
                chunk_text = getattr(chunk, 'text', '')
                metadata = getattr(chunk, 'metadata', {})
                try:
                    # Convert metadata dict to JSON string for Neo4j compatibility
                    metadata_json = "{}"
                    if metadata:
                        try:
                            metadata_json = json.dumps(metadata, default=str)
                        except Exception as json_err:
                            logger.warning(f"Failed to serialize metadata for chunk {chunk_id}: {json_err}, using empty JSON")
                            metadata_json = "{}"
                    
                    node = Node(
                        label="Document",
                        properties={
                            "chunk_id": chunk_id,
                            "text": chunk_text[:500],  # Truncate long text
                            "metadata": metadata_json,  # Store as JSON string
                            "import_timestamp": datetime.now().isoformat(),
                            "source_namespace": namespace,
                            "index_name": index_name
                        }
                    )
                    self.graph_service.create_node(node, namespace=namespace, index_name=index_name)
                    nodes_created += 1
                except Exception as e:
                    logger.error(f"Error creating document node for chunk {chunk_id}: {e}")
                    continue
        
        return {
            "namespace": namespace,
            "chunks_processed": len(chunks),
            "nodes_created": nodes_created,
            "relationships_created": 0,
            "status": "simple_documents"
        }

    async def answer_with_graph(self, question: str, namespace: str, max_results: Optional[int] = 10, similarity_threshold: Optional[float] = 0.3, llm=None, llm_embeddings=None, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a question using the knowledge graph (GraphRAG).

        Args:
            question: The question to answer
            namespace: Namespace/collection to search within (optional, may be used for filtering)
            max_results: Maximum number of graph elements to consider
            similarity_threshold: Similarity threshold for relevance
            llm: LLM instance for answer synthesis (if None, uses self.llm)
            llm_embeddings: Embedding model (optional)

        Returns:
            Dictionary with answer and graph elements used
        """
        # TODO: Implement actual GraphRAG QA
        # For now return stub
        logger.info(f"Graph QA for question: {question}, namespace: {namespace}")
        
        # Use defaults if None
        max_results = max_results or 10
        similarity_threshold = similarity_threshold or 0.3
        
        # Determine which LLM to use (optional for now as synthesis not implemented)
        llm_to_use = llm or self.llm
        if llm_to_use is None:
            logger.warning("LLM not provided - answer synthesis will be basic. Provide LLM for GraphRAG QA.")
        
        # Ensure repository is available
        self.graph_service._ensure_repository()
        repo = self.graph_service._get_repository()
        
        # Example: search nodes and relationships by text
        nodes = repo.search_nodes_by_text(question, limit=max_results, namespace=namespace, index_name=index_name)
        relationships = repo.search_relationships_by_text(question, limit=max_results, namespace=namespace, index_name=index_name)
        
        # Convert to simple dicts for response
        entities = []
        for node in nodes:
            entities.append({
                "id": node.id,
                "label": node.label,
                "properties": node.properties
            })
        
        rels = []
        for rel in relationships:
            rels.append(rel)
        
        # Generate answer using LLM if available
        if llm_to_use is not None:
            try:
                # Prepare context for LLM
                context = f"Question: {question}\n\n"
                if entities:
                    context += "Relevant entities found:\n"
                    for i, entity in enumerate(entities[:5], 1):
                        context += f"{i}. {entity.get('label', 'Unknown')}: {entity.get('properties', {}).get('name', 'No name')}\n"
                        if 'description' in entity.get('properties', {}):
                            context += f"   Description: {entity['properties']['description'][:100]}...\n"
                
                if rels:
                    context += "\nRelevant relationships found:\n"
                    for i, rel in enumerate(rels[:5], 1):
                        if hasattr(rel, 'properties') and rel.properties:
                            desc = rel.properties.get('description', 'No description')
                            context += f"{i}. {rel.type}: {desc[:100]}...\n"
                
                # Create prompt for LLM
                prompt = f"""Based on the following knowledge graph information, answer the question.

{context}

Question: {question}

Provide a concise and accurate answer based on the knowledge graph. If the information is insufficient, say so."""
                
                # Call LLM
                if hasattr(llm_to_use, 'invoke'):
                    from langchain_core.messages import HumanMessage, SystemMessage
                    messages = [
                        SystemMessage(content="You are a helpful assistant that answers questions based on knowledge graph information."),
                        HumanMessage(content=prompt)
                    ]
                    response = await llm_to_use.ainvoke(messages)
                    answer = response.content if hasattr(response, 'content') else str(response)
                elif hasattr(llm_to_use, 'chat'):
                    answer = await llm_to_use.chat(
                        system="You are a helpful assistant that answers questions based on knowledge graph information.",
                        messages=[{"role": "user", "content": prompt}]
                    )
                else:
                    answer = await llm_to_use(prompt)
                
                logger.info(f"Generated answer using LLM: {answer[:100]}...")
                
            except Exception as e:
                logger.error(f"Error generating answer with LLM: {e}")
                answer = f"I found {len(entities)} entities and {len(rels)} relationships related to your question. LLM synthesis failed: {str(e)}"
        else:
            answer = f"I found {len(entities)} entities and {len(rels)} relationships related to your question. Provide LLM configuration for detailed answer synthesis."
        
        return {
            "answer": answer,
            "entities": entities,
            "relationships": rels,
            "query_used": question
        }

    async def answer_with_graph_advanced(self, question: str, namespace: str, max_results: Optional[int] = 20, 
                                         similarity_threshold: Optional[float] = 0.3, llm=None,
                                         vector_weight: float = 1.0, keyword_weight: float = 1.0, 
                                         graph_weight: float = 1.0, query_type: Optional[str] = None,
                                         use_reranking: bool = True, max_expansion_nodes: int = 20,
                                         engine: Optional[Engine] = None, vector_store_repo=None,
                                         embedding_dimension=None, llm_embeddings=None,
                                         chat_history_dict=None) -> Dict[str, Any]:
        """
        Advanced GraphRAG QA with hybrid retrieval and graph expansion.
        
        Implements the full pipeline:
        1. Query analysis and type detection
        2. Global Search (Community-based) if applicable
        3. Parallel vector and keyword retrieval from vector store
        4. Reciprocal Rank Fusion (RRF) to combine results
        5. Graph expansion from seed nodes in Neo4j
        6. Cross-encoder reranking (applied to vector results)
        7. LLM answer synthesis
        
        Args:
            question: The question to answer
            namespace: Namespace/collection to search within
            max_results: Maximum number of graph elements to consider
            similarity_threshold: Similarity threshold for relevance
            llm: LLM instance for answer synthesis
            vector_weight: Weight for vector similarity results
            keyword_weight: Weight for keyword search results
            graph_weight: Weight for graph expansion results
            query_type: Type of query ("exploratory", "technical", "relational")
            use_reranking: Whether to use cross-encoder reranking
            max_expansion_nodes: Maximum nodes to expand from seed nodes
            engine: Engine configuration for vector store access
            vector_store_repo: Vector store repository instance
            llm_embeddings: Embedding model for vector store access (REQUIRED)
            chat_history_dict: chatHistory
            
        Returns:
            Dictionary with answer, graph elements, and retrieval details
        """
        logger.info(f"Advanced Graph QA for question: {question}, namespace: {namespace}")
        logger.info(f"Parameters - vector_weight: {vector_weight}, keyword_weight: {keyword_weight}, graph_weight: {graph_weight}, query_type: {query_type}")
        logger.info(f"Engine config: name={engine.name if engine else 'None'}, index_name={engine.index_name if engine else 'None'}, host={engine.host if engine else 'None'}")
        
        # Determine query type if not provided
        if not query_type:
            query_type = self._detect_query_type(question)
            logger.info(f"Detected query type: {query_type}")
        
        # Adjust weights based on query type
        adjusted_weights = self._adjust_weights_by_query_type(vector_weight, keyword_weight, graph_weight, query_type)
        
        # Validate vector store dependencies
        if engine is None:
            raise ValueError("Engine configuration is required for vector store retrieval.")
        if vector_store_repo is None:
            raise ValueError("Vector store repository is required for retrieval.")
        if llm_embeddings is None:
            logger.error("Embeddings must be provided for vector store retrieval!")
            raise ValueError("Embeddings must be provided for vector store retrieval.")
        
        # Ensure graph repository is available
        self.graph_service._ensure_repository()
        graph_repo = self.graph_service._get_repository()
        logger.info(f"Vector store repo type: {type(vector_store_repo).__name__}")
        logger.info(f"Embeddings available: {llm_embeddings is not None}")
        
        # 1. Global Search (Community-based)
        # Perform global search if query is exploratory or graph weight is high
        community_reports = []
        if query_type == "exploratory" or adjusted_weights["graph"] > 1.0:
            logger.info("Performing Global Search (Community-based)")
            try:
                # Search for community reports
                reports = graph_repo.search_community_reports(question, limit=3)
                if reports:
                    logger.info(f"Found {len(reports)} community reports")
                    community_reports = reports
                else:
                    logger.info("No community reports found")
            except Exception as e:
                logger.warning(f"Global search failed: {e}")

        # 2. & 3. Parallel Retrieval & RRF (Handled by vector_store_repo.perform_hybrid_search)
        
        # Create a QuestionAnswer object for vector store search
        from tilellm.models import QuestionAnswer
        
        # Calculate alpha based on vector and keyword weights
        total_weight = adjusted_weights["vector"] + adjusted_weights["keyword"]
        alpha = adjusted_weights["vector"] / total_weight if total_weight > 0 else 0.5
        
        # Determine search type based on weights
        if adjusted_weights["vector"] > 0 and adjusted_weights["keyword"] > 0:
            search_type = "hybrid"
        elif adjusted_weights["vector"] > 0:
            search_type = "similarity"
        else:
            search_type = "keyword"
        
        # Prepare values for QuestionAnswer
        max_results_val = max_results if max_results is not None else 20
        similarity_threshold_val = similarity_threshold if similarity_threshold is not None else 0.3
        
        # Determine embedding model name from embeddings if available
        embedding_model = "text-embedding-ada-002"  # default
        if llm_embeddings is not None:
            if hasattr(llm_embeddings, 'model_name'):
                embedding_model = llm_embeddings.model_name
            elif hasattr(llm_embeddings, 'model'):
                embedding_model = llm_embeddings.model
        logger.info(f"Using embedding model: {embedding_model}")
        
        # Build QuestionAnswer object
        question_answer = QuestionAnswer(
            question=question,
            namespace=namespace,
            engine=engine,
            top_k=max_results_val,
            alpha=alpha,
            search_type=search_type,
            # Set defaults for other required fields
            llm="openai",  # placeholder, not used for retrieval
            model="gpt-3.5-turbo",
            embedding=embedding_model,
            sparse_encoder="splade",
            temperature=0.0,
            max_tokens=512,
            similarity_threshold=similarity_threshold_val,
            debug=False,
            citations=False,
            reranking=use_reranking, # 5. Reranking (for documents)
            reranking_multiplier=3,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            contextualize_prompt=False,
            chat_history_dict=chat_history_dict if chat_history_dict else None
        )
        
        # Initialize embeddings and index via repository
        try:
            emb_dimension, sparse_encoder, index = await vector_store_repo.initialize_embeddings_and_index(
                question_answer, llm_embeddings, embedding_dimension, None
            )
            
            # Fetch vectors for the question
            from tilellm.controller.controller_utils import fetch_question_vectors
            dense_vector, sparse_vector = await fetch_question_vectors(question_answer, sparse_encoder, llm_embeddings)
            print("===============================>dense")
            # Perform hybrid search (or similarity/keyword based on search_type)
            logger.info(f"Performing hybrid search with namespace: {namespace}, engine: {engine}")
            search_results = await vector_store_repo.perform_hybrid_search(
                question_answer, index, dense_vector, sparse_vector
            )
            print(f"===============================>dense 12 {search_results}")
            #logger.info(f"Search results keys: {list(search_results.keys()) if search_results else 'None'}")
            
            # Extract chunk IDs from search results
            chunk_ids = []
            chunk_texts = []
            if search_results and "matches" in search_results:
                logger.info(f"Number of matches: {len(search_results['matches'])}")
                if search_results["matches"]:
                    logger.info(f"First match type: {type(search_results['matches'][0])}")
                for i, match in enumerate(search_results["matches"]):
                    # Robust extraction for both dict and object types
                    m_id = None
                    if isinstance(match, dict):
                        m_id = match.get("id")
                        metadata = match.get("metadata", {})
                        m_text = metadata.get("text")
                        logger.info(f"Match {i}: dict id={m_id}, metadata keys={list(metadata.keys()) if metadata else 'None'}")
                    else:
                        m_id = getattr(match, "id", None)
                        metadata = getattr(match, "metadata", {})
                        m_text = metadata.get("text") if isinstance(metadata, dict) else getattr(metadata, "text", None)
                        logger.info(f"Match {i}: object id={m_id}, metadata type={type(metadata)}")
                    
                    if m_id:
                        chunk_ids.append(m_id)
                    if m_text:
                        chunk_texts.append(m_text)
            
            logger.info(f"Retrieved {len(chunk_ids)} chunks from vector store")
            logger.info(f"All chunk IDs: {chunk_ids}")
            logger.info(f"Chunk IDs sample (first 5): {chunk_ids[:5]}")
            
            # 4. Graph Expansion
            # Find seed nodes in Neo4j associated with these chunk IDs
            seed_node_ids = []
            if chunk_ids:
                # Query Neo4j for nodes associated with the retrieved chunk IDs
                logger.info(f"Looking for seed nodes in Neo4j for {len(chunk_ids[:10])} chunk IDs")
                logger.info(f"Chunk IDs sample: {chunk_ids[:5]}")
                for chunk_id in chunk_ids[:10]:  # Limit to top results to avoid overhead
                    logger.info(f"Searching for nodes with source_id/chunk_id: {chunk_id}")
                    nodes = graph_repo.find_nodes_by_source_id(chunk_id, limit=5, namespace=namespace, index_name=engine.index_name)
                    if nodes:
                        logger.info(f"Found {len(nodes)} nodes for chunk {chunk_id}")
                        for node in nodes:
                            logger.info(f"  Node ID: {node.id}, label: {node.label}, source_ids: {node.properties.get('source_ids', 'N/A')}, chunk_id: {node.properties.get('chunk_id', 'N/A')}")
                            seed_node_ids.append(node.id)
                    else:
                        logger.info(f"No nodes found for chunk {chunk_id}")
            
            # Also use community reports as seed nodes if available
            for report in community_reports:
                seed_node_ids.append(report.id)

            # Deduplicate seed node IDs
            seed_node_ids = list(set(seed_node_ids))

            # Expand graph from seed nodes
            expanded_graph = {"nodes": [], "relationships": []}
            if seed_node_ids and adjusted_weights["graph"] > 0:
                # Scale expansion based on graph weight
                expansion_limit = int(max_expansion_nodes * adjusted_weights["graph"])
                expanded_graph = graph_repo.expand_from_nodes(
                    node_ids=seed_node_ids,
                    max_hops=1,  # Direct connections for now
                    limit=expansion_limit,
                    namespace=namespace,
                    index_name=engine.index_name
                )
                logger.info(f"Expanded graph: {len(expanded_graph['nodes'])} nodes, {len(expanded_graph['relationships'])} relationships")
            
            # 5. Final Reranking (Cross-Encoder)
            # Collect all pieces of information as Document objects for reranking
            all_context_docs = []
            
            # Community reports
            for report in community_reports:
                content = report.properties.get("report") or report.properties.get("summary") or report.properties.get("title", "")
                all_context_docs.append(Document(page_content=f"[Community Report] {content}", metadata={"source": "graph_community"}))
            
            # Vector chunks
            for text in chunk_texts:
                all_context_docs.append(Document(page_content=f"[Document Chunk] {text}", metadata={"source": "vector_store"}))
            
            # Graph nodes
            for node in expanded_graph["nodes"]:
                name = node.get("properties", {}).get("name") or node.get("properties", {}).get("title") or "Unnamed"
                desc = node.get("properties", {}).get("description") or ""
                label = node.get("label", "Entity")
                all_context_docs.append(Document(page_content=f"[Graph Node] {label}: {name}. {desc}", metadata={"source": "graph_node"}))
                
            # Graph relationships
            for rel in expanded_graph["relationships"]:
                rel_type = rel.get("type", "RELATED")
                source_id = rel.get("source_id", "")
                target_id = rel.get("target_id", "")
                # Find node names for IDs
                source_node = next((n for n in expanded_graph["nodes"] if n.get("id") == source_id), None)
                target_node = next((n for n in expanded_graph["nodes"] if n.get("id") == target_id), None)
                source_name = source_node.get("properties", {}).get("name", "Unknown") if source_node else "Unknown"
                target_name = target_node.get("properties", {}).get("name", "Unknown") if target_node else "Unknown"
                desc = rel.get("properties", {}).get("description", "")
                all_context_docs.append(Document(page_content=f"[Graph Relationship] {source_name} --[{rel_type}]--> {target_name}. {desc}", metadata={"source": "graph_relationship"}))

            # Apply final reranking if enabled
            context_parts = []
            if use_reranking and all_context_docs:
                logger.info(f"Applying final reranking to {len(all_context_docs)} context items")
                try:
                    reranker = TileReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
                    # Rerank and keep top_k (e.g., 20)
                    reranked_docs = reranker.rerank_documents(question, all_context_docs, top_k=20)
                    context_parts = [doc.page_content for doc in reranked_docs]
                except Exception as e:
                    logger.error(f"Final reranking failed: {e}")
                    # Fallback to non-reranked context
                    context_parts = [doc.page_content for doc in all_context_docs[:20]]
            else:
                # Use default formatting if no reranking
                # Add community reports
                if community_reports:
                    context_parts.append("Global Context (Community Reports):")
                    for i, report in enumerate(community_reports):
                        content = report.properties.get("report") or report.properties.get("summary") or report.properties.get("title", "")
                        context_parts.append(f"{i+1}. {content[:500]}...")
                    context_parts.append("\n")

                # Add chunk texts as document context
                if chunk_texts:
                    context_parts.append("Relevant Document Excerpts:")
                    for i, text in enumerate(chunk_texts[:5]):
                        context_parts.append(f"{i+1}. {text[:300]}...")
                
                # Add graph context
                if expanded_graph["nodes"]:
                    context_parts.append("\nKnowledge Graph Context:")
                    # Group nodes by label
                    nodes_by_label = {}
                    for node in expanded_graph["nodes"]:
                        label = node.get("label", "Unknown")
                        name = node.get("properties", {}).get("name", node.get("properties", {}).get("title", "Unnamed"))
                        nodes_by_label.setdefault(label, []).append(name)
                    
                    for label, names in nodes_by_label.items():
                        context_parts.append(f"{label}: {', '.join(names[:5])}")
                    
                    # Add relationships
                    if expanded_graph["relationships"]:
                        context_parts.append("\nRelationships:")
                        for rel in expanded_graph["relationships"][:5]:
                            rel_type = rel.get("type", "RELATED")
                            source_id = rel.get("source_id", "")
                            target_id = rel.get("target_id", "")
                            # Find node names for IDs
                            source_node = next((n for n in expanded_graph["nodes"] if n.get("id") == source_id), None)
                            target_node = next((n for n in expanded_graph["nodes"] if n.get("id") == target_id), None)
                            source_name = source_node.get("properties", {}).get("name", "Unknown") if source_node else "Unknown"
                            target_name = target_node.get("properties", {}).get("name", "Unknown") if target_node else "Unknown"
                            context_parts.append(f"{source_name} --[{rel_type}]--> {target_name}")
            
            # Generate answer using LLM if available
            answer = ""
            if llm is not None:
                try:
                    prompt = f"""Based on the following information, answer the question.

                            Question: {question}
                        
                            {"\n".join(context_parts) if context_parts else "No relevant information found."}
                            
                            Provide a concise and accurate answer based on the available information. If the information is insufficient, say so.
                            Use This chat history {chat_history_dict} to answer the question.
                            """
                    
                    if hasattr(llm, 'invoke'):
                        from langchain_core.messages import HumanMessage, SystemMessage
                        messages = [
                            SystemMessage(content="You are a helpful assistant that answers questions based on knowledge graph and document information."),
                            HumanMessage(content=prompt)
                        ]
                        response = await llm.ainvoke(messages)
                        answer = response.content if hasattr(response, 'content') else str(response)
                    elif hasattr(llm, 'chat'):
                        answer = await llm.chat(
                            system="You are a helpful assistant that answers questions based on knowledge graph and document information.",
                            messages=[{"role": "user", "content": prompt}]
                        )
                    else:
                        answer = await llm(prompt)
                    
                    logger.info(f"Generated answer using LLM: {answer[:100]}...")
                    
                except Exception as e:
                    logger.error(f"Error generating answer with LLM: {e}")
                    answer = f"I found {len(chunk_ids)} document chunks and {len(expanded_graph['nodes'])} graph nodes. LLM synthesis failed: {str(e)}"
            else:
                answer = f"I found {len(chunk_ids)} document chunks and {len(expanded_graph['nodes'])} graph nodes. Provide LLM configuration for detailed answer synthesis."
            
            # Prepare response
            enhanced_result = {
                "answer": answer,
                "entities": [{"id": n["id"], "label": n["label"], "properties": n["properties"]} for n in expanded_graph["nodes"]],
                "relationships": [{"id": r["id"], "type": r["type"], "properties": r["properties"], "source_id": r["source_id"], "target_id": r["target_id"]} for r in expanded_graph["relationships"]],
                "query_used": question,
                "retrieval_strategy": f"hybrid_{query_type}" if query_type else "hybrid",
                "scores": {
                    "vector_weight": adjusted_weights["vector"],
                    "keyword_weight": adjusted_weights["keyword"],
                    "graph_weight": adjusted_weights["graph"],
                    "query_type": query_type,
                    "alpha": alpha,
                    "chunks_retrieved": len(chunk_ids),
                    "seed_nodes": len(seed_node_ids),
                    "community_reports": len(community_reports)
                },
                "expanded_nodes": expanded_graph["nodes"],
                "expanded_relationships": expanded_graph["relationships"]
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in advanced GraphRAG pipeline: {e}")
            # Fall back to basic graph search
            logger.info("Falling back to basic graph search due to error")
            basic_result = await self.answer_with_graph(
                question=question,
                namespace=namespace,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                llm=llm
            )
            enhanced_result = {
                **basic_result,
                "retrieval_strategy": f"hybrid_{query_type}" if query_type else "hybrid",
                "scores": {
                    "vector_weight": adjusted_weights["vector"],
                    "keyword_weight": adjusted_weights["keyword"],
                    "graph_weight": adjusted_weights["graph"],
                    "query_type": query_type
                },
                "expanded_nodes": [],
                "expanded_relationships": []
            }
            return enhanced_result
    
    def _detect_query_type(self, question: str) -> str:
        """
        Detect query type based on question content.
        Returns "exploratory", "technical", or "relational".
        """
        question_lower = question.lower()
        
        # Exploratory queries: broad, asking about concepts, "tell me about", "what is"
        exploratory_phrases = ["tell me about", "what is", "what are", "explain", "describe", "overview", "introduction"]
        if any(phrase in question_lower for phrase in exploratory_phrases):
            return "exploratory"
        
        # Technical queries: specific errors, codes, technical details
        technical_phrases = ["error", "code", "bug", "issue", "fix", "problem", "how to", "tutorial", "step by step"]
        if any(phrase in question_lower for phrase in technical_phrases):
            return "technical"
        
        # Relational queries: connections, relationships, influence, impact
        relational_phrases = ["relation", "relationship", "connect", "influence", "impact", "affect", "correlate", "between", "vs", "versus"]
        if any(phrase in question_lower for phrase in relational_phrases):
            return "relational"
        
        # Default to exploratory for general questions
        return "exploratory"
    
    def _adjust_weights_by_query_type(self, vector_weight: float, keyword_weight: float, 
                                      graph_weight: float, query_type: str) -> Dict[str, float]:
        """
        Adjust weights based on query type according to the balancing matrix.
        """
        # Base weights from user input
        weights = {
            "vector": vector_weight,
            "keyword": keyword_weight,
            "graph": graph_weight
        }
        
        # Apply query type adjustments
        if query_type == "exploratory":
            weights["vector"] *= 1.5  # High
            weights["keyword"] *= 0.5  # Low
            weights["graph"] *= 1.0   # Medium
        elif query_type == "technical":
            weights["vector"] *= 0.5  # Low
            weights["keyword"] *= 1.5  # High
            weights["graph"] *= 0.5   # Low
        elif query_type == "relational":
            weights["vector"] *= 1.0  # Medium
            weights["keyword"] *= 1.0  # Medium
            weights["graph"] *= 2.0   # Very High
        
        return weights

    async def cluster_graph(self, llm=None, level: int = 0, namespace: Optional[str] = None, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect communities in the graph and generate reports for each.
        
        Args:
            llm: LLM instance for report generation
            level: The hierarchy level
            namespace: Optional namespace to filter graph partition
            index_name: Optional index_name to filter graph partition
            
        Returns:
            Statistics about the clustering process
        """
        llm_to_use = llm or self.llm
        if not llm_to_use:
            raise ValueError("LLM configuration is required for community report generation.")
            
        repo = self.graph_service._get_repository()
        cluster_service = ClusterService(repository=repo, llm=llm_to_use)
        
        return await cluster_service.perform_clustering(level=level, namespace=namespace, index_name=index_name)
