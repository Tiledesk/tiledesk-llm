"""
Service layer for Knowledge Graph operations.
Handles business logic and orchestrates repository operations.
"""

import json
import logging
from typing import Optional, List, Dict, Any, cast
from datetime import datetime
import asyncio
import duckdb
from pathlib import Path
import tempfile
import shutil
import os

from tilellm.modules.knowledge_graph.models import Node, NodeUpdate, Relationship, RelationshipUpdate
from ..tools.extraction_prompts import get_extraction_config
from tilellm.store.graph import BaseGraphRepository
from tilellm.models import Engine
from tilellm.models.schemas import RepositoryItems
from tilellm.tools.reranker import TileReranker
from langchain_core.documents import Document
from .clustering import ClusterService  # type: ignore
from .minio_storage import MinIOStorageService

from ..utils import (
    GRAPH_QA_PROMPT_TEMPLATE,
    GRAPH_QA_SYSTEM_PROMPT,
    ADVANCED_GRAPH_QA_PROMPT_TEMPLATE,
    ADVANCED_GRAPH_QA_SYSTEM_PROMPT,
    format_chat_history,
    format_graph_context,
    format_community_reports,
    format_document_excerpts,
    GraphExpander
)
from ..utils.query_analysis import (
    detect_query_type_with_llm,
    detect_query_type_heuristic,
    apply_weight_adjustments
)

GRAPHRAG_AVAILABLE = False
GraphRAGExtractor = None

logger = logging.getLogger(__name__)

try:
    from tilellm.modules.knowledge_graph_falkor.tools.graphrag_extractor import GraphRAGExtractor
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

    def __init__(self, repository: Optional[BaseGraphRepository] = None):
        """
        Initialize the service with a repository.

        Args:
            repository: GraphRepository instance. If None, needs to be set later.
        """
        self.repository = repository

    def set_repository(self, repository: BaseGraphRepository):
        """Set the repository instance"""
        self.repository = repository

    def _ensure_repository(self):
        """Ensure repository is initialized"""
        if not self.repository:
            raise RuntimeError("Repository not initialized. Call set_repository first.")

    def _get_repository(self) -> BaseGraphRepository:
        """Get the repository instance, ensuring it's initialized"""
        self._ensure_repository()
        return cast(BaseGraphRepository, self.repository)
        #return self.repository


    # ==================== NODE OPERATIONS ====================

    async def create_node(self, node: Node, namespace: Optional[str] = None, index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None, metadata_id: Optional[str] = None, graph_name: Optional[str] = None) -> Node:
        """
        Create a new node with business logic validation.

        Args:
            node: Node to create
            namespace: Optional namespace to partition the graph
            index_name: Optional index_name (collection name) to partition the graph
            engine_name: Optional engine name (e.g., 'pinecone', 'qdrant') to partition the graph
            engine_type: Optional engine type (e.g., 'pod', 'serverless')
            metadata_id: Optional metadata ID from vector store chunks for cleanup
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

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

        logger.info(f"Creating node with label: {node.label}, namespace: {namespace}, index_name: {index_name}, engine_name: {engine_name}, engine_type: {engine_type}, metadata_id: {metadata_id}, graph_name: {graph_name}")
        return await self._get_repository().create_node(node, namespace=namespace, index_name=index_name, engine_name=engine_name, engine_type=engine_type, metadata_id=metadata_id, graph_name=graph_name)

    async def get_node(self, node_id: str, namespace: Optional[str] = None, graph_name: Optional[str] = None) -> Optional[Node]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID
            namespace: Optional namespace to search in (for multi-graph isolation)
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            Node if found, None otherwise
        """
        self._ensure_repository()
        logger.info(f"Retrieving node with ID: {node_id}, namespace: {namespace}, graph_name: {graph_name}")
        return await self._get_repository().find_node_by_id(node_id, namespace=namespace, graph_name=graph_name)

    async def get_nodes_by_label(self, label: str, limit: int = 100, namespace: Optional[str] = None, index_name: Optional[str] = None, graph_name: Optional[str] = None) -> List[Node]:
        """
        Get all nodes with a specific label.

        Args:
            label: Node label
            limit: Maximum number of nodes to return
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            List of nodes
        """
        self._ensure_repository()
        logger.info(f"Retrieving nodes with label: {label}, limit: {limit}, namespace: {namespace}, index_name: {index_name}, graph_name: {graph_name}")
        return await self._get_repository().find_nodes_by_label(label, limit, namespace=namespace, index_name=index_name, graph_name=graph_name)

    async def search_nodes(self, label: str, property_key: str, property_value: Any, limit: int = 100, namespace: Optional[str] = None, index_name: Optional[str] = None, graph_name: Optional[str] = None) -> List[Node]:
        """
        Search nodes by property value.

        Args:
            label: Node label
            property_key: Property key to search
            property_value: Property value to match
            limit: Maximum number of nodes to return
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            List of matching nodes
        """
        self._ensure_repository()
        logger.info(f"Searching nodes with {label}.{property_key} = {property_value}, namespace: {namespace}, index_name: {index_name}, graph_name: {graph_name}")
        return await self._get_repository().find_nodes_by_property(label, property_key, property_value, limit, namespace=namespace, index_name=index_name, graph_name=graph_name)

    async def find_entity_by_name_and_type(self, entity_name: str, entity_type: str, namespace: Optional[str] = None, index_name: Optional[str] = None, graph_name: Optional[str] = None) -> Optional[Node]:
        """
        Find an existing entity node by name and type for deduplication.

        Args:
            entity_name: Entity name to search for
            entity_type: Entity type/label
            namespace: Optional namespace to filter
            index_name: Optional index_name to filter
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            First matching node if found, None otherwise
        """
        matching_nodes = await self.search_nodes(
            label=entity_type,
            property_key="name",
            property_value=entity_name,
            limit=1,
            namespace=namespace,
            index_name=index_name,
            graph_name=graph_name
        )
        return matching_nodes[0] if matching_nodes else None

    async def update_node(self, node_id: str, node_update: NodeUpdate, namespace: Optional[str] = None, graph_name: Optional[str] = None) -> Optional[Node]:
        """
        Update a node's properties.

        Args:
            node_id: Node ID
            node_update: Update data
            namespace: Optional namespace where the node exists (for multi-graph isolation)
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            Updated node if found, None otherwise

        Raises:
            ValueError: If validation fails
        """
        self._ensure_repository()

        # Check if node exists
        existing_node = await self._get_repository().find_node_by_id(node_id, namespace=namespace, graph_name=graph_name)
        if not existing_node:
            logger.warning(f"Node with ID {node_id} not found for update")
            return None

        # Validate label if provided
        if node_update.label and not node_update.label[0].isupper():
            logger.warning(f"Node label '{node_update.label}' should start with uppercase for Neo4j convention.")

        logger.info(f"Updating node with ID: {node_id}, namespace: {namespace}, graph_name: {graph_name}")
        return await self._get_repository().update_node(
            node_id=node_id,
            label=node_update.label,
            properties=node_update.properties,
            namespace=namespace,
            graph_name=graph_name
        )

    async def delete_node(self, node_id: str, detach: bool = True, namespace: Optional[str] = None, graph_name: Optional[str] = None) -> bool:
        """
        Delete a node.

        Args:
            node_id: Node ID
            detach: If True, also delete all relationships
            namespace: Optional namespace where the node exists (for multi-graph isolation)
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            True if deleted, False if not found
        """
        self._ensure_repository()
        logger.info(f"Deleting node with ID: {node_id}, detach: {detach}, namespace: {namespace}, graph_name: {graph_name}")

        # Check if node exists before deletion
        existing_node = await self._get_repository().find_node_by_id(node_id, namespace=namespace, graph_name=graph_name)
        if not existing_node:
            logger.warning(f"Node with ID {node_id} not found for deletion")
            return False

        return await self._get_repository().delete_node(node_id, detach, namespace=namespace, graph_name=graph_name)

    # ==================== RELATIONSHIP OPERATIONS ====================

    async def create_relationship(self, relationship: Relationship, namespace: Optional[str] = None, index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None, metadata_id: Optional[str] = None, graph_name: Optional[str] = None) -> Relationship:
        """
        Create a relationship between nodes.

        Args:
            relationship: Relationship to create
            namespace: Optional namespace to partition the graph
            index_name: Optional index_name (collection name) to partition the graph
            engine_name: Optional engine name (e.g., 'pinecone', 'qdrant') to partition the graph
            engine_type: Optional engine type (e.g., 'pod', 'serverless')
            metadata_id: Optional metadata ID from vector store chunks for cleanup
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

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

        # Verify source and target nodes exist (with namespace for multi-graph isolation)
        source_node = await self._get_repository().find_node_by_id(relationship.source_id, namespace=namespace, graph_name=graph_name)
        if not source_node:
            raise RuntimeError(f"Source node with ID '{relationship.source_id}' does not exist.")

        target_node = await self._get_repository().find_node_by_id(relationship.target_id, namespace=namespace, graph_name=graph_name)
        if not target_node:
            raise RuntimeError(f"Target node with ID '{relationship.target_id}' does not exist.")

        logger.info(f"Creating relationship: {relationship.source_id} -[{relationship.type}]-> {relationship.target_id}")
        return await self._get_repository().create_relationship(relationship, namespace=namespace, index_name=index_name, engine_name=engine_name, engine_type=engine_type, metadata_id=metadata_id, graph_name=graph_name)

    async def get_relationship(self, relationship_id: str, namespace: Optional[str] = None, graph_name: Optional[str] = None) -> Optional[Relationship]:
        """
        Get a relationship by ID.

        Args:
            relationship_id: Relationship ID
            namespace: Optional namespace to search in (for multi-graph isolation)
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            Relationship if found, None otherwise
        """
        self._ensure_repository()
        logger.info(f"Retrieving relationship with ID: {relationship_id}, namespace: {namespace}, graph_name: {graph_name}")
        return await self._get_repository().find_relationship_by_id(relationship_id, namespace=namespace, graph_name=graph_name)

    async def get_node_relationships(self, node_id: str, direction: str = "both", namespace: Optional[str] = None, graph_name: Optional[str] = None) -> List[Relationship]:
        """
        Get all relationships for a node.

        Args:
            node_id: Node ID
            direction: "incoming", "outgoing", or "both"
            namespace: Optional namespace to search in (for multi-graph isolation)
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            List of relationships

        Raises:
            ValueError: If direction is invalid
        """
        self._ensure_repository()

        if direction not in ["incoming", "outgoing", "both"]:
            raise ValueError(f"Invalid direction: {direction}. Must be 'incoming', 'outgoing', or 'both'.")

        logger.info(f"Retrieving {direction} relationships for node: {node_id}, namespace: {namespace}, graph_name: {graph_name}")
        return await self._get_repository().find_relationships_by_node(node_id, direction, namespace=namespace, graph_name=graph_name)

    async def update_relationship(self, relationship_id: str, relationship_update: RelationshipUpdate, namespace: Optional[str] = None, graph_name: Optional[str] = None) -> Optional[Relationship]:
        """
        Update a relationship.

        Args:
            relationship_id: Relationship ID
            relationship_update: Update data
            namespace: Optional namespace where the relationship exists (for multi-graph isolation)
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            Updated relationship if found, None otherwise

        Raises:
            ValueError: If validation fails
        """
        self._ensure_repository()

        # Check if relationship exists
        existing_rel = await self._get_repository().find_relationship_by_id(relationship_id, namespace=namespace, graph_name=graph_name)
        if not existing_rel:
            logger.warning(f"Relationship with ID {relationship_id} not found for update")
            return None

        # Validate type if provided
        if relationship_update.type and not relationship_update.type.isupper():
            logger.warning(f"Relationship type '{relationship_update.type}' should be uppercase for Neo4j convention.")

        logger.info(f"Updating relationship with ID: {relationship_id}, namespace: {namespace}, graph_name: {graph_name}")
        return await self._get_repository().update_relationship(
            relationship_id=relationship_id,
            rel_type=relationship_update.type,
            properties=relationship_update.properties,
            namespace=namespace,
            graph_name=graph_name
        )

    async def delete_relationship(self, relationship_id: str, namespace: Optional[str] = None, graph_name: Optional[str] = None) -> bool:
        """
        Delete a relationship.

        Args:
            relationship_id: Relationship ID
            namespace: Optional namespace where the relationship exists (for multi-graph isolation)
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            True if deleted, False if not found
        """
        self._ensure_repository()
        logger.info(f"Deleting relationship with ID: {relationship_id}, namespace: {namespace}, graph_name: {graph_name}")

        # Check if relationship exists before deletion
        existing_rel = await self._get_repository().find_relationship_by_id(relationship_id, namespace=namespace, graph_name=graph_name)
        if not existing_rel:
            logger.warning(f"Relationship with ID {relationship_id} not found for deletion")
            return False

        return await self._get_repository().delete_relationship(relationship_id, namespace=namespace, graph_name=graph_name)

    # ==================== UTILITY OPERATIONS ====================

    async def verify_connection(self) -> bool:
        """
        Verify the connection to FalkorDB.

        Returns:
            True if connection is successful
        """
        self._ensure_repository()
        logger.info("Verifying FalkorDB connection")
        return await self._get_repository().verify_connection()

    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with node count, relationship count, etc.
        """
        self._ensure_repository()
        logger.info("Retrieving database statistics")
        return await self._get_repository().get_database_info()

    async def get_graph_network(
        self,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        node_limit: int = 1000,
        relationship_limit: int = 5000,
        node_labels: Optional[List[str]] = None,
        community: bool = False,
        graph_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve the graph network (nodes + relationships) for visualization.

        Args:
            namespace: Optional namespace to filter nodes/relationships
            index_name: Optional index_name to filter
            node_limit: Maximum number of nodes to return (default: 1000)
            relationship_limit: Maximum number of relationships to return (default: 5000)
            node_labels: Optional list of node labels to filter (e.g., ["PERSON", "ORGANIZATION"])
            community: If True, returns the community graph (BELONGS_TO_COMMUNITY)
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            Dictionary with:
            {
                "nodes": [{"id": str, "label": str, "properties": dict}, ...],
                "relationships": [{"id": str, "source_id": str, "target_id: str, "type": str, "properties": dict}, ...],
                "stats": {"node_count": int, "relationship_count": int, "filtered_by": {...}}
            }
        """
        self._ensure_repository()
        repo = self._get_repository()

        logger.info(f"Retrieving graph network - namespace: {namespace}, index_name: {index_name}, "
                   f"node_limit: {node_limit}, node_labels: {node_labels}, community: {community}, graph_name: {graph_name}")

        if community:
            # Check if namespace and index_name are provided, as they are required for community query
            if not namespace or not index_name:
                logger.warning("Namespace and index_name are required for community network query.")
                # Fall back to normal behavior or return empty?
                # User query requires them: where n.namespace=$namesapce and m.namespace=$namesapce and n.index_name=$index_name and m.index_name=$index_name

            result = await repo.get_community_network(
                namespace=namespace or "",
                index_name=index_name or "",
                limit=relationship_limit,
                graph_name=graph_name
            )

            result["stats"] = {
                "node_count": len(result["nodes"]),
                "relationship_count": len(result["relationships"]),
                "filtered_by": {
                    "namespace": namespace,
                    "index_name": index_name,
                    "community": True,
                    "graph_name": graph_name
                }
            }
            return result

        # Retrieve nodes
        nodes = []
        if node_labels:
            # Get nodes by specific labels
            for label in node_labels:
                label_nodes = await repo.find_nodes_by_label(
                    label=label,
                    limit=node_limit // len(node_labels),
                    namespace=namespace,
                    index_name=index_name,
                    graph_name=graph_name
                )
                nodes.extend(label_nodes)
        else:
            # Get all nodes (no label filter)
            all_labels = ["PERSON", "ORGANIZATION", "CATEGORY", "LOCATION", "EVENT", "ENTITY", "Document", "GEO"]
            for label in all_labels:
                try:
                    label_nodes = await repo.find_nodes_by_label(
                        label=label,
                        limit=node_limit // len(all_labels),
                        namespace=namespace,
                        index_name=index_name,
                        graph_name=graph_name
                    )
                    nodes.extend(label_nodes)
                except Exception as e:
                    logger.debug(f"No nodes found for label {label}: {e}")
                    continue

        # Limit total nodes
        nodes = nodes[:node_limit]

        # Get node IDs for relationship filtering
        node_ids = {node.id for node in nodes if node.id}

        # Retrieve relationships
        # For each node, get its relationships
        relationships = []
        for node in nodes:
            if not node.id:
                continue
            try:
                node_rels = await repo.find_relationships_by_node(
                    node_id=node.id,
                    direction="both",
                    namespace=namespace,
                    graph_name=graph_name
                )
                # Only include relationships where both source and target are in our node set
                for rel in node_rels:
                    if rel.source_id in node_ids and rel.target_id in node_ids:
                        relationships.append(rel)
                        if len(relationships) >= relationship_limit:
                            break
            except Exception as e:
                logger.debug(f"Error getting relationships for node {node.id}: {e}")
                continue

            if len(relationships) >= relationship_limit:
                break

        # Convert to dicts for JSON serialization
        nodes_data = []
        for node in nodes:
            nodes_data.append({
                "id": node.id,
                "label": node.label,
                "properties": node.properties or {}
            })

        relationships_data = []
        seen_rels = set()
        for rel in relationships:
            rel_key = (rel.source_id, rel.target_id, rel.type)
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                relationships_data.append({
                    "id": rel.id if rel.id else f"{rel.source_id}-{rel.target_id}",
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "type": rel.type,
                    "properties": rel.properties or {}
                })

        stats = {
            "node_count": len(nodes_data),
            "relationship_count": len(relationships_data),
            "filtered_by": {
                "namespace": namespace,
                "index_name": index_name,
                "node_labels": node_labels,
                "graph_name": graph_name
            }
        }

        logger.info(f"Retrieved {len(nodes_data)} nodes and {len(relationships_data)} relationships")

        return {
            "nodes": nodes_data,
            "relationships": relationships_data,
            "stats": stats
        }


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
        self.minio_storage = MinIOStorageService()

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

    async def import_from_vector_store(self, namespace: str, creation_prompt:str, vector_store_repo=None, engine: Optional[Engine] = None, limit: int = 100, llm=None, index_name: Optional[str] = None, overwrite: bool = True, engine_type: Optional[str] = None, graph_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Import documents from a vector store namespace into the knowledge graph.

        Steps:
        1. Retrieve all chunks from the vector store namespace
        2. Extract entities and relationships using GraphRAG extractor
        3. Store them as nodes and relationships in Neo4j

        Args:
            :param engine_type:
            :param overwrite:
            :param namespace: Namespace/collection name in vector store
            :param creation_prompt: Creation prompt
            :param vector_store_repo: Vector store repository instance (if None, uses self.vector_store_repository)
            :param engine: Engine configuration for vector store (required if repo doesn't store engine)
            :param limit: Maximum number of chunks to process
            :param llm: LLM instance for extraction (if None, uses self.llm)
            :param index_name: Optional index_name for graph partition (defaults to engine.index_name)
            :param graph_name: Optional explicit graph name (overrides namespace for graph selection)

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
        if llm_to_use is None and GRAPHRAG_AVAILABLE:
            logger.error("LLM not provided - GraphRAG extraction requires LLM for entity and relationship extraction.")
            raise ValueError("LLM configuration is required for GraphRAG extraction. Provide LLM instance with chat/invoke method.")
        self.graph_service._ensure_repository()
        index_name = index_name if index_name is not None else (engine.index_name if engine else None)
        engine_name = engine.name if engine else None
        engine_type = engine_type if engine_type is not None else (engine.type if engine else None)

        # Determine graph name to use (fallback to namespace if not provided)
        graph_name_to_use = graph_name if graph_name is not None else namespace

        logger.info(f"Importing from vector store namespace: {namespace}, limit: {limit}, index_name: {index_name}")

        # Get repository instance for index creation and cleanup
        graph_repo = self.graph_service._get_repository()
        
        # Clean up existing nodes if overwrite is True
        if overwrite:
            await graph_repo.delete_nodes_by_metadata(namespace=namespace, index_name=index_name, engine_name=engine_name, engine_type=engine_type, graph_name=graph_name_to_use)

        # Ensure indexes for the graph using entity types from extraction config
        config = get_extraction_config(creation_prompt)
        node_labels = config.entity_types
        await graph_repo.ensure_indexes_for_graph(graph_name=graph_name_to_use, node_labels=node_labels)

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
            
            # Extract text and IDs from chunks, and build chunk_id to metadata_id mapping
            chunks_data = []
            chunk_id_to_metadata_id = {}

            for chunk in chunks_to_process:
                chunk_id = getattr(chunk, 'id', 'unknown')
                chunk_text = getattr(chunk, 'text', '')
                
                # Extract metadata_id from chunk (RepositoryQueryResult has metadata_id attribute)
                metadata_id = getattr(chunk, 'metadata_id', None)
                if metadata_id is None:
                    # Fall back to metadata dict
                    metadata = getattr(chunk, 'metadata', {}) or {}
                    if isinstance(metadata, dict):
                        metadata_id = metadata.get('id')
                if metadata_id is None:
                    metadata_id = chunk_id
                # Ensure metadata_id is string and not empty
                if not isinstance(metadata_id, str):
                    metadata_id = str(metadata_id)
                if not metadata_id or metadata_id.strip() == '':
                    metadata_id = 'unknown'
                
                chunk_id_to_metadata_id[chunk_id] = metadata_id
                
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
                    extractor = GraphRAGExtractor(llm_invoker=llm_to_use, creation_prompt=creation_prompt)

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
                            
                            # Determine metadata_id for entity based on source chunks
                            entity_metadata_id = None
                            if source_ids and isinstance(source_ids, list) and len(source_ids) > 0:
                                first_source_id = source_ids[0]
                                entity_metadata_id = chunk_id_to_metadata_id.get(first_source_id)
                            # If no metadata_id found, use 'unknown'
                            if entity_metadata_id is None:
                                entity_metadata_id = 'unknown'
                            
                            # Check if entity already exists (by name and type)
                            # For simplicity, create new node each time (could be deduplicated later)
                            node = Node(
                                label=entity_type,
                                properties={
                                    "name": entity_name,
                                    "description": description,
                                    "source_ids": source_ids if source_ids else [],
                                    "entity_type": entity_type,
                                    "import_timestamp": datetime.now().isoformat(),
                                    "engine_name": engine_name,
                                    "engine_type": engine_type,
                                    "metadata_id": entity_metadata_id  # Also store in properties for consistency
                                }
                            )
                            created_node = await self.graph_service.create_node(node,
                                                                           namespace=namespace,
                                                                           index_name=index_name,
                                                                           engine_name=engine_name,
                                                                           engine_type=engine_type,
                                                                           metadata_id=entity_metadata_id,
                                                                           graph_name=graph_name_to_use)
                            if created_node.id is None:
                                logger.error("Created entity node has no ID, skipping")
                                continue
                            
                            entity_node_map[entity_name] = created_node.id
                            nodes_created += 1
                            
                        except Exception as ex:
                            logger.error(f"Error creating entity node {entity.get('entity_name', 'unknown')}: {ex}")
                            continue
                    
                    # Create relationships between entities
                    for rel in relationships:
                        try:
                            source_name = rel.get("src_id", "").strip()
                            target_name = rel.get("tgt_id", "").strip()
                            # Use extracted relationship_type, fallback to RELATED_TO
                            rel_type = rel.get("relationship_type", "RELATED_TO").upper()
                            weight = rel.get("weight", 1.0)
                            description = rel.get("description", "")
                            source_ids = rel.get("source_id", [])

                            # DEBUG: Log what type was extracted
                            logger.info(f"Creating relationship: {source_name} -[{rel_type}]-> {target_name} (extracted from: {rel.get('relationship_type', 'NOT_FOUND')})")

                            if not source_name or not target_name:
                                continue
                            
                            source_node_id = entity_node_map.get(source_name)
                            target_node_id = entity_node_map.get(target_name)
                            
                            if not source_node_id or not target_node_id:
                                logger.debug(f"Missing source or target node for relationship {source_name} -> {target_name}")
                                continue
                            
                            # Determine metadata_id for relationship based on source chunks
                            rel_metadata_id = None
                            if source_ids and isinstance(source_ids, list) and len(source_ids) > 0:
                                first_source_id = source_ids[0]
                                rel_metadata_id = chunk_id_to_metadata_id.get(first_source_id)
                            # If no metadata_id found, use None (will not be added to properties)
                            
                            rel_properties = {
                                "weight": weight,
                                "description": description,
                                "source_ids": source_ids if source_ids else [],
                                "source_entity": source_name,
                                "target_entity": target_name,
                                "engine_name": engine_name,
                                "engine_type": engine_type
                            }
                            if rel_metadata_id is not None:
                                rel_properties["metadata_id"] = rel_metadata_id
                            
                            relationship = Relationship(
                                source_id=source_node_id,
                                target_id=target_node_id,
                                type=rel_type,
                                properties=rel_properties
                            )
                            await self.graph_service.create_relationship(relationship, namespace=namespace, index_name=index_name, engine_name=engine_name, engine_type=engine_type, metadata_id=rel_metadata_id, graph_name=graph_name_to_use)
                            relationships_created += 1
                            
                        except Exception as ex:
                            logger.error(f"Error creating relationship {rel.get('src_id', 'unknown')} -> {rel.get('tgt_id', 'unknown')}: {ex}")
                            continue
                    
                    logger.info(f"Created {nodes_created} entity nodes and {relationships_created} relationships")
                    
                except Exception as e:
                    logger.error(f"GraphRAG extraction failed: {e}")
                    # Fall back to simple document nodes
                    logger.info("Falling back to simple document nodes")
                    return await self._create_simple_document_nodes(chunks_to_process, namespace, index_name=index_name, engine_name=engine_name, engine_type=engine_type, graph_name=graph_name_to_use)
            else:
                # No LLM available, create simple document nodes
                logger.info("LLM not available for GraphRAG, creating simple document nodes")
                return await self._create_simple_document_nodes(chunks_to_process, namespace, index_name=index_name, engine_name=engine_name, engine_type=engine_type, graph_name=graph_name_to_use)
            
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

    async def _create_simple_document_nodes(self, chunks, namespace: str, index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None, graph_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create simple Document nodes for chunks when GraphRAG extraction is not possible.
        
        Args:
            chunks: List of chunk objects from vector store
            namespace: Namespace/collection name
            index_name: Optional index_name for graph partition
            engine_name: Optional engine name (e.g., 'pinecone', 'qdrant') for graph partition
            engine_type: Optional engine type (e.g., 'pod', 'serverless')
            graph_name: Optional graph name (overrides namespace for graph selection)
            
        Returns:
            Dictionary with import statistics
        """
        nodes_created = 0
        for chunk in chunks:
            chunk_id = getattr(chunk, 'id', 'unknown') or 'unknown'
            chunk_text = getattr(chunk, 'text', '') or ''
            metadata = getattr(chunk, 'metadata', {}) or {}
            try:
                # Extract metadata fields - handle both RepositoryQueryResult objects and dict metadata
                # First try to get metadata_id from chunk attributes (RepositoryQueryResult has metadata_id)
                metadata_id = getattr(chunk, 'metadata_id', None)
                if metadata_id is None:
                    # Fall back to metadata dict
                    if isinstance(metadata, dict):
                        metadata_id = metadata.get('id')
                    if metadata_id is None:
                        metadata_id = chunk_id
                
                # Similarly for metadata_source
                metadata_source = getattr(chunk, 'metadata_source', '')
                if not metadata_source and isinstance(metadata, dict):
                    metadata_source = metadata.get('source', '')
                
                # Convert metadata_id to string if needed
                if not isinstance(metadata_id, str):
                    metadata_id = str(metadata_id)
                
                # Ensure metadata_id is not None or empty
                if not metadata_id or (isinstance(metadata_id, str) and metadata_id.strip() == ''):
                    metadata_id = 'unknown'
                
                # Convert metadata dict to JSON string for Neo4j compatibility
                metadata_json = "{}"
                if metadata and isinstance(metadata, dict):
                    try:
                        metadata_json = json.dumps(metadata, default=str)
                    except Exception as json_err:
                        logger.warning(f"Failed to serialize metadata for chunk {chunk_id}: {json_err}, using empty JSON")
                        metadata_json = "{}"
                
                # Build node properties without metadata_id (will be added by create_node)
                node_properties = {
                    "chunk_id": chunk_id,
                    "metadata_source": metadata_source,
                    "text": chunk_text[:500],  # Truncate long text
                    "metadata": metadata_json,  # Store as JSON string
                    "import_timestamp": datetime.now().isoformat(),
                    "source_namespace": namespace,
                    "index_name": index_name,
                    "engine_name": engine_name
                }
                # Add metadata_id to properties only if not None (create_node will add it)
                if metadata_id is not None:
                    node_properties["metadata_id"] = metadata_id
                
                node = Node(
                    label="Document",
                    properties=node_properties
                )
                await self.graph_service.create_node(node, namespace=namespace, index_name=index_name, engine_name=engine_name, engine_type=engine_type, metadata_id=metadata_id, graph_name=graph_name)
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

    async def add_document_to_graph(
        self,
        metadata_id: str,
        namespace: str,
        engine: Engine,
        vector_store_repo,
        llm=None,
        deduplicate_entities: bool = True,
        graph_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add all chunks of a document to the knowledge graph by retrieving them from vector store.

        This method allows incremental updates to the graph when a new document is added
        to the knowledge base, without regenerating the entire namespace graph.
        It:
        1. Retrieves all chunks of the document from vector store using metadata_id
        2. Extracts entities and relationships from each chunk using GraphRAG
        3. Creates nodes and relationships in Neo4j
        4. Optionally deduplicates entities by checking if they already exist

        Args:
            metadata_id: Unique identifier for the document in vector store
            namespace: Namespace for the graph (e.g., 'bancaitalia')
            engine: Engine configuration (must include name, index_name, and type/deployment)
            vector_store_repo: Vector store repository instance for retrieving chunks
            llm: LLM instance for entity extraction (required)
            deduplicate_entities: If True, reuses existing entity nodes when found (default: True)

        Returns:
            Dictionary with statistics:
            {
                "metadata_id": str,
                "chunks_processed": int,
                "entities_extracted": int,
                "entities_new": int,
                "entities_reused": int,
                "relationships_created": int,
                "status": "success" | "no_llm" | "no_entities" | "no_chunks"
            }

        Raises:
            ValueError: If LLM is not provided or engine is not configured properly
            RuntimeError: If GraphRAG extractor is not available

        Example:
            result = await graph_rag_service.add_chunk_to_graph(
                metadata_id="doc_123",
                namespace="economia",
                engine=engine,
                vector_store_repo=repo,
                llm=chat_model
            )
        """
        logger.info(f"Adding document to graph: metadata_id={metadata_id}, namespace={namespace}")

        # Validate inputs
        if not metadata_id or not metadata_id.strip():
            raise ValueError("metadata_id cannot be empty")

        if llm is None:
            raise ValueError("LLM is required for entity/relationship extraction. Provide an LLM instance.")

        if not GRAPHRAG_AVAILABLE or GraphRAGExtractor is None:
            raise RuntimeError("GraphRAG extractor is not available. Install required dependencies.")

        if vector_store_repo is None:
            raise ValueError("vector_store_repo is required to retrieve chunks from vector store")

        if engine is None:
            raise ValueError("engine configuration is required")

        # Ensure graph service repository is available
        self.graph_service._ensure_repository()

        # Determine engine_name and engine_type based on engine configuration
        engine_name = engine.name
        index_name = engine.index_name
        graph_name_to_use = graph_name if graph_name is not None else namespace

        # For Pinecone use 'type', for Qdrant use 'deployment'
        if engine.name == "pinecone":
            engine_type = engine.type if hasattr(engine, 'type') else None
        elif engine.name == "qdrant":
            engine_type = engine.deployment if hasattr(engine, 'deployment') else None
        else:
            engine_type = None

        # Step 1: Retrieve all chunks of the document from vector store
        logger.info(f"Retrieving chunks for document {metadata_id} from vector store")
        try:
            repository_items = await vector_store_repo.get_ids_namespace(
                engine=engine,
                metadata_id=metadata_id,
                namespace=namespace
            )
            chunks = repository_items.matches if repository_items and repository_items.matches else []
            logger.info(f"Retrieved {len(chunks)} chunks for document {metadata_id}")
        except Exception as e:
            logger.error(f"Failed to retrieve chunks from vector store: {e}")
            raise RuntimeError(f"Failed to retrieve chunks: {e}")

        if not chunks:
            logger.warning(f"No chunks found for document {metadata_id}")
            return {
                "metadata_id": metadata_id,
                "chunks_processed": 0,
                "entities_extracted": 0,
                "entities_new": 0,
                "entities_reused": 0,
                "relationships_created": 0,
                "status": "no_chunks"
            }

        # Step 2: Prepare chunks for GraphRAG extraction
        chunks_for_extraction = []
        for chunk in chunks:
            chunk_id = chunk.id if hasattr(chunk, 'id') else str(chunk)
            chunk_text = chunk.text if hasattr(chunk, 'text') else ""

            if chunk_text and chunk_text.strip():
                chunks_for_extraction.append({
                    "id": chunk_id,
                    "text": chunk_text
                })

        if not chunks_for_extraction:
            logger.warning(f"No valid chunks with text found for document {metadata_id}")
            return {
                "metadata_id": metadata_id,
                "chunks_processed": 0,
                "entities_extracted": 0,
                "entities_new": 0,
                "entities_reused": 0,
                "relationships_created": 0,
                "status": "no_chunks"
            }

        # Step 3: Extract entities and relationships using GraphRAG from all chunks
        logger.info(f"Extracting entities and relationships from {len(chunks_for_extraction)} chunks")
        try:
            extractor = GraphRAGExtractor(llm)
            entities, relationships = await extractor.extract(
                doc_id=metadata_id,
                chunks=chunks_for_extraction
            )
            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from document {metadata_id}")
        except Exception as e:
            logger.error(f"GraphRAG extraction failed: {e}")
            raise RuntimeError(f"Failed to extract entities/relationships: {e}")

        if not entities:
            logger.warning(f"No entities extracted from document {metadata_id}")
            return {
                "metadata_id": metadata_id,
                "chunks_processed": len(chunks_for_extraction),
                "entities_extracted": 0,
                "entities_new": 0,
                "entities_reused": 0,
                "relationships_created": 0,
                "status": "no_entities"
            }

        # Step 4: Create or find entity nodes (with optional deduplication)
        entity_node_map = {}  # entity_name -> node_id
        entities_new = 0
        entities_reused = 0

        for entity in entities:
            try:
                entity_name = entity.get("entity_name", "").strip()
                entity_type = entity.get("entity_type", "ENTITY").strip()
                description = entity.get("description", "")
                source_ids = entity.get("source_id", [])

                if not entity_name:
                    continue

                existing_node = None

                # Check if entity already exists (if deduplication enabled)
                if deduplicate_entities:
                    existing_node = await self.graph_service.find_entity_by_name_and_type(
                        entity_name=entity_name,
                        entity_type=entity_type,
                        namespace=namespace,
                        index_name=index_name,
                        graph_name=graph_name_to_use
                    )

                if existing_node:
                    # Reuse existing entity node
                    logger.debug(f"Reusing existing entity: {entity_name} (ID: {existing_node.id})")
                    entity_node_map[entity_name] = existing_node.id
                    entities_reused += 1
                else:
                    # Create new entity node
                    node = Node(
                        label=entity_type,
                        properties={
                            "name": entity_name,
                            "description": description,
                            "source_ids": source_ids if source_ids else [],
                            "entity_type": entity_type,
                            "import_timestamp": datetime.now().isoformat(),
                            "engine_name": engine_name,
                            "engine_type": engine_type,
                            "metadata_id": metadata_id
                        }
                    )
                    created_node = await self.graph_service.create_node(
                        node,
                        namespace=namespace,
                        index_name=index_name,
                        engine_name=engine_name,
                        engine_type=engine_type,
                        metadata_id=metadata_id,
                        graph_name=graph_name_to_use
                    )

                    if created_node.id is None:
                        logger.error(f"Created entity node {entity_name} has no ID, skipping")
                        continue

                    logger.debug(f"Created new entity: {entity_name} (ID: {created_node.id})")
                    entity_node_map[entity_name] = created_node.id
                    entities_new += 1

            except Exception as ex:
                logger.error(f"Error processing entity {entity.get('entity_name', 'unknown')}: {ex}")
                continue

        # Step 5: Create relationships between entities
        relationships_created = 0

        for rel in relationships:
            try:
                source_name = rel.get("src_id", "").strip()
                target_name = rel.get("tgt_id", "").strip()
                # Use extracted relationship_type, fallback to RELATED_TO
                rel_type = rel.get("relationship_type", "RELATED_TO").upper()
                weight = rel.get("weight", 1.0)
                description = rel.get("description", "")
                source_ids = rel.get("source_id", [])

                # DEBUG: Log what type was extracted
                logger.info(f"Creating relationship: {source_name} -[{rel_type}]-> {target_name} (extracted from: {rel.get('relationship_type', 'NOT_FOUND')})")

                if not source_name or not target_name:
                    continue

                source_node_id = entity_node_map.get(source_name)
                target_node_id = entity_node_map.get(target_name)

                if not source_node_id or not target_node_id:
                    logger.debug(f"Missing source or target node for relationship {source_name} -> {target_name}")
                    continue

                rel_properties = {
                    "weight": weight,
                    "description": description,
                    "source_ids": source_ids if source_ids else [],
                    "source_entity": source_name,
                    "target_entity": target_name,
                    "engine_name": engine_name,
                    "engine_type": engine_type,
                    "metadata_id": metadata_id
                }

                relationship = Relationship(
                    source_id=source_node_id,
                    target_id=target_node_id,
                    type=rel_type,
                    properties=rel_properties
                )

                await self.graph_service.create_relationship(
                    relationship,
                    namespace=namespace,
                    index_name=index_name,
                    engine_name=engine_name,
                    engine_type=engine_type,
                    metadata_id=metadata_id,
                    graph_name=graph_name_to_use
                )
                relationships_created += 1
                logger.debug(f"Created relationship: {source_name} -> {target_name}")

            except Exception as ex:
                logger.error(f"Error creating relationship {rel.get('src_id', 'unknown')} -> {rel.get('tgt_id', 'unknown')}: {ex}")
                continue

        logger.info(f"Document {metadata_id} added to graph: {len(chunks_for_extraction)} chunks, "
                   f"{entities_new} new entities, {entities_reused} reused, {relationships_created} relationships")

        return {
            "metadata_id": metadata_id,
            "chunks_processed": len(chunks_for_extraction),
            "entities_extracted": len(entities),
            "entities_new": entities_new,
            "entities_reused": entities_reused,
            "relationships_created": relationships_created,
            "status": "success"
        }

    async def answer_with_graph(self, question: str, namespace: str, max_results: Optional[int] = 10, similarity_threshold: Optional[float] = 0.3, llm=None, llm_embeddings=None, index_name: Optional[str] = None, chat_history_dict: Optional[Dict[str, Any]] = None, graph_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a question using the knowledge graph (GraphRAG).

        Args:
            question: The question to answer
            namespace: Namespace/collection to search within (optional, may be used for filtering)
            max_results: Maximum number of graph elements to consider
            similarity_threshold: Similarity threshold for relevance
            llm: LLM instance for answer synthesis (if None, uses self.llm)
            llm_embeddings: Embedding model (optional)
            index_name: Optional index_name for graph partition
            chat_history_dict: Optional chat history dictionary for context

        Returns:
            Dictionary with answer and graph elements used
        """
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
        nodes = await repo.search_nodes_by_text(question, limit=max_results, namespace=namespace, index_name=index_name)
        relationships = await repo.search_relationships_by_text(question, limit=max_results, namespace=namespace, index_name=index_name)
        
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
                # Prepare context using formatted graph context
                graph_context = format_graph_context(entities, rels)
                chat_history_formatted = format_chat_history(chat_history_dict) if chat_history_dict else "No chat history available."
                
                # Create prompt using template
                prompt = GRAPH_QA_PROMPT_TEMPLATE.format(
                    question=question,
                    context=graph_context,
                    chat_history=chat_history_formatted
                )
                
                # Call LLM
                if hasattr(llm_to_use, 'invoke'):
                    from langchain_core.messages import HumanMessage, SystemMessage
                    messages = [
                        SystemMessage(content=GRAPH_QA_SYSTEM_PROMPT),
                        HumanMessage(content=prompt)
                    ]
                    response = await llm_to_use.ainvoke(messages)
                    answer = response.content if hasattr(response, 'content') else str(response)
                elif hasattr(llm_to_use, 'chat'):
                    answer = await llm_to_use.chat(
                        system=GRAPH_QA_SYSTEM_PROMPT,
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
            "query_used": question,
            "chat_history_dict": chat_history_dict
        }

    async def _search_community_reports_parquet(self, question: str, namespace: str, limit: int = 5, index_name: Optional[str] = None, index_type: Optional[str] = None) -> List[Dict]:
        """
        Fast search for community reports using DuckDB on Parquet files from MinIO.
        """
        temp_dir = None
        try:
            # 1. Download Parquet file from MinIO
            temp_dir = tempfile.mkdtemp(prefix=f"report_search_{namespace}_")
            local_path = Path(temp_dir) / "community_reports.parquet"
            
            try:
                self.minio_storage.download_parquet_file(
                    namespace=namespace,
                    file_name="community_reports.parquet",
                    local_path=str(local_path),
                    index_name=index_name,
                    index_type=index_type
                )
                logger.info(f"Downloaded reports for {namespace} to {local_path}")
            except Exception as e:
                logger.warning(f"Could not download reports for {namespace}: {e}. Global search might be empty.")
                return []

            if not os.path.exists(local_path):
                return []

            # 2. Query with DuckDB
            safe_question = question.replace("'", "''")
            keywords = [w for w in safe_question.split() if len(w) > 3]
            if not keywords: keywords = [safe_question]
            
            keyword_conditions = " OR ".join(["(title ILIKE ? OR summary ILIKE ?)" for _ in keywords[:5]])
            params = [str(local_path)] + [item for kw in keywords[:5] for item in (f"%{kw}%", f"%{kw}%")]
            
            query = f"""
                SELECT title, summary, full_report, level, rating
                FROM read_parquet(?)
                WHERE {keyword_conditions}
                ORDER BY rating DESC, level ASC
                LIMIT {limit}
            """
            conn = duckdb.connect(database=':memory:')
            results = conn.execute(query, params).fetchall()
            conn.close()

            # Convert results to list of dicts
            return [{"title": r[0], "summary": r[1], "full_report": r[2], "level": r[3], "rating": r[4]} for r in results]

        except Exception as e:
            logger.error(f"Error searching community reports in Parquet: {e}")
            return []
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

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
            embedding_dimension: embedding dimension
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
        
        # 1. Global Search (Community-based) - NOW FAST via Parquet/DuckDB
        community_reports = []
        if query_type == "exploratory" or adjusted_weights["graph"] > 1.0:
            logger.info("Performing Global Search (Community-based via Parquet/DuckDB)")
            try:
                # Determine index_name and index_type for MinIO download
                index_name_val = None
                index_type_val = None
                if engine is not None:
                    index_name_val = engine.index_name
                    if engine.name == "pinecone":
                        index_type_val = engine.type  # "serverless" or "pod"
                    elif engine.name == "qdrant":
                        index_type_val = engine.deployment  # "local" or "cloud"
                # FAST SEARCH on Parquet files
                reports = await self._search_community_reports_parquet(
                    question, namespace, limit=3, 
                    index_name=index_name_val, index_type=index_type_val
                )
                if reports:
                    logger.info(f"Found {len(reports)} community reports from Parquet")
                    # Convert to Node-like structure for consistency if needed, or use as dict
                    community_reports = [Node(label="CommunityReport", properties=report) for report in reports]
                else:
                    logger.info("No community reports found in Parquet")
            except Exception as e:
                logger.warning(f"Global search on Parquet failed: {e}")

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

            # Perform hybrid search (or similarity/keyword based on search_type)
            logger.info(f"Performing hybrid search with namespace: {namespace}, engine: {engine}")
            search_results = await vector_store_repo.perform_hybrid_search(
                question_answer, index, dense_vector, sparse_vector
            )

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
                    nodes = await graph_repo.find_nodes_by_source_id(chunk_id, limit=5, namespace=namespace, index_name=engine.index_name)
                    if nodes:
                        logger.info(f"Found {len(nodes)} nodes for chunk {chunk_id}")
                        for node in nodes:
                            logger.info(f"  Node ID: {node.id}, label: {node.label}, source_ids: {node.properties.get('source_ids', 'N/A')}, chunk_id: {node.properties.get('chunk_id', 'N/A')}")
                            seed_node_ids.append(node.id)
                    else:
                        logger.info(f"No nodes found for chunk {chunk_id}")
            
            # Also use community reports as seed nodes if available
            for report in community_reports:
                if report.id:
                    seed_node_ids.append(report.id)

            # Deduplicate seed node IDs
            seed_node_ids = list(set(seed_node_ids))

            # Expand graph from seed nodes using adaptive GraphExpander
            expanded_graph = {"nodes": [], "relationships": []}
            if seed_node_ids and adjusted_weights["graph"] > 0:
                try:
                    # Scale expansion based on graph weight
                    expansion_limit = int(max_expansion_nodes * adjusted_weights["graph"])

                    # Use adaptive GraphExpander with dynamic hop count
                    graph_expander = GraphExpander(repository=graph_repo)

                    expansion_result = await graph_expander.expand_from_seeds(
                        seed_node_ids=seed_node_ids,
                        query_type=query_type,  # Determines max hops: technical=1, exploratory=2, relational=3
                        max_nodes=expansion_limit,
                        namespace=namespace,
                        index_name=engine.index_name,
                        min_relationship_weight=0.0
                    )

                    # Convert to expected format
                    expanded_graph = {
                        "nodes": [
                            {
                                "id": node.id,
                                "label": node.label,
                                "properties": node.properties
                            } for node in expansion_result["nodes"]
                        ],
                        "relationships": [
                            {
                                "id": rel.id,
                                "type": rel.type,
                                "source_id": rel.source_id,
                                "target_id": rel.target_id,
                                "properties": rel.properties
                            } for rel in expansion_result["relationships"]
                        ]
                    }

                    logger.info(f"Adaptive graph expansion complete: {len(expanded_graph['nodes'])} nodes, "
                               f"{len(expanded_graph['relationships'])} relationships in "
                               f"{expansion_result['hops_executed']} hops (query_type: {query_type})")

                except Exception as e:
                    logger.error(f"Graph expansion failed: {e}, continuing without expansion")
                    expanded_graph = {"nodes": [], "relationships": []}
            
            # 5. Final Reranking (Cross-Encoder)
            # Collect all pieces of information as Document objects for reranking
            all_context_docs = []
            
            # Community reports
            for report in community_reports:
                content = report.properties.get("summary") or report.properties.get("title", "")
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
                        content = report.properties.get("summary") or report.properties.get("title", "")
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
                            Chat History:
                            {format_chat_history(chat_history_dict) if chat_history_dict else 'No chat history available.'}
                            """
                    
                    if hasattr(llm, 'invoke'):
                        from langchain_core.messages import HumanMessage, SystemMessage
                        messages = [
                            SystemMessage(content=ADVANCED_GRAPH_QA_SYSTEM_PROMPT),
                            HumanMessage(content=prompt)
                        ]
                        response = await llm.ainvoke(messages)
                        answer = response.content if hasattr(response, 'content') else str(response)
                    elif hasattr(llm, 'chat'):
                        answer = await llm.chat(
                            system=ADVANCED_GRAPH_QA_SYSTEM_PROMPT,
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
                "expanded_relationships": expanded_graph["relationships"],
                "chat_history_dict": chat_history_dict
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
                llm=llm,
                llm_embeddings=None,
                index_name=None,
                chat_history_dict=chat_history_dict
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
                "expanded_relationships": [],
                "chat_history_dict": basic_result.get("chat_history_dict")
            }
            return enhanced_result
    
    def _detect_query_type(self, question: str, llm: Optional[Any] = None, use_llm: bool = True) -> str:
        """
        Detect query type based on question content.
        Uses LLM-based detection if available, falls back to heuristics.

        Args:
            question: User's question
            llm: Optional LLM instance for accurate detection
            use_llm: Whether to attempt LLM-based detection

        Returns:
            Query type: "exploratory", "technical", or "relational"
        """
        # Try LLM-based detection first
        if use_llm and llm:
            try:
                # Use async wrapper if needed
                import asyncio
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No running loop, create new one
                    pass

                if loop:
                    # We're in async context, use sync fallback
                    logger.debug("In async context, using heuristic query type detection")
                    return detect_query_type_heuristic(question)
                else:
                    # Not in async context, can run async
                    return asyncio.run(detect_query_type_with_llm(question, llm, fallback_to_heuristic=True))
            except Exception as e:
                logger.warning(f"LLM-based query detection failed: {e}, using heuristics")

        # Fall back to heuristic detection
        return detect_query_type_heuristic(question)
    
    def _adjust_weights_by_query_type(self, vector_weight: float, keyword_weight: float,
                                      graph_weight: float, query_type: str) -> Dict[str, float]:
        """
        Adjust weights based on query type according to the balancing matrix.
        Uses centralized weight adjustment logic.
        """
        base_weights = {
            "vector": vector_weight,
            "keyword": keyword_weight,
            "graph": graph_weight
        }

        # Use centralized weight adjustment logic
        adjusted_weights = apply_weight_adjustments(base_weights, query_type)

        return adjusted_weights

    async def cluster_graph(self, llm=None, level: int = 0, namespace: Optional[str] = None, index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None, graph_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect communities in the graph and generate reports for each.
        
        Args:
            llm: LLM instance for report generation
            level: The hierarchy level
            namespace: Optional namespace to filter graph partition
            index_name: Optional index_name to filter graph partition
            engine_name: Optional engine name (e.g., 'pinecone', 'qdrant') to filter graph partition
            engine_type: Optional engine type (e.g., 'pod', 'serverless') to filter graph partition
            graph_name: Optional explicit graph name (overrides namespace for graph selection)
            
        Returns:
            Statistics about the clustering process
        """
        llm_to_use = llm or self.llm
        if not llm_to_use:
            raise ValueError("LLM configuration is required for community report generation.")
            
        repo = self.graph_service._get_repository()
        cluster_service = ClusterService(repository=repo, llm=llm_to_use)
        
        return await cluster_service.perform_clustering(
            level=level, 
            namespace=namespace if namespace is not None else "default", 
            index_name=index_name,
            engine_name=engine_name,
            engine_type=engine_type,
            graph_name=graph_name
        )
