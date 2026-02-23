"""
Abstract base class for graph database repositories.
Defines the interface for graph operations that must be implemented by concrete repositories.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from tilellm.modules.knowledge_graph.models import Node, Relationship

logger = logging.getLogger(__name__)


class BaseGraphRepository(ABC):
    """
    Abstract base repository for graph database operations.
    All concrete implementations must implement these methods.
    """
    
    SEARCHABLE_PROPERTIES = ['text', 'content', 'name', 'title', 'description', 'summary', 'label', 'type']
    SEARCHABLE_REL_PROPERTIES = ['type', 'weight', 'context', 'description', 'source', 'target']
    
    @abstractmethod
    def __init__(self):
        """Initialize connection to graph database."""
        pass
    
    @abstractmethod
    async def ensure_indexes(self):
        """Ensure that required indexes exist in the database."""
        pass

    @abstractmethod
    async def close(self):
        """Close the database connection pool."""
        pass

    @abstractmethod
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher/openCypher query and return results.

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Returns:
            List of result records as dictionaries
        """
        pass
    
    # ==================== NODE OPERATIONS ====================

    @abstractmethod
    async def create_node(self, node: Node, namespace: Optional[str] = None,
                   index_name: Optional[str] = None, engine_name: Optional[str] = None,
                   engine_type: Optional[str] = None, metadata_id: Optional[str] = None) -> Node:
        """
        Create a new node in the graph.

        Args:
            node: Node object with label and properties
            namespace: Optional namespace to partition the graph
            index_name: Optional index_name (collection name) to partition the graph
            engine_name: Optional engine name (e.g., 'pinecone', 'qdrant') to partition the graph
            engine_type: Optional engine type (e.g., 'pod', 'serverless', 'local', 'cloud')
            metadata_id: Optional metadata ID from vector store chunks for cleanup

        Returns:
            Created node with generated ID
        """
        pass

    @abstractmethod
    async def find_node_by_id(self, node_id: str, namespace: Optional[str] = None) -> Optional[Node]:
        """
        Find a node by its internal database ID.

        Args:
            node_id: Internal node ID
            namespace: Optional namespace to search in (for multi-graph isolation)

        Returns:
            Node if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_nodes_by_label(self, label: str, limit: int = 100,
                           namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
        """
        Find all nodes with a specific label.

        Args:
            label: Node label to search for
            limit: Maximum number of nodes to return
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes

        Returns:
            List of matching nodes
        """
        pass

    @abstractmethod
    async def find_nodes_by_property(self, label: str, property_key: str, property_value: Any,
                              limit: int = 100, namespace: Optional[str] = None,
                              index_name: Optional[str] = None) -> List[Node]:
        """
        Find nodes by a specific property value.

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
        pass

    @abstractmethod
    async def find_nodes_by_source_id(self, source_id: str, limit: int = 10,
                                     namespace: Optional[str] = None,
                                     index_name: Optional[str] = None) -> List[Node]:
        """
        Find nodes that reference a specific source ID (vector store chunk ID).
        Checks 'chunk_id', 'metadata_id', and 'source_ids' list property.

        Args:
            source_id: The source ID to look for
            limit: Maximum number of nodes to return
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes

        Returns:
            List of matching nodes
        """
        pass

    @abstractmethod
    async def update_node(self, node_id: str, label: Optional[str] = None,
                   properties: Optional[Dict[str, Any]] = None,
                   namespace: Optional[str] = None) -> Optional[Node]:
        """
        Update a node's label and/or properties.

        Args:
            node_id: Internal node ID
            label: New label (optional)
            properties: New properties to set/update (optional)
            namespace: Optional namespace where the node exists (for multi-graph isolation)

        Returns:
            Updated node if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_node(self, node_id: str, detach: bool = True, namespace: Optional[str] = None) -> bool:
        """
        Delete a node from the graph.

        Args:
            node_id: Internal node ID
            detach: If True, delete all relationships before deleting node
            namespace: Optional namespace where the node exists (for multi-graph isolation)

        Returns:
            True if deleted, False if not found
        """
        pass
    
    # ==================== RELATIONSHIP OPERATIONS ====================

    @abstractmethod
    async def create_relationship(self, relationship: Relationship, namespace: Optional[str] = None,
                           index_name: Optional[str] = None, engine_name: Optional[str] = None,
                           engine_type: Optional[str] = None, metadata_id: Optional[str] = None) -> Relationship:
        """
        Create a relationship between two nodes.

        Args:
            relationship: Relationship object with source, target, type, and properties
            namespace: Optional namespace to partition the graph
            index_name: Optional index_name (collection name) to partition the graph
            engine_name: Optional engine name (e.g., 'pinecone', 'qdrant') to partition the graph
            engine_type: Optional engine type (e.g., 'pod', 'serverless')
            metadata_id: Optional metadata ID from vector store chunks for cleanup

        Returns:
            Created relationship with generated ID
        """
        pass

    @abstractmethod
    async def find_relationship_by_id(self, relationship_id: str, namespace: Optional[str] = None) -> Optional[Relationship]:
        """
        Find a relationship by its internal database ID.

        Args:
            relationship_id: Internal relationship ID
            namespace: Optional namespace to search in (for multi-graph isolation)

        Returns:
            Relationship if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_relationships_by_node(self, node_id: str, direction: str = "both",
                                   namespace: Optional[str] = None) -> List[Relationship]:
        """
        Find all relationships connected to a node.

        Args:
            node_id: Internal node ID
            direction: "incoming", "outgoing", or "both"
            namespace: Optional namespace to search in (for multi-graph isolation)

        Returns:
            List of relationships
        """
        pass

    @abstractmethod
    async def update_relationship(self, relationship_id: str, rel_type: Optional[str] = None,
                           properties: Optional[Dict[str, Any]] = None,
                           namespace: Optional[str] = None) -> Optional[Relationship]:
        """
        Update a relationship's type and/or properties.

        Args:
            relationship_id: Internal relationship ID
            rel_type: New relationship type (may require recreation)
            properties: New properties to set/update
            namespace: Optional namespace where the relationship exists (for multi-graph isolation)

        Returns:
            Updated relationship if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_relationship(self, relationship_id: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a relationship from the graph.

        Args:
            relationship_id: Internal relationship ID
            namespace: Optional namespace where the relationship exists (for multi-graph isolation)

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def delete_nodes_by_metadata(
        self,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        engine_name: Optional[str] = None,
        engine_type: Optional[str] = None,
        metadata_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Delete all nodes (Entity and CommunityReport) that match the specified metadata.
        Also deletes all relationships connected to those nodes.

        Args:
            namespace: Namespace to filter nodes
            index_name: Index name/collection to filter nodes
            engine_name: Engine name (e.g., 'pinecone', 'qdrant')
            engine_type: Engine type (e.g., 'pod', 'serverless')
            metadata_id: Metadata ID from vector store chunks

        Returns:
            Dictionary with deletion statistics
        """
        pass
    
    # ==================== UTILITY OPERATIONS ====================

    @abstractmethod
    async def verify_connection(self) -> bool:
        """
        Verify the connection to the graph database is working.

        Returns:
            True if connection is successful
        """
        pass

    @abstractmethod
    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database.

        Returns:
            Dictionary with database statistics
        """
        pass

    @abstractmethod
    async def search_nodes_by_text(self, search_text: str, limit: int = 10,
                            namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
        """
        Search nodes using a full-text index.

        Args:
            search_text: Text to search for. Can include Lucene syntax if supported.
            limit: Maximum number of nodes to return.
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes

        Returns:
            List of matching nodes.
        """
        pass

    @abstractmethod
    async def search_relationships_by_text(self, search_text: str, limit: int = 10,
                                    namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search relationships using a full-text index.

        Args:
            search_text: Text to search for. Can include Lucene syntax if supported.
            limit: Maximum number of relationships to return.
            namespace: Optional namespace to filter relationships
            index_name: Optional index_name (collection name) to filter relationships

        Returns:
            List of matching relationships with node info.
        """
        pass

    @abstractmethod
    async def search_community_reports(self, search_text: str, limit: int = 5) -> List[Node]:
        """
        Search community reports by text.

        Args:
            search_text: Text to search for
            limit: Maximum number of reports to return

        Returns:
            List of matching community report nodes
        """
        pass

    @abstractmethod
    async def get_all_nodes_and_relationships(self, namespace: Optional[str] = None,
                                       index_name: Optional[str] = None,
                                       engine_name: Optional[str] = None,
                                       engine_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch the entire graph, optionally filtered by namespace, index_name, engine_name and engine_type.

        Args:
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes
            engine_name: Optional engine name (e.g., 'pinecone', 'qdrant') to filter nodes
            engine_type: Optional engine type (e.g., 'pod', 'serverless') to filter nodes

        Returns:
            Dictionary with filtered nodes and relationships
        """
        pass

    @abstractmethod
    async def get_community_network(self, namespace: str, index_name: str, limit: int = 1000) -> Dict[str, Any]:
        """
        Fetch the community graph (nodes + BELONGS_TO_COMMUNITY relationships).

        Args:
            namespace: Namespace to filter
            index_name: index_name to filter
            limit: Maximum number of paths to return

        Returns:
            Dictionary with nodes and relationships
        """
        pass

    def delete_graph(self, namespace: str) -> bool:
        """
        Delete an entire graph (all data for a namespace/client).
        This is an optional method for databases that support multi-graph architecture.

        Note: This is a destructive operation - use with caution!
        Perfect for: client data removal, testing cleanup, GDPR compliance.

        Args:
            namespace: The namespace (graph name) to delete

        Returns:
            True if deletion successful, False otherwise

        Raises:
            NotImplementedError: If the database implementation doesn't support this operation
        """
        raise NotImplementedError("delete_graph() not implemented for this database backend")

    # ==================== HELPER METHODS ====================
    
    def _convert_database_value(self, value) -> Any:
        """
        Convert database-specific types (e.g., temporal types) to Python serializable types.
        Default implementation returns value as-is. Override in concrete implementations.
        
        Args:
            value: Value from database
            
        Returns:
            Python serializable value
        """
        return value
    
    def _convert_properties(self, props_dict) -> Dict[str, Any]:
        """
        Convert database properties dict to Python serializable dict.
        
        Args:
            props_dict: Properties dictionary
            
        Returns:
            Converted properties dictionary
        """
        if not props_dict:
            return {}
        if isinstance(props_dict, dict):
            return {k: self._convert_database_value(v) for k, v in props_dict.items()}
        return {}