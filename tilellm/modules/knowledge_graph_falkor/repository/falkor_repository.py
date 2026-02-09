"""
FalkorDB Repository for Knowledge Graph operations.
Provides CRUD operations for nodes and relationships with connection pooling.
Uses FalkorDB (Redis Graph) with openCypher support.
"""

import json
import logging
import os
from typing import Optional, List, Dict, Any, cast, Union
from urllib.parse import urlparse

from tilellm.shared.utility import get_service_config
from tilellm.store.graph import BaseGraphRepository
from tilellm.modules.knowledge_graph.models import Node, Relationship

logger = logging.getLogger(__name__)

try:
    from falkordb import FalkorDB, Node as FalkorNode, Edge as FalkorEdge, Path as FalkorPath
    from redis import ConnectionPool
    FALKORDB_AVAILABLE = True
except ImportError:
    FalkorDB = None
    FalkorNode = None
    FalkorEdge = None
    FalkorPath = None
    ConnectionPool = None
    FALKORDB_AVAILABLE = False
    logger.warning("FalkorDB client not available. Please install 'falkordb' package.")


class FalkorGraphRepository(BaseGraphRepository):
    """
    Repository for interacting with FalkorDB graph database.
    Uses FalkorDB Python client.

    Strategy: One graph per namespace for perfect multi-tenant isolation.
    Indexes are created per-graph on first access.
    """

    _client: Optional[Any] = None
    _default_graph_name: str = "knowledge_graph"
    _indexed_graphs: set = set()  # Track graphs with indexes already created

    def __init__(self):
        """
        Initialize FalkorDB connection using environment variables.
        Uses the official FalkorDB Python client.
        """
        if FalkorGraphRepository._client is None:
            if not FALKORDB_AVAILABLE:
                raise ImportError("FalkorDB client not installed. Please install with 'pip install falkordb'")

            service_config = get_service_config()
            falkordb_config = service_config.get("falkordb") or service_config.get("redis")  # Fallback to redis config
            
            if not falkordb_config:
                raise ValueError("FalkorDB configuration not found in environment variables. "
                               "Set FALKORDB_URI or REDIS_URI.")
            
            # FalkorDB connection parameters
            uri = falkordb_config.get("uri")
            host = falkordb_config.get("host", "localhost")
            port = falkordb_config.get("port", 6379)
            password = falkordb_config.get("password")
            username = falkordb_config.get("username")

            graph_name = falkordb_config.get("graph_name", "knowledge_graph")

            # Connection pool configuration (for performance optimization)
            max_connections = falkordb_config.get("max_connections", 50)
            socket_timeout = falkordb_config.get("socket_timeout", 30.0)
            socket_connect_timeout = falkordb_config.get("socket_connect_timeout", 10.0)
            socket_keepalive = falkordb_config.get("socket_keepalive", True)
            retry_on_timeout = falkordb_config.get("retry_on_timeout", True)

            # Override with environment variable if present
            env_uri = os.environ.get("FALKORDB_URI")
            if env_uri:
                uri = env_uri

            FalkorGraphRepository._default_graph_name = graph_name

            try:
                logger.info(f"Initializing FalkorDB client for host={host}:{port} with connection pooling (max_connections={max_connections})")

                # Create connection pool with optimized settings
                pool_kwargs = {
                    "max_connections": max_connections,
                    #"decode_responses": True,  # Auto-decode to strings
                    "socket_timeout": socket_timeout,
                    "socket_connect_timeout": socket_connect_timeout,
                    "socket_keepalive": socket_keepalive,
                    "retry_on_timeout": retry_on_timeout,
                    "health_check_interval": 30,  # Check connection health every 30s
                }

                if uri:
                    # Use URI directly with FalkorDB client (it handles pooling internally)
                    connection_kwargs = {
                        "url": uri,
                        "max_connections": max_connections,
                        "socket_timeout": socket_timeout,
                        #"decode_responses": True
                    }
                else:
                    # Build connection pool from host/port
                    pool_kwargs["host"] = host
                    pool_kwargs["port"] = port
                    if password:
                        pool_kwargs["password"] = password
                    if username:
                        pool_kwargs["username"] = username

                    pool = ConnectionPool(**pool_kwargs)
                    connection_kwargs = {"connection_pool": pool}

                FalkorGraphRepository._client = FalkorDB(**connection_kwargs)

                logger.info(f"FalkorDB connection pool initialized with {max_connections} max connections")
                
                logger.info("FalkorDB client initialized")
                
                # Ensure indexes on default graph
                self.ensure_indexes()
                
            except Exception as e:
                logger.error(f"Failed to initialize FalkorDB connection: {e}")
                raise ConnectionError(f"Failed to connect to FalkorDB: {e}")

    def _get_graph_name(self, namespace: Optional[str] = None, index_name: Optional[str] = None) -> str:
        """
        Determine graph name based on namespace-first strategy.

        Strategy hierarchy:
        1. Use namespace if provided (primary - aligns with vector store multi-tenancy)
        2. Fallback to default_graph_name for global operations

        Args:
            namespace: Client/tenant namespace (e.g., "cliente_A")
            index_name: Collection name (stored as property, not used for graph selection)

        Returns:
            Graph name to use
        """
        if namespace:
            return namespace

        return self._default_graph_name

    def _get_graph(self, namespace: Optional[str] = None, index_name: Optional[str] = None):
        """
        Select the graph based on namespace and ensure indexes exist.
        Creates indexes on first access to each graph (lazy initialization).

        Args:
            namespace: Client/tenant namespace
            index_name: Collection name (for tracking only)

        Returns:
            FalkorDB Graph instance
        """
        graph_name = self._get_graph_name(namespace, index_name)
        graph = self._client.select_graph(graph_name)

        # Ensure indexes on first access to this graph
        if graph_name not in FalkorGraphRepository._indexed_graphs:
            self._ensure_indexes_for_graph(graph_name)
            FalkorGraphRepository._indexed_graphs.add(graph_name)

        return graph

    def _ensure_indexes_for_graph(self, graph_name: str):
        """
        Create indexes on a specific graph using FalkorDB openCypher syntax.
        Called automatically on first access to each graph.

        FalkorDB uses standard openCypher syntax for indexes:
        CREATE INDEX FOR (n:Label) ON (n.property)

        Note: FalkorDB creates indexes across all node labels, not per-label.
        We create indexes on common labels used in GraphRAG.

        Args:
            graph_name: Name of the graph to create indexes on
        """
        try:
            logger.info(f"Creating indexes on graph '{graph_name}'")

            # Common node labels in GraphRAG
            node_labels = ['Entity', 'Document', 'CommunityReport', 'Person', 'Organization']

            # Create range indexes for searchable properties on each label
            for label in node_labels:
                for prop in self.SEARCHABLE_PROPERTIES:
                    # OpenCypher standard syntax for FalkorDB
                    query = f"CREATE INDEX FOR (n:{label}) ON (n.{prop})"
                    try:
                        self._execute_query(query, namespace=graph_name)
                        logger.debug(f"Created index on {label}.{prop} for graph '{graph_name}'")
                    except Exception as e:
                        # Index might already exist or property doesn't exist yet
                        # This is expected and not an error
                        logger.debug(f"Index creation skipped for {label}.{prop}: {e}")

            logger.info(f"Indexes ensured on graph '{graph_name}'")
        except Exception as e:
            logger.error(f"Error ensuring indexes on graph '{graph_name}': {e}")

    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute a FalkorDB query (openCypher) and return results.
        
        Args:
            query: openCypher query string
            parameters: Optional query parameters
            namespace: Optional namespace to select graph
            
        Returns:
            List of result records as dictionaries
        """
        graph = self._get_graph(namespace)
        logger.debug(f"Executing query on graph '{graph.name}': {query}, params: {parameters}")
        
        try:
            result = graph.query(query, params=parameters if parameters else None)
            
            # Convert result to list of dicts
            records = []
            header = result.header
            # Header might be simple list of strings or list of [type, name]
            clean_header = []
            if header:
                for col in header:
                    if isinstance(col, list) and len(col) > 1:
                        clean_header.append(col[1])
                    else:
                        clean_header.append(col)
            
            for record in result.result_set:
                record_dict = {}
                for idx, key in enumerate(clean_header):
                    record_dict[key] = self._convert_database_value(record[idx])
                records.append(record_dict)
            return records
            
        except Exception as e:
            logger.error(f"Query failed on graph '{graph.name}': {e}")
            raise

    def _convert_database_value(self, value) -> Any:
        """
        Convert FalkorDB-specific types to Python serializable types.
        """
        if value is None:
            return None
        
        # Handle FalkorDB Node objects
        if FALKORDB_AVAILABLE and FalkorNode is not None and isinstance(value, FalkorNode):
            return {
                "id": value.id,
                "labels": value.labels,
                "properties": self._convert_database_value(value.properties)
            }
        
        # Handle FalkorDB Edge (Relationship) objects
        if FALKORDB_AVAILABLE and FalkorEdge is not None and isinstance(value, FalkorEdge):
            return {
                "id": value.id,
                "type": value.relation,
                "properties": self._convert_database_value(value.properties),
                "source_id": value.src_node,
                "target_id": value.dest_node
            }
        
        # Handle FalkorDB Path objects
        if FALKORDB_AVAILABLE and FalkorPath is not None and isinstance(value, FalkorPath):
            return {
                "nodes": [self._convert_database_value(node) for node in value.nodes()],
                "edges": [self._convert_database_value(edge) for edge in value.edges()]
            }
        
        # Handle lists and tuples recursively
        if isinstance(value, (list, tuple)):
            return [self._convert_database_value(v) for v in value]
        
        # Handle dictionaries recursively
        if isinstance(value, dict):
            return {k: self._convert_database_value(v) for k, v in value.items()}
        
        return value

    def ensure_indexes(self):
        """
        Ensure indexes on the default graph.
        Note: With per-namespace strategy, indexes are created automatically
        on first access to each graph via _get_graph().
        This method ensures indexes on the default graph only.
        """
        self._ensure_indexes_for_graph(self._default_graph_name)

    def close(self):
        """Close the connection pool"""
        if FalkorGraphRepository._client:
            try:
                # FalkorDB client manages connection pool internally
                # Close will release all connections
                logger.info("Closing FalkorDB connection pool")
                FalkorGraphRepository._client = None
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")

    @classmethod
    def get_connection_stats(cls) -> Dict[str, Any]:
        """
        Get connection pool statistics for monitoring.

        Returns:
            Dictionary with connection pool stats
        """
        if not cls._client:
            return {"status": "not_initialized"}

        try:
            # Try to get pool info if available
            return {
                "status": "connected",
                "indexed_graphs": len(cls._indexed_graphs),
                "graphs": list(cls._indexed_graphs)
            }
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {"status": "error", "error": str(e)}

    def delete_graph(self, namespace: str) -> bool:
        """
        Delete an entire graph (all data for a namespace/client).
        This is a destructive operation - use with caution!

        Perfect for:
        - Removing all data for a client
        - Cleanup after testing
        - GDPR compliance (complete data removal)

        Args:
            namespace: The namespace (graph name) to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if not namespace:
                logger.error("Cannot delete graph: namespace is required")
                return False

            if namespace == self._default_graph_name:
                logger.warning(f"Attempting to delete default graph '{namespace}'")

            graph = self._client.select_graph(namespace)
            graph.delete()

            # Remove from indexed graphs tracking
            if namespace in FalkorGraphRepository._indexed_graphs:
                FalkorGraphRepository._indexed_graphs.remove(namespace)

            logger.info(f"Successfully deleted graph '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete graph '{namespace}': {e}")
            return False

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a query on the DEFAULT graph.
        """
        return self._execute_query(query, parameters)

    # ==================== NODE OPERATIONS ====================

    def create_node(self, node: Node, namespace: Optional[str] = None, 
                   index_name: Optional[str] = None, engine_name: Optional[str] = None, 
                   engine_type: Optional[str] = None, metadata_id: Optional[str] = None) -> Node:
        """
        Create a new node in the graph.
        """
        properties = dict(node.properties) if node.properties else {}
        if namespace is not None:
            properties["namespace"] = namespace
        if index_name is not None:
            properties["index_name"] = index_name
        if engine_name is not None:
            properties["engine_name"] = engine_name
        if engine_type is not None:
            properties["engine_type"] = engine_type
        if metadata_id is not None:
            properties["metadata_id"] = metadata_id
        
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        
        query = f"""
        CREATE (n:{node.label} {{{props_str}}})
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        """
        
        result = self._execute_query(query, properties, namespace=namespace)
        if result:
            record = result[0]
            return Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else node.label,
                properties=self._convert_properties(dict(record["properties"]))
            )
        raise RuntimeError("Failed to create node")

    def find_node_by_id(self, node_id: str, namespace: Optional[str] = None) -> Optional[Node]:
        """
        Find a node by its internal FalkorDB ID.

        Args:
            node_id: Internal node ID (numeric string or int)
            namespace: Namespace/graph to search in. If None, uses default graph.

        Returns:
            Node if found, None otherwise
        """
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        """

        result = self._execute_query(query, {"node_id": int(node_id)}, namespace=namespace)
        if result:
            record = result[0]
            return Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else "Unknown",
                properties=self._convert_properties(dict(record["properties"]))
            )
        return None

    def find_nodes_by_label(self, label: str, limit: int = 100, 
                           namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
        """
        Find all nodes with a specific label.
        """
        where_clauses = []
        params = {"limit": limit}
        if namespace is not None:
            where_clauses.append("n.namespace = $namespace")
            params["namespace"] = namespace
        if index_name is not None:
            where_clauses.append("n.index_name = $index_name")
            params["index_name"] = index_name
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (n:{label})
        {where_clause}
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        LIMIT $limit
        """
        
        result = self._execute_query(query, params, namespace=namespace)
        nodes = []
        for record in result:
            nodes.append(Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else label,
                properties=self._convert_properties(dict(record["properties"]))
            ))
        return nodes

    def find_nodes_by_property(self, label: str, property_key: str, property_value: Any, 
                              limit: int = 100, namespace: Optional[str] = None, 
                              index_name: Optional[str] = None) -> List[Node]:
        """
        Find nodes by a specific property value.
        """
        where_clauses = [f"n.{property_key} = $value"]
        params = {"value": property_value, "limit": limit}
        if namespace is not None:
            where_clauses.append("n.namespace = $namespace")
            params["namespace"] = namespace
        if index_name is not None:
            where_clauses.append("n.index_name = $index_name")
            params["index_name"] = index_name
        
        where_clause = "WHERE " + " AND ".join(where_clauses)
        query = f"""
        MATCH (n:{label})
        {where_clause}
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        LIMIT $limit
        """
        
        result = self._execute_query(query, params, namespace=namespace)
        nodes = []
        for record in result:
            nodes.append(Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else label,
                properties=self._convert_properties(dict(record["properties"]))
            ))
        return nodes

    def update_node(self, node_id: str, label: Optional[str] = None,
                   properties: Optional[Dict[str, Any]] = None,
                   namespace: Optional[str] = None) -> Optional[Node]:
        """
        Update a node's label and/or properties.

        Args:
            node_id: Internal node ID
            label: New label (currently not supported - would require recreate)
            properties: Properties to update
            namespace: Namespace/graph where the node exists

        Returns:
            Updated node if found, None otherwise
        """
        existing_node = self.find_node_by_id(node_id, namespace=namespace)
        if not existing_node:
            return None

        if properties:
            set_clauses = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"""
            MATCH (n)
            WHERE id(n) = $node_id
            SET {set_clauses}
            RETURN id(n) as id, labels(n) as labels, properties(n) as properties
            """
            params = {"node_id": int(node_id), **properties}
            self._execute_query(query, params, namespace=namespace)

        return self.find_node_by_id(node_id, namespace=namespace)

    def delete_node(self, node_id: str, detach: bool = True, namespace: Optional[str] = None) -> bool:
        """
        Delete a node from the graph.

        Args:
            node_id: Internal node ID
            detach: If True, delete all relationships before deleting node
            namespace: Namespace/graph where the node exists

        Returns:
            True if deleted, False if not found
        """
        if detach:
            query = """
            MATCH (n)
            WHERE id(n) = $node_id
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
        else:
            query = """
            MATCH (n)
            WHERE id(n) = $node_id
            DELETE n
            RETURN count(n) as deleted_count
            """

        result = self._execute_query(query, {"node_id": int(node_id)}, namespace=namespace)
        if result:
            return result[0].get("deleted_count", 0) > 0
        return False

    # ==================== RELATIONSHIP OPERATIONS ====================

    def create_relationship(self, relationship: Relationship, namespace: Optional[str] = None,
                           index_name: Optional[str] = None, engine_name: Optional[str] = None,
                           engine_type: Optional[str] = None, metadata_id: Optional[str] = None) -> Relationship:
        """
        Create a relationship between two nodes.
        """
        properties = dict(relationship.properties) if relationship.properties else {}
        if namespace is not None:
            properties["namespace"] = namespace
        if index_name is not None:
            properties["index_name"] = index_name
        if engine_name is not None:
            properties["engine_name"] = engine_name
        if engine_type is not None:
            properties["engine_type"] = engine_type
        if metadata_id is not None:
            properties["metadata_id"] = metadata_id
        
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()]) if properties else ""
        rel_clause = f"[r:{relationship.type} {{{props_str}}}]" if props_str else f"[r:{relationship.type}]"
        
        query = f"""
        MATCH (source), (target)
        WHERE id(source) = $source_id AND id(target) = $target_id
        CREATE (source)-{rel_clause}->(target)
        RETURN id(r) as id, type(r) as type, properties(r) as properties,
               id(source) as source_id, id(target) as target_id
        """
        
        params = {
            "source_id": int(relationship.source_id),
            "target_id": int(relationship.target_id),
            **properties
        }
        
        result = self._execute_query(query, params, namespace=namespace)
        if result:
            record = result[0]
            return Relationship(
                id=str(record["id"]),
                source_id=str(record["source_id"]),
                target_id=str(record["target_id"]),
                type=record["type"],
                properties=self._convert_properties(dict(record["properties"]))
            )
        raise RuntimeError("Failed to create relationship. Check if source and target nodes exist.")

    def find_relationship_by_id(self, relationship_id: str, namespace: Optional[str] = None) -> Optional[Relationship]:
        """
        Find a relationship by its internal FalkorDB ID.

        Args:
            relationship_id: Internal relationship ID
            namespace: Namespace/graph to search in

        Returns:
            Relationship if found, None otherwise
        """
        query = """
        MATCH (source)-[r]->(target)
        WHERE id(r) = $relationship_id
        RETURN id(r) as id, type(r) as type, properties(r) as properties,
               id(source) as source_id, id(target) as target_id
        """

        result = self._execute_query(query, {"relationship_id": int(relationship_id)}, namespace=namespace)
        if result:
            record = result[0]
            return Relationship(
                id=str(record["id"]),
                source_id=str(record["source_id"]),
                target_id=str(record["target_id"]),
                type=record["type"],
                properties=self._convert_properties(dict(record["properties"]))
            )
        return None

    def find_relationships_by_node(self, node_id: str, direction: str = "both",
                                   namespace: Optional[str] = None) -> List[Relationship]:
        """
        Find all relationships connected to a node.

        Args:
            node_id: Internal node ID
            direction: "incoming", "outgoing", or "both"
            namespace: Namespace/graph to search in

        Returns:
            List of relationships
        """
        if direction == "outgoing":
            query = """
            MATCH (source)-[r]->(target)
            WHERE id(source) = $node_id
            RETURN id(r) as id, type(r) as type, properties(r) as properties,
                   id(source) as source_id, id(target) as target_id
            """
        elif direction == "incoming":
            query = """
            MATCH (source)-[r]->(target)
            WHERE id(target) = $node_id
            RETURN id(r) as id, type(r) as type, properties(r) as properties,
                   id(source) as source_id, id(target) as target_id
            """
        else:  # both
            query = """
            MATCH (n)-[r]-(other)
            WHERE id(n) = $node_id
            RETURN id(r) as id, type(r) as type, properties(r) as properties,
                   id(startNode(r)) as source_id, id(endNode(r)) as target_id
            """

        result = self._execute_query(query, {"node_id": int(node_id)}, namespace=namespace)
        relationships = []
        for record in result:
            relationships.append(Relationship(
                id=str(record["id"]),
                source_id=str(record["source_id"]),
                target_id=str(record["target_id"]),
                type=record["type"],
                properties=self._convert_properties(dict(record["properties"]))
            ))
        return relationships

    def update_relationship(self, relationship_id: str, rel_type: Optional[str] = None,
                           properties: Optional[Dict[str, Any]] = None,
                           namespace: Optional[str] = None) -> Optional[Relationship]:
        """
        Update a relationship's type and/or properties.

        Args:
            relationship_id: Internal relationship ID
            rel_type: New relationship type (may require recreation)
            properties: Properties to update
            namespace: Namespace/graph where the relationship exists

        Returns:
            Updated relationship if found, None otherwise
        """
        existing_rel = self.find_relationship_by_id(relationship_id, namespace=namespace)
        if not existing_rel:
            return None

        if rel_type and rel_type != existing_rel.type:
            self.delete_relationship(relationship_id, namespace=namespace)
            new_rel = Relationship(
                source_id=existing_rel.source_id,
                target_id=existing_rel.target_id,
                type=rel_type,
                properties=properties if properties else existing_rel.properties
            )
            return self.create_relationship(new_rel, namespace=namespace)

        if properties:
            set_clauses = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
            query = f"""
            MATCH ()-[r]->()
            WHERE id(r) = $relationship_id
            SET {set_clauses}
            RETURN id(r) as id, type(r) as type, properties(r) as properties,
                   id(startNode(r)) as source_id, id(endNode(r)) as target_id
            """
            params = {"relationship_id": int(relationship_id), **properties}
            self._execute_query(query, params, namespace=namespace)

        return self.find_relationship_by_id(relationship_id, namespace=namespace)

    def delete_relationship(self, relationship_id: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a relationship from the graph.

        Args:
            relationship_id: Internal relationship ID
            namespace: Namespace/graph where the relationship exists

        Returns:
            True if deleted, False if not found
        """
        query = """
        MATCH ()-[r]->()
        WHERE id(r) = $relationship_id
        DELETE r
        RETURN count(r) as deleted_count
        """

        result = self._execute_query(query, {"relationship_id": int(relationship_id)}, namespace=namespace)
        if result:
            return result[0].get("deleted_count", 0) > 0
        return False

    def delete_nodes_by_metadata(
        self,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        engine_name: Optional[str] = None,
        engine_type: Optional[str] = None,
        metadata_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Delete all nodes (Entity and CommunityReport) that match the specified metadata.
        """
        where_clauses = []
        parameters = {}
        
        if namespace is not None:
            where_clauses.append("n.namespace = $namespace")
            parameters["namespace"] = namespace
        if index_name is not None:
            where_clauses.append("n.index_name = $index_name")
            parameters["index_name"] = index_name
        if engine_name is not None:
            where_clauses.append("n.engine_name = $engine_name")
            parameters["engine_name"] = engine_name
        if engine_type is not None:
            where_clauses.append("n.engine_type = $engine_type")
            parameters["engine_type"] = engine_type
        if metadata_id is not None:
            where_clauses.append("n.metadata_id = $metadata_id")
            parameters["metadata_id"] = metadata_id
        
        if not where_clauses:
            logger.warning("No metadata parameters provided, skipping delete")
            return {"nodes_deleted": 0}
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        MATCH (n)
        WHERE {where_clause}
        WITH n LIMIT 10000
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        
        result = self._execute_query(query, parameters, namespace=namespace)
        deleted_count = result[0].get("deleted_count", 0) if result else 0
        
        logger.info(f"Deleted {deleted_count} nodes matching metadata")
        
        return {"nodes_deleted": deleted_count}

    # ==================== UTILITY OPERATIONS ====================

    def verify_connection(self) -> bool:
        """
        Verify the connection to FalkorDB is working.
        """
        try:
            if self._client is None:
                return False
            # Check default graph connection
            graph = self._get_graph()
            # Simple query to verify
            graph.query("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database (DEFAULT graph).
        """
        try:
            node_query = "MATCH (n) RETURN count(n) as node_count"
            node_result = self._execute_query(node_query)
            node_count = node_result[0].get("node_count", 0) if node_result else 0
            
            rel_query = "MATCH ()-[r]->() RETURN count(r) as relationship_count"
            rel_result = self._execute_query(rel_query)
            rel_count = rel_result[0].get("relationship_count", 0) if rel_result else 0
            
            return {
                "node_count": node_count,
                "relationship_count": rel_count,
                "graph_name": self._default_graph_name
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {
                "node_count": 0,
                "relationship_count": 0,
                "graph_name": self._default_graph_name,
                "error": str(e)
            }

    def search_nodes_by_text(self, search_text: str, limit: int = 10,
                            namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
        """
        Search nodes using a full-text index.
        """
        where_clauses = []
        params = {"search_text": f"%{search_text}%", "limit": limit}
        if namespace is not None:
            where_clauses.append("n.namespace = $namespace")
            params["namespace"] = namespace
        if index_name is not None:
            where_clauses.append("n.index_name = $index_name")
            params["index_name"] = index_name
        
        property_conditions = []
        for prop in self.SEARCHABLE_PROPERTIES:
            property_conditions.append(f"n.{prop} CONTAINS $search_text")
        
        property_clause = " OR ".join(property_conditions)
        if property_clause:
            where_clauses.append(f"({property_clause})")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (n)
        {where_clause}
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        LIMIT $limit
        """
        
        result = self._execute_query(query, params, namespace=namespace)
        nodes = []
        for record in result:
            nodes.append(Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else "Unknown",
                properties=self._convert_properties(dict(record["properties"]))
            ))
        return nodes

    def search_relationships_by_text(self, search_text: str, limit: int = 10,
                                    namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search relationships using a full-text index.
        """
        where_clauses = []
        params = {"search_text": f"%{search_text}%", "limit": limit}
        if namespace is not None:
            where_clauses.append("r.namespace = $namespace")
            params["namespace"] = namespace
        if index_name is not None:
            where_clauses.append("r.index_name = $index_name")
            params["index_name"] = index_name
        
        property_conditions = []
        for prop in self.SEARCHABLE_REL_PROPERTIES:
            property_conditions.append(f"r.{prop} CONTAINS $search_text")
        
        property_clause = " OR ".join(property_conditions)
        if property_clause:
            where_clauses.append(f"({property_clause})")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (source)-[r]->(target)
        {where_clause}
        RETURN id(r) as id, type(r) as type, properties(r) as properties,
               id(source) as source_id, id(target) as target_id,
               labels(source) as source_labels, labels(target) as target_labels,
               properties(source) as source_props, properties(target) as target_props
        LIMIT $limit
        """
        
        result = self._execute_query(query, params, namespace=namespace)
        relationships = []
        for record in result:
            relationships.append({
                "id": str(record["id"]),
                "type": record["type"],
                "properties": self._convert_properties(dict(record["properties"])),
                "source": {
                    "id": str(record["source_id"]),
                    "label": record["source_labels"][0] if record["source_labels"] else "Unknown",
                    "properties": self._convert_properties(dict(record["source_props"]))
                },
                "target": {
                    "id": str(record["target_id"]),
                    "label": record["target_labels"][0] if record["target_labels"] else "Unknown",
                    "properties": self._convert_properties(dict(record["target_props"]))
                }
            })
        return relationships

    def search_community_reports(self, search_text: str, limit: int = 5) -> List[Node]:
        """
        Search community reports by text (DEFAULT graph).
        """
        query = """
        MATCH (n:CommunityReport)
        WHERE 
            (n.full_report IS NOT NULL AND n.full_report CONTAINS $search_text) OR
            (n.summary IS NOT NULL AND n.summary CONTAINS $search_text) OR
            (n.title IS NOT NULL AND n.title CONTAINS $search_text)
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        LIMIT $limit
        """
        
        result = self._execute_query(query, {"search_text": search_text, "limit": limit})
        nodes = []
        for record in result:
            nodes.append(Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else "CommunityReport",
                properties=self._convert_properties(dict(record["properties"]))
            ))
        return nodes

    def get_all_nodes_and_relationships(self, namespace: Optional[str] = None,
                                       index_name: Optional[str] = None,
                                       engine_name: Optional[str] = None,
                                       engine_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch the entire graph, optionally filtered by namespace, index_name, engine_name and engine_type.
        """
        where_clauses = []
        params = {}
        if namespace is not None:
            where_clauses.append("n.namespace = $namespace")
            params["namespace"] = namespace
        if index_name is not None:
            where_clauses.append("n.index_name = $index_name")
            params["index_name"] = index_name
        if engine_name is not None:
            where_clauses.append("n.engine_name = $engine_name")
            params["engine_name"] = engine_name
        if engine_type is not None:
            where_clauses.append("n.engine_type = $engine_type")
            params["engine_type"] = engine_type
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (n)
        {where_clause}
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        
        result = self._execute_query(query, params, namespace=namespace)
        
        nodes = {}
        relationships = []
        seen_nodes = set()
        seen_rels = set()
        
        for record in result:
            n = record.get("n")
            if n:
                n_id = str(n.get("id"))
                if n_id not in seen_nodes:
                    seen_nodes.add(n_id)
                    nodes[n_id] = {
                        "id": n_id,
                        "label": n.get("labels", ["Unknown"])[0],
                        "properties": n.get("properties", {})
                    }
            
            m = record.get("m")
            if m:
                m_id = str(m.get("id"))
                if m_id not in seen_nodes:
                    seen_nodes.add(m_id)
                    nodes[m_id] = {
                        "id": m_id,
                        "label": m.get("labels", ["Unknown"])[0],
                        "properties": m.get("properties", {})
                    }
            
            r = record.get("r")
            if r:
                r_id = str(r.get("id"))
                if r_id not in seen_rels:
                    seen_rels.add(r_id)
                    relationships.append({
                        "id": r_id,
                        "type": r.get("type", ""),
                        "properties": r.get("properties", {}),
                        "source_id": str(r.get("source_id", "")),
                        "target_id": str(r.get("target_id", ""))
                    })
        
        return {
            "nodes": list(nodes.values()),
            "relationships": relationships
        }

    def save_community_report(
        self,
        community_id: str,
        report: Dict[str, Any],
        level: int,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        engine_name: Optional[str] = None,
        engine_type: Optional[str] = None,
        metadata_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Save a community report as a node in FalkorDB.

        Args:
            community_id: Identifier for the community
            report: The generated report data (title, summary, findings, etc.)
            level: The hierarchy level
            namespace: Optional namespace to partition the graph
            index_name: Optional index_name (collection name) to partition the graph
            engine_name: Optional engine name (e.g., 'pinecone', 'qdrant') to partition the graph
            engine_type: Optional engine type (e.g., 'pod', 'serverless')
            metadata_id: Optional metadata ID from vector store chunks for cleanup

        Returns:
            The created node ID
        """
        # Build properties dictionary
        properties = {
            "community_id": community_id,
            "level": level,
            "title": report.get("title", ""),
            "summary": report.get("summary", ""),
            "rating": float(report.get("rating", 0.0)),
            "rating_explanation": report.get("rating_explanation", ""),
            "findings": json.dumps(report.get("findings", [])),
            "full_report": report.get("full_report", ""),  # Optional markdown version
            "import_timestamp": "datetime()"  # FalkorDB will evaluate this
        }

        # Add optional partitioning properties
        if namespace is not None:
            properties["namespace"] = namespace
        if index_name is not None:
            properties["index_name"] = index_name
        if engine_name is not None:
            properties["engine_name"] = engine_name
        if engine_type is not None:
            properties["engine_type"] = engine_type
        if metadata_id is not None:
            properties["metadata_id"] = metadata_id

        # Build property string for MERGE/SET
        prop_assignments = []
        for key in ["title", "summary", "rating", "rating_explanation", "findings", "full_report"]:
            if key in properties:
                prop_assignments.append(f"c.{key} = ${key}")

        # Add optional props
        for key in ["namespace", "index_name", "engine_name", "engine_type", "metadata_id"]:
            if key in properties:
                prop_assignments.append(f"c.{key} = ${key}")

        set_clause = ", ".join(prop_assignments)

        # OpenCypher MERGE query for FalkorDB
        query = f"""
        MERGE (c:CommunityReport {{community_id: $community_id, level: $level}})
        SET {set_clause}
        RETURN id(c) as node_id
        """

        try:
            result = self._execute_query(query, properties, namespace=namespace)

            # _execute_query returns List[Dict], so access directly
            if result and len(result) > 0:
                report_node_id = result[0].get("node_id")  # First row, get node_id key
                logger.info(f"Saved community report for community_id={community_id}, level={level}, node_id={report_node_id}")

                # Link entities in the community to this report (BELONGS_TO_COMMUNITY relationships)
                entities = report.get("entities", [])
                if entities and report_node_id:
                    try:
                        # Convert entity IDs to integers (FalkorDB uses integer IDs)
                        entity_ids = [int(eid) if isinstance(eid, str) else eid for eid in entities]

                        # FalkorDB version: use id() instead of elementId()
                        link_query = """
                        UNWIND $entity_ids as entity_id
                        MATCH (e) WHERE id(e) = entity_id
                        MATCH (c) WHERE id(c) = $report_id
                        MERGE (e)-[:BELONGS_TO_COMMUNITY]->(c)
                        """
                        result = self._execute_query(
                            link_query,
                            {"entity_ids": entity_ids, "report_id": int(report_node_id)},
                            namespace=namespace
                        )
                        logger.info(f"Linked {len(entity_ids)} entities to community report {report_node_id}")
                    except Exception as link_error:
                        logger.error(f"Failed to link entities to community report: {link_error}", exc_info=True)
                        # Don't fail the entire operation if linking fails

                return str(report_node_id)
            else:
                logger.warning(f"Failed to save community report for community_id={community_id}")
                return None

        except Exception as e:
            logger.error(f"Error saving community report: {e}", exc_info=True)
            return None

    def get_community_network(self, namespace: str, index_name: str, limit: int = 1000) -> Dict[str, Any]:
        """
        Fetch the community graph (nodes + BELONGS_TO_COMMUNITY relationships).
        """
        query = """
        MATCH p=(n)-[r:BELONGS_TO_COMMUNITY]->(m)
        WHERE n.namespace = $namespace AND m.namespace = $namespace 
          AND n.index_name = $index_name AND m.index_name = $index_name
        RETURN p
        LIMIT $limit
        """
        
        result = self._execute_query(query, {"namespace": namespace, "index_name": index_name, "limit": limit}, namespace=namespace)
        
        # Simplified response
        return {
            "nodes": [],
            "relationships": []
        }