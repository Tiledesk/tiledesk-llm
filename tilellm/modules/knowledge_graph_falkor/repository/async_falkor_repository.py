"""
Async FalkorDB Repository for Knowledge Graph operations.
Provides async CRUD operations for nodes and relationships with connection pooling.
Uses FalkorDB async client with Redis async connection pool.

Perfect for applications with:
- High concurrent load
- Multiple simultaneous clients
- I/O-bound operations
- FastAPI async endpoints
"""

import json
import logging
from typing import Optional, List, Dict, Any, Union, cast
import os

from tilellm.shared.utility import get_service_config
from tilellm.store.graph import BaseGraphRepository
from tilellm.modules.knowledge_graph.models import Node, Relationship

logger = logging.getLogger(__name__)

try:
    from falkordb.asyncio import FalkorDB
    from falkordb import Node as FalkorNode, Edge as FalkorEdge, Path as FalkorPath
    from redis.asyncio import BlockingConnectionPool
    FALKORDB_ASYNC_AVAILABLE = True
except ImportError:
    FalkorDB = None
    FalkorNode = None
    FalkorEdge = None
    FalkorPath = None
    BlockingConnectionPool = None
    FALKORDB_ASYNC_AVAILABLE = False
    logger.warning("FalkorDB async client not available. Please install 'falkordb' package.")


class AsyncFalkorGraphRepository(BaseGraphRepository):
    """
    Async repository for interacting with FalkorDB graph database.
    Uses FalkorDB async Python client with connection pooling.

    Strategy: One graph per namespace for perfect multi-tenant isolation.
    Indexes are created per-graph on first access.

    Performance benefits:
    - Concurrent query execution with asyncio
    - Connection pooling with BlockingConnectionPool
    - Non-blocking I/O for multiple clients
    - Automatic connection health checks
    """

    _client: Optional[Any] = None
    _pool: Optional[Any] = None
    _default_graph_name: str = "knowledge_graph"
    _indexed_graphs: set = set()  # Track graphs with indexes already created

    def __init__(self):
        """
        Initialize async FalkorDB connection using environment variables.
        Uses the official FalkorDB async Python client with connection pooling.
        """
        if AsyncFalkorGraphRepository._client is None:
            if not FALKORDB_ASYNC_AVAILABLE:
                raise ImportError("FalkorDB async client not installed. Please install with 'pip install falkordb'")

            service_config = get_service_config()
            falkordb_config = service_config.get("falkordb") or service_config.get("redis")

            if not falkordb_config:
                raise ValueError("FalkorDB configuration not found in environment variables. "
                               "Set FALKORDB_URI or REDIS_URI.")

            # FalkorDB connection parameters
            host = falkordb_config.get("host", "localhost")
            port = falkordb_config.get("port", 6379)
            password = falkordb_config.get("password")
            username = falkordb_config.get("username")

            graph_name = falkordb_config.get("graph_name", "knowledge_graph")

            # Async connection pool configuration
            max_connections = falkordb_config.get("max_connections", 50)
            timeout = falkordb_config.get("timeout", None)  # None = no timeout

            # Override with environment variable if present
            env_uri = os.environ.get("FALKORDB_URI")

            AsyncFalkorGraphRepository._default_graph_name = graph_name

            try:
                logger.info(f"Initializing async FalkorDB client for host={host}:{port} with max_connections={max_connections}")

                # Create async connection pool
                pool_kwargs = {
                    "max_connections": max_connections,
                    "timeout": timeout,
                    "decode_responses": True,
                    "host": host,
                    "port": port,
                }

                if password:
                    pool_kwargs["password"] = password
                if username:
                    pool_kwargs["username"] = username

                AsyncFalkorGraphRepository._pool = BlockingConnectionPool(**pool_kwargs)

                # Create async FalkorDB client with pool
                AsyncFalkorGraphRepository._client = FalkorDB(
                    connection_pool=AsyncFalkorGraphRepository._pool
                )

                logger.info(f"Async FalkorDB connection pool initialized with {max_connections} max connections")

            except Exception as e:
                logger.error(f"Failed to initialize async FalkorDB connection: {e}")
                raise ConnectionError(f"Failed to connect to async FalkorDB: {e}")

    def _get_graph_name(self, namespace: Optional[str] = None, index_name: Optional[str] = None, graph_name: Optional[str] = None) -> str:
        """
        Determine graph name based on namespace-first strategy.

        Args:
            namespace: Client/tenant namespace (e.g., "cliente_A")
            index_name: Collection name (stored as property, not used for graph selection)
            graph_name: Optional explicit graph name (overrides namespace)

        Returns:
            Graph name to use
        """
        if graph_name:
            return graph_name
        if namespace:
            return namespace
        return self._default_graph_name

    def _get_graph(self, namespace: Optional[str] = None, index_name: Optional[str] = None, graph_name: Optional[str] = None):
        """
        Select the graph based on namespace and ensure indexes exist.
        Creates indexes on first access to each graph (lazy initialization).

        Note: Index creation is sync in _ensure_indexes_for_graph.
        Call await ensure_indexes_async() if you need async index creation.

        Args:
            namespace: Client/tenant namespace
            index_name: Collection name (for tracking only)
            graph_name: Optional explicit graph name (overrides namespace)

        Returns:
            FalkorDB Graph instance
        """
        graph_name_to_use = self._get_graph_name(namespace, index_name, graph_name)
        graph = self._client.select_graph(graph_name_to_use)

        # Note: We skip automatic index creation in async to avoid blocking
        # Call ensure_indexes_async() explicitly if needed

        return graph

    async def _ensure_indexes_for_graph(self, graph_name: str, node_labels: Optional[List[str]] = None):
        """
        Create indexes on a specific graph using FalkorDB openCypher syntax (async).
        Called explicitly when needed.

        FalkorDB uses standard openCypher syntax for indexes:
        CREATE INDEX FOR (n:Label) ON (n.property)

        Args:
            graph_name: Name of the graph to create indexes on
            node_labels: Optional list of node labels to create indexes for.
                        If None, uses default labels: ['Entity', 'Document', 'CommunityReport', 'Person', 'Organization']
        """
        try:
            logger.info(f"Creating indexes on graph '{graph_name}' (async)")

            # Common node labels in GraphRAG
            if node_labels is None:
                node_labels = ['Entity', 'Document', 'CommunityReport', 'Person', 'Organization']

            # Create range indexes for searchable properties on each label
            for label in node_labels:
                for prop in self.SEARCHABLE_PROPERTIES:
                    # OpenCypher standard syntax for FalkorDB
                    query = f"CREATE INDEX FOR (n:{label}) ON (n.{prop})"
                    try:
                        await self._execute_query(query, namespace=graph_name)
                        logger.debug(f"Created index on {label}.{prop} for graph '{graph_name}'")
                    except Exception as e:
                        # Index might already exist or property doesn't exist yet
                        logger.debug(f"Index creation skipped for {label}.{prop}: {e}")

            # Track indexed graphs
            AsyncFalkorGraphRepository._indexed_graphs.add(graph_name)

            logger.info(f"Indexes ensured on graph '{graph_name}'")
        except Exception as e:
            logger.error(f"Error ensuring indexes on graph '{graph_name}': {e}")

    async def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                            namespace: Optional[str] = None, graph_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute a FalkorDB query (openCypher) asynchronously.

        Args:
            query: openCypher query string
            parameters: Optional query parameters
            namespace: Optional namespace to select graph
            graph_name: Optional explicit graph name (overrides namespace)

        Returns:
            List of result records as dictionaries
        """
        graph = self._get_graph(namespace=namespace, graph_name=graph_name)
        logger.debug(f"Executing async query on graph '{graph.name}': {query}, params: {parameters}")

        try:
            result = await graph.query(query, params=parameters if parameters else None)

            # Convert result to list of dicts
            records = []
            header = result.header
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
            logger.error(f"Async query failed on graph '{graph.name}': {e}")
            raise

    def _convert_database_value(self, value) -> Any:
        """
        Convert FalkorDB-specific types to Python serializable types.
        This method is sync as it's just data conversion.
        """
        if value is None:
            return None

        # Handle FalkorDB Node objects
        if FALKORDB_ASYNC_AVAILABLE and FalkorNode is not None and isinstance(value, FalkorNode):
            return {
                "id": value.id,
                "labels": value.labels,
                "properties": self._convert_database_value(value.properties)
            }

        # Handle FalkorDB Edge (Relationship) objects
        if FALKORDB_ASYNC_AVAILABLE and FalkorEdge is not None and isinstance(value, FalkorEdge):
            return {
                "id": value.id,
                "type": value.relation,
                "properties": self._convert_database_value(value.properties),
                "source_id": value.src_node,
                "target_id": value.dest_node
            }

        # Handle FalkorDB Path objects
        if FALKORDB_ASYNC_AVAILABLE and FalkorPath is not None and isinstance(value, FalkorPath):
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

    # ==================== LIFECYCLE METHODS ====================

    def ensure_indexes(self):
        """
        Sync wrapper - not recommended for async code.
        Use await ensure_indexes_async() instead.
        """
        raise NotImplementedError("Use 'await ensure_indexes_async()' for async repository")

    async def ensure_indexes_async(self):
        """
        Ensure indexes on the default graph (async).
        """
        await self._ensure_indexes_for_graph(self._default_graph_name)

    async def ensure_indexes_for_graph(self, graph_name: str, node_labels: Optional[List[str]] = None):
        """
        Ensure that required indexes exist for a specific graph.
        
        Args:
            graph_name: Name of the graph to ensure indexes for
            node_labels: Optional list of node labels to create indexes for.
                        If None, uses default entity types from the domain.
        """
        await self._ensure_indexes_for_graph(graph_name, node_labels)

    async def close(self):
        """Close the async connection pool"""
        if AsyncFalkorGraphRepository._pool:
            try:
                logger.info("Closing async FalkorDB connection pool")
                await AsyncFalkorGraphRepository._pool.aclose()
                AsyncFalkorGraphRepository._pool = None
                AsyncFalkorGraphRepository._client = None
            except Exception as e:
                logger.error(f"Error closing async connection pool: {e}")

    #def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                          namespace: Optional[str] = None,
                          graph_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute a FalkorDB query (openCypher) asynchronously.

        Args:
            query: openCypher query string
            parameters: Optional query parameters
            namespace: Optional namespace to select graph
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            List of result records as dictionaries
        """
        graph = self._get_graph(namespace=namespace, graph_name=graph_name)
        logger.debug(f"Executing async query on graph '{graph.name}': {query}, params: {parameters}")

        try:
            # Esecuzione asincrona vera
            result = await graph.query(query, params=parameters if parameters else None)

            # Conversione risultati (CPU bound, può rimanere sincrona)
            records = []
            header = result.header
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
            logger.error(f"Async query failed on graph '{graph.name}': {e}")
            raise

    @classmethod
    def get_connection_stats(cls) -> Dict[str, Any]:
        """
        Get connection pool statistics for monitoring (sync).

        Returns:
            Dictionary with connection pool stats
        """
        if not cls._client:
            return {"status": "not_initialized"}

        try:
            return {
                "status": "connected",
                "indexed_graphs": len(cls._indexed_graphs),
                "graphs": list(cls._indexed_graphs)
            }
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {"status": "error", "error": str(e)}

    async def delete_graph(self, namespace: str) -> bool:
        """
        Delete an entire graph (all data for a namespace/client) - async.

        Args:
            namespace: The namespace (graph name) to delete

        Returns:
            True if deletion successful
        """
        try:
            if not namespace:
                logger.error("Cannot delete graph: namespace is required")
                return False

            if namespace == self._default_graph_name:
                logger.warning(f"Attempting to delete default graph '{namespace}'")

            graph = self._client.select_graph(namespace)
            await graph.delete()

            # Remove from indexed graphs tracking
            if namespace in AsyncFalkorGraphRepository._indexed_graphs:
                AsyncFalkorGraphRepository._indexed_graphs.remove(namespace)

            logger.info(f"Successfully deleted graph '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete graph '{namespace}': {e}")
            return False

    # ==================== NODE OPERATIONS ====================

    async def create_node(self, node: Node, namespace: Optional[str] = None,
                         index_name: Optional[str] = None, engine_name: Optional[str] = None,
                         engine_type: Optional[str] = None, metadata_id: Optional[str] = None,
                         graph_name: Optional[str] = None) -> Node:
        """
        Create a new node in the graph (async).
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

        result = await self._execute_query(query, properties, namespace=namespace, graph_name=graph_name)
        if result:
            record = result[0]
            return Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else node.label,
                properties=self._convert_properties(dict(record["properties"]))
            )
        raise RuntimeError("Failed to create node")

    async def find_node_by_id(self, node_id: str, namespace: Optional[str] = None, graph_name: Optional[str] = None) -> Optional[Node]:
        """
        Find a node by its internal FalkorDB ID (async).
        """
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        """

        result = await self._execute_query(query, {"node_id": int(node_id)}, namespace=namespace, graph_name=graph_name)
        if result:
            record = result[0]
            return Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else "Unknown",
                properties=self._convert_properties(dict(record["properties"]))
            )
        return None

    async def find_nodes_by_label(self, label: str, limit: int = 100,
                                 namespace: Optional[str] = None, index_name: Optional[str] = None,
                                 graph_name: Optional[str] = None) -> List[Node]:
        """
        Find all nodes with a specific label (async).
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

        result = await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)
        nodes = []
        for record in result:
            nodes.append(Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else label,
                properties=self._convert_properties(dict(record["properties"]))
            ))
        return nodes

    async def find_nodes_by_property(self, label: str, property_key: str, property_value: Any,
                                     limit: int = 100, namespace: Optional[str] = None,
                                     index_name: Optional[str] = None, graph_name: Optional[str] = None) -> List[Node]:
        """
        Find nodes by a specific property value (async).
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

        result = await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)
        nodes = []
        for record in result:
            nodes.append(Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else label,
                properties=self._convert_properties(dict(record["properties"]))
            ))
        return nodes

    async def find_nodes_by_source_id(self, source_id: str, limit: int = 10,
                                     namespace: Optional[str] = None,
                                     index_name: Optional[str] = None,
                                     graph_name: Optional[str] = None) -> List[Node]:
        """
        Find nodes that reference a specific source ID (vector store chunk ID).
        Checks 'source_ids' list property (async).

        Args:
            source_id: The source ID to look for
            limit: Maximum number of nodes to return
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes
            graph_name: Optional explicit graph name (overrides namespace for graph selection)

        Returns:
            List of matching nodes
        """
        # Build WHERE clause - check if source_id is in source_ids list
        where_clauses = ["$source_id IN n.source_ids"]
        params: Dict[str, Any] = {"source_id": source_id, "limit": limit}

        if namespace is not None:
            where_clauses.append("n.namespace = $namespace")
            params["namespace"] = namespace
        if index_name is not None:
            where_clauses.append("n.index_name = $index_name")
            params["index_name"] = index_name

        where_clause = "WHERE " + " AND ".join(where_clauses)
        query = f"""
        MATCH (n)
        {where_clause}
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        LIMIT $limit
        """

        logger.debug(f"Searching for nodes with source_id: {source_id}, namespace: {namespace}")
        logger.info(f"QUERY=====================> {query} {namespace} {graph_name} {index_name} {limit}")

        result = await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)

        nodes = []
        for record in result:
            label = record["labels"][0] if record["labels"] else "Unknown"
            nodes.append(Node(
                id=str(record["id"]),
                label=label,
                properties=self._convert_properties(dict(record["properties"]))
            ))

        logger.info(f"Total nodes found for source_id {source_id}: {len(nodes)}")
        return nodes

    async def update_node(self, node_id: str, label: Optional[str] = None,
                         properties: Optional[Dict[str, Any]] = None,
                         namespace: Optional[str] = None, graph_name: Optional[str] = None) -> Optional[Node]:
        """
        Update a node's properties (async).
        """
        existing_node = await self.find_node_by_id(node_id, namespace=namespace, graph_name=graph_name)
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
            await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)

        return await self.find_node_by_id(node_id, namespace=namespace, graph_name=graph_name)

    async def delete_node(self, node_id: str, detach: bool = True, namespace: Optional[str] = None,
                         graph_name: Optional[str] = None) -> bool:
        """
        Delete a node from the graph (async).
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

        result = await self._execute_query(query, {"node_id": int(node_id)}, namespace=namespace, graph_name=graph_name)
        if result:
            return result[0].get("deleted_count", 0) > 0
        return False

    # ==================== RELATIONSHIP OPERATIONS ====================

    async def create_relationship(self, relationship: Relationship, namespace: Optional[str] = None,
                                 index_name: Optional[str] = None, engine_name: Optional[str] = None,
                                 engine_type: Optional[str] = None, metadata_id: Optional[str] = None,
                                 graph_name: Optional[str] = None) -> Relationship:
        """
        Create a relationship between two nodes (async).
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

        result = await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)
        if result:
            record = result[0]
            return Relationship(
                id=str(record["id"]),
                source_id=str(record["source_id"]),
                target_id=str(record["target_id"]),
                type=record["type"],
                properties=self._convert_properties(dict(record["properties"]))
            )
        raise RuntimeError("Failed to create relationship")

    async def find_relationship_by_id(self, relationship_id: str, namespace: Optional[str] = None, graph_name: Optional[str] = None) -> Optional[Relationship]:
        """
        Find a relationship by its internal FalkorDB ID (async).
        """
        query = """
        MATCH (source)-[r]->(target)
        WHERE id(r) = $relationship_id
        RETURN id(r) as id, type(r) as type, properties(r) as properties,
               id(source) as source_id, id(target) as target_id
        """

        result = await self._execute_query(query, {"relationship_id": int(relationship_id)}, namespace=namespace, graph_name=graph_name)
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

    async def find_relationships_by_node(self, node_id: str, direction: str = "both",
                                         namespace: Optional[str] = None, graph_name: Optional[str] = None) -> List[Relationship]:
        """
        Find all relationships connected to a node (async).
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

        result = await self._execute_query(query, {"node_id": int(node_id)}, namespace=namespace, graph_name=graph_name)
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

    async def update_relationship(self, relationship_id: str, rel_type: Optional[str] = None,
                                 properties: Optional[Dict[str, Any]] = None,
                                 namespace: Optional[str] = None, graph_name: Optional[str] = None) -> Optional[Relationship]:
        """
        Update a relationship's properties (async).
        """
        existing_rel = await self.find_relationship_by_id(relationship_id, namespace=namespace, graph_name=graph_name)
        if not existing_rel:
            return None

        if rel_type and rel_type != existing_rel.type:
            await self.delete_relationship(relationship_id, namespace=namespace, graph_name=graph_name)
            new_rel = Relationship(
                source_id=existing_rel.source_id,
                target_id=existing_rel.target_id,
                type=rel_type,
                properties=properties if properties else existing_rel.properties
            )
            return await self.create_relationship(new_rel, namespace=namespace, graph_name=graph_name)

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
            await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)

        return await self.find_relationship_by_id(relationship_id, namespace=namespace, graph_name=graph_name)

    async def delete_relationship(self, relationship_id: str, namespace: Optional[str] = None, graph_name: Optional[str] = None) -> bool:
        """
        Delete a relationship from the graph (async).
        """
        query = """
        MATCH ()-[r]->()
        WHERE id(r) = $relationship_id
        DELETE r
        RETURN count(r) as deleted_count
        """

        result = await self._execute_query(query, {"relationship_id": int(relationship_id)}, namespace=namespace, graph_name=graph_name)
        if result:
            return result[0].get("deleted_count", 0) > 0
        return False

    async def delete_nodes_by_metadata(self, namespace: Optional[str] = None,
                                       index_name: Optional[str] = None,
                                       engine_name: Optional[str] = None,
                                       engine_type: Optional[str] = None,
                                       metadata_id: Optional[str] = None,
                                       graph_name: Optional[str] = None) -> Dict[str, int]:
        """
        Delete all nodes that match the specified metadata (async).
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

        result = await self._execute_query(query, parameters, namespace=namespace, graph_name=graph_name)
        deleted_count = result[0].get("deleted_count", 0) if result else 0

        logger.info(f"Deleted {deleted_count} nodes matching metadata")

        return {"nodes_deleted": deleted_count}

    # ==================== UTILITY OPERATIONS ====================

    async def verify_connection(self) -> bool:
        """
        Verify the connection to FalkorDB is working (async).
        """
        try:
            if self._client is None:
                return False
            graph = self._get_graph()
            await graph.query("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return False

    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database (async).
        """
        try:
            node_query = "MATCH (n) RETURN count(n) as node_count"
            node_result = await self._execute_query(node_query)
            node_count = node_result[0].get("node_count", 0) if node_result else 0

            rel_query = "MATCH ()-[r]->() RETURN count(r) as relationship_count"
            rel_result = await self._execute_query(rel_query)
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

    async def search_nodes_by_text(self, search_text: str, limit: int = 10,
                                  namespace: Optional[str] = None, index_name: Optional[str] = None,
                                  graph_name: Optional[str] = None) -> List[Node]:
        """
        Search nodes using text match (async).
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

        result = await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)
        nodes = []
        for record in result:
            nodes.append(Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else "Unknown",
                properties=self._convert_properties(dict(record["properties"]))
            ))
        return nodes

    async def search_relationships_by_text(self, search_text: str, limit: int = 10,
                                          namespace: Optional[str] = None, index_name: Optional[str] = None,
                                          graph_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search relationships using text match (async).
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

        result = await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)
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

    async def search_community_reports(self, search_text: str, limit: int = 5,
                                      namespace: Optional[str] = None, graph_name: Optional[str] = None) -> List[Node]:
        """
        Search community reports by text (async).
        """
        where_clauses = []
        params = {"search_text": search_text, "limit": limit}
        
        text_conditions = [
            "(n.full_report IS NOT NULL AND n.full_report CONTAINS $search_text)",
            "(n.summary IS NOT NULL AND n.summary CONTAINS $search_text)",
            "(n.title IS NOT NULL AND n.title CONTAINS $search_text)"
        ]
        where_clauses.append(f"({' OR '.join(text_conditions)})")
        
        if namespace is not None:
            where_clauses.append("n.namespace = $namespace")
            params["namespace"] = namespace
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (n:CommunityReport)
        {where_clause}
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        LIMIT $limit
        """

        result = await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)
        nodes = []
        for record in result:
            nodes.append(Node(
                id=str(record["id"]),
                label=record["labels"][0] if record["labels"] else "CommunityReport",
                properties=self._convert_properties(dict(record["properties"]))
            ))
        return nodes

    async def get_all_nodes_and_relationships(self, namespace: Optional[str] = None,
                                             index_name: Optional[str] = None,
                                             engine_name: Optional[str] = None,
                                             engine_type: Optional[str] = None,
                                             graph_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch the entire graph with optional filters (async).
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

        result = await self._execute_query(query, params, namespace=namespace, graph_name=graph_name)

        nodes = {}
        relationships = []
        seen_nodes = set()
        seen_rels = set()

        for record in result:
            n = record.get("n")
            if n:
                n_labels = n.get("labels", ["Unknown"])
                if "CommunityReport" in n_labels:
                    continue  # Skip: community reports must not be clustered
                n_id = str(n.get("id"))
                if n_id not in seen_nodes:
                    seen_nodes.add(n_id)
                    nodes[n_id] = {
                        "id": n_id,
                        "label": n_labels[0],
                        "properties": n.get("properties", {})
                    }

            m = record.get("m")
            if m:
                m_labels = m.get("labels", ["Unknown"])
                if "CommunityReport" not in m_labels:
                    m_id = str(m.get("id"))
                    if m_id not in seen_nodes:
                        seen_nodes.add(m_id)
                        nodes[m_id] = {
                            "id": m_id,
                            "label": m_labels[0],
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

    async def save_community_report(
        self,
        community_id: str,
        report: Dict[str, Any],
        level: int,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        engine_name: Optional[str] = None,
        engine_type: Optional[str] = None,
        metadata_id: Optional[str] = None,
        graph_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Save a community report as a node in FalkorDB (async).

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
            result = await self._execute_query(query, properties, namespace=namespace, graph_name=graph_name)

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
                        result = await self._execute_query(
                            link_query,
                            {"entity_ids": entity_ids, "report_id": int(report_node_id)},
                            namespace=namespace,
                            graph_name=graph_name
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

    async def get_community_network(self, namespace: str, index_name: str, limit: int = 1000,
                                   graph_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch the community graph (async).
        """
        query = """
        MATCH p=(n)-[r:BELONGS_TO_COMMUNITY]->(m)
        WHERE n.namespace = $namespace AND m.namespace = $namespace
          AND n.index_name = $index_name AND m.index_name = $index_name
        RETURN p
        LIMIT $limit
        """

        result = await self._execute_query(query, {"namespace": namespace, "index_name": index_name, "limit": limit}, namespace=namespace, graph_name=graph_name)

        return {
            "nodes": [],
            "relationships": []
        }
