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

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize an entity name for deduplication: strip whitespace and lowercase."""
        return name.strip().lower()

    async def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                            namespace: Optional[str] = None, graph_name: Optional[str] = None,
                            timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute a FalkorDB query (openCypher) asynchronously.

        Args:
            query: openCypher query string
            parameters: Optional query parameters
            namespace: Optional namespace to select graph
            graph_name: Optional explicit graph name (overrides namespace)
            timeout: Optional per-query timeout in ms (overrides connection-level timeout)

        Returns:
            List of result records as dictionaries
        """
        graph = self._get_graph(namespace=namespace, graph_name=graph_name)
        logger.debug(f"Executing async query on graph '{graph.name}': {query}, params: {parameters}")

        try:
            result = await graph.query(query, params=parameters if parameters else None, timeout=timeout)

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

    async def batch_create_nodes(
        self,
        entities: List[Dict[str, Any]],
        namespace: str,
        index_name: Optional[str] = None,
        engine_name: Optional[str] = None,
        engine_type: Optional[str] = None,
        graph_name: Optional[str] = None,
        batch_size: int = 100,
        existing_entity_node_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Bulk-create entity nodes grouped by label using UNWIND.

        If existing_entity_node_map is provided (normalized_name → node_id),
        entities whose normalized name already appears in it are skipped and their
        existing node_id is returned directly — preventing duplicates across windows
        and across runs (Level 1+2 deduplication).

        Returns a dict mapping normalized_name → node_id for every node
        (newly created or reused).  Falls back to individual creates if UNWIND fails.
        """
        from collections import defaultdict
        from datetime import datetime

        entity_node_map: Dict[str, str] = {}
        timestamp = datetime.now().isoformat()
        existing = existing_entity_node_map or {}

        by_label: Dict[str, list] = defaultdict(list)
        reused = 0
        for entity in entities:
            norm = self._normalize_name(entity.get("entity_name", ""))
            if norm and norm in existing:
                entity_node_map[norm] = existing[norm]
                reused += 1
                continue
            label = (entity.get("entity_type") or "ENTITY").strip().upper() or "ENTITY"
            by_label[label].append(entity)
        if reused:
            logger.info(f"batch_create_nodes: {reused} entities reused from existing map")

        for label, label_entities in by_label.items():
            for offset in range(0, len(label_entities), batch_size):
                batch = label_entities[offset : offset + batch_size]

                nodes_params = []
                for entity in batch:
                    source_ids = entity.get("source_id", [])
                    if not isinstance(source_ids, list):
                        source_ids = [source_ids] if source_ids else []
                    nodes_params.append({
                        "name": entity.get("entity_name", ""),
                        "description": entity.get("description", ""),
                        "source_ids": source_ids,
                        "entity_type": label,
                        "import_timestamp": timestamp,
                        "engine_name": engine_name or "",
                        "engine_type": engine_type or "",
                        "namespace": namespace,
                        "index_name": index_name or "",
                        "metadata_id": entity.get("metadata_id", "unknown"),
                    })

                query = f"""
                UNWIND $nodes AS nd
                CREATE (n:{label} {{
                    name: nd.name,
                    description: nd.description,
                    source_ids: nd.source_ids,
                    entity_type: nd.entity_type,
                    import_timestamp: nd.import_timestamp,
                    engine_name: nd.engine_name,
                    engine_type: nd.engine_type,
                    namespace: nd.namespace,
                    index_name: nd.index_name,
                    metadata_id: nd.metadata_id
                }})
                RETURN id(n) AS id, n.name AS name
                """

                try:
                    results = await self._execute_query(query, {"nodes": nodes_params}, graph_name=graph_name)
                    for record in results:
                        name = record.get("name", "")
                        if name:
                            entity_node_map[self._normalize_name(name)] = str(record.get("id", ""))
                    logger.info(f"Batch created {len(results)} {label} nodes")
                except Exception as e:
                    logger.warning(f"UNWIND node creation failed for label {label} (batch {offset}): {e} — falling back to individual creates")
                    for entity in batch:
                        try:
                            node = Node(
                                label=label,
                                properties={
                                    "name": entity.get("entity_name", ""),
                                    "description": entity.get("description", ""),
                                    "source_ids": entity.get("source_id", []),
                                    "entity_type": label,
                                    "import_timestamp": timestamp,
                                    "engine_name": engine_name,
                                    "engine_type": engine_type,
                                    "metadata_id": entity.get("metadata_id", "unknown"),
                                },
                            )
                            created = await self.create_node(
                                node,
                                namespace=namespace,
                                index_name=index_name,
                                engine_name=engine_name,
                                engine_type=engine_type,
                                metadata_id=entity.get("metadata_id", "unknown"),
                                graph_name=graph_name,
                            )
                            if created.id:
                                entity_node_map[self._normalize_name(entity.get("entity_name", ""))] = created.id
                        except Exception as ex:
                            logger.error(f"Individual node creation failed for {entity.get('entity_name')}: {ex}")

        return entity_node_map

    async def batch_create_relationships(
        self,
        relationships: List[Dict[str, Any]],
        entity_node_map: Dict[str, str],
        namespace: str,
        index_name: Optional[str] = None,
        engine_name: Optional[str] = None,
        engine_type: Optional[str] = None,
        graph_name: Optional[str] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Bulk-create relationships grouped by type using UNWIND.

        Relationships whose source or target entity is not found in
        entity_node_map are skipped (can happen with cross-window references).
        Falls back to individual creates if the UNWIND query fails.
        Returns the total count of created relationships.
        """
        from collections import defaultdict

        created_count = 0
        skipped = 0

        by_type: Dict[str, list] = defaultdict(list)
        for rel in relationships:
            rel_type = (rel.get("relationship_type") or "RELATED_TO").upper().strip() or "RELATED_TO"
            source_name = rel.get("src_id", "").strip()
            target_name = rel.get("tgt_id", "").strip()
            source_id = entity_node_map.get(self._normalize_name(source_name))
            target_id = entity_node_map.get(self._normalize_name(target_name))

            if not source_id or not target_id:
                skipped += 1
                continue

            source_ids = rel.get("source_id", [])
            if not isinstance(source_ids, list):
                source_ids = [source_ids] if source_ids else []

            by_type[rel_type].append({
                "source_id": int(source_id),
                "target_id": int(target_id),
                "weight": rel.get("weight", 1.0),
                "description": rel.get("description", ""),
                "source_ids": source_ids,
                "source_entity": source_name,
                "target_entity": target_name,
                "engine_name": engine_name or "",
                "engine_type": engine_type or "",
                "namespace": namespace,
                "metadata_id": rel.get("metadata_id") or "",
            })

        if skipped:
            logger.warning(f"Skipped {skipped} relationships: source/target not in entity_node_map")

        for rel_type, type_rels in by_type.items():
            for offset in range(0, len(type_rels), batch_size):
                batch = type_rels[offset : offset + batch_size]

                query = f"""
                UNWIND $rels AS rd
                MATCH (source), (target)
                WHERE id(source) = rd.source_id AND id(target) = rd.target_id
                CREATE (source)-[r:{rel_type} {{
                    weight: rd.weight,
                    description: rd.description,
                    source_ids: rd.source_ids,
                    source_entity: rd.source_entity,
                    target_entity: rd.target_entity,
                    engine_name: rd.engine_name,
                    engine_type: rd.engine_type,
                    namespace: rd.namespace,
                    metadata_id: rd.metadata_id
                }}]->(target)
                RETURN count(r) AS created
                """

                try:
                    results = await self._execute_query(query, {"rels": batch}, graph_name=graph_name)
                    batch_created = results[0].get("created", 0) if results else 0
                    created_count += batch_created
                    logger.info(f"Batch created {batch_created} {rel_type} relationships")
                except Exception as e:
                    logger.warning(f"UNWIND relationship creation failed for type {rel_type} (batch {offset}): {e} — falling back to individual creates")
                    for rd in batch:
                        try:
                            rel_obj = Relationship(
                                source_id=str(rd["source_id"]),
                                target_id=str(rd["target_id"]),
                                type=rel_type,
                                properties={
                                    "weight": rd["weight"],
                                    "description": rd["description"],
                                    "source_ids": rd["source_ids"],
                                    "source_entity": rd["source_entity"],
                                    "target_entity": rd["target_entity"],
                                    "engine_name": rd["engine_name"],
                                    "engine_type": rd["engine_type"],
                                },
                            )
                            await self.create_relationship(
                                rel_obj,
                                namespace=namespace,
                                index_name=index_name,
                                engine_name=engine_name,
                                engine_type=engine_type,
                                graph_name=graph_name,
                            )
                            created_count += 1
                        except Exception as ex:
                            logger.error(f"Individual relationship creation failed: {ex}")

        return created_count

    async def load_entity_name_map(
        self,
        namespace: Optional[str] = None,
        graph_name: Optional[str] = None,
        query_timeout: int = 300000,
    ) -> Dict[str, str]:
        """
        Load all existing entity nodes as normalized_name → node_id.
        Used to pre-populate entity_node_map before extraction so that
        subsequent runs do not create duplicate nodes.
        Paginates in batches of 5000 to bypass FalkorDB's resultset_size cap.
        """
        PAGE_SIZE = 5000
        where_parts = ["NOT 'CommunityReport' IN labels(n)"]
        params: Dict[str, Any] = {}
        if namespace is not None:
            where_parts.append("n.namespace = $namespace")
            params["namespace"] = namespace

        where_clause = "WHERE " + " AND ".join(where_parts)
        query_tpl = f"""
        MATCH (n)
        {where_clause}
        RETURN id(n) AS id, n.name AS name
        ORDER BY id(n)
        SKIP $skip LIMIT $page_size
        """

        result: Dict[str, str] = {}
        skip = 0
        while True:
            page_params = {**params, "skip": skip, "page_size": PAGE_SIZE}
            page = await self._execute_query(
                query_tpl, page_params, namespace=namespace, graph_name=graph_name, timeout=query_timeout
            )
            for record in page:
                name = record.get("name") or ""
                node_id = str(record.get("id", ""))
                if name and node_id:
                    result[self._normalize_name(name)] = node_id
            if len(page) < PAGE_SIZE:
                break
            skip += PAGE_SIZE

        logger.info(f"load_entity_name_map: {len(result)} existing entities loaded (graph='{graph_name or namespace}')")
        return result

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

        total_deleted = 0
        while True:
            result = await self._execute_query(query, parameters, namespace=namespace, graph_name=graph_name)
            batch = result[0].get("deleted_count", 0) if result else 0
            total_deleted += batch
            if batch == 0:
                break
            logger.info(f"Deleted batch of {batch} nodes (total so far: {total_deleted})")

        logger.info(f"Deleted {total_deleted} nodes matching metadata")

        return {"nodes_deleted": total_deleted}

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
                                             graph_name: Optional[str] = None,
                                             query_timeout: int = 300000) -> Dict[str, Any]:
        """
        Fetch the entire graph with optional filters (async).

        Uses two separate queries (nodes then relationships) instead of a single
        OPTIONAL MATCH to avoid the cartesian-explosion that caused timeouts on
        large graphs (>30k nodes).  query_timeout defaults to 5 minutes (300 000 ms).
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

        # FalkorDB enforces a server-side resultset_size cap (default 10 000).
        # Paginate with SKIP/LIMIT so we fetch the full graph regardless of that cap.
        PAGE_SIZE = 5000

        # --- Query 1: entity nodes only (skip CommunityReport), paginated ---
        if where_clause:
            node_query_tpl = f"""
        MATCH (n)
        {where_clause}
        AND NOT 'CommunityReport' IN labels(n)
        RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties
        ORDER BY id(n)
        SKIP $skip LIMIT $page_size
        """
        else:
            node_query_tpl = """
        MATCH (n)
        WHERE NOT 'CommunityReport' IN labels(n)
        RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties
        ORDER BY id(n)
        SKIP $skip LIMIT $page_size
        """

        nodes = {}
        skip = 0
        while True:
            page_params = {**params, "skip": skip, "page_size": PAGE_SIZE}
            page = await self._execute_query(
                node_query_tpl, page_params, namespace=namespace, graph_name=graph_name, timeout=query_timeout
            )
            for record in page:
                n_id = str(record.get("id", ""))
                n_labels = record.get("labels") or ["Unknown"]
                if n_id:
                    nodes[n_id] = {
                        "id": n_id,
                        "label": n_labels[0] if n_labels else "Unknown",
                        "properties": record.get("properties") or {}
                    }
            if len(page) < PAGE_SIZE:
                break
            skip += PAGE_SIZE

        # --- Query 2: relationships between entity nodes only, paginated ---
        rel_where_parts = []
        rel_params = {}
        if namespace is not None:
            rel_where_parts.append("n.namespace = $namespace")
            rel_params["namespace"] = namespace
        if index_name is not None:
            rel_where_parts.append("n.index_name = $index_name")
            rel_params["index_name"] = index_name

        rel_where = ("WHERE " + " AND ".join(rel_where_parts)) if rel_where_parts else ""

        if rel_where:
            rel_query_tpl = f"""
        MATCH (n)-[r]->(m)
        {rel_where}
        AND NOT 'CommunityReport' IN labels(n)
        AND NOT 'CommunityReport' IN labels(m)
        RETURN id(r) AS id, type(r) AS type, properties(r) AS properties,
               id(n) AS source_id, id(m) AS target_id
        ORDER BY id(r)
        SKIP $skip LIMIT $page_size
        """
        else:
            rel_query_tpl = """
        MATCH (n)-[r]->(m)
        WHERE NOT 'CommunityReport' IN labels(n)
        AND NOT 'CommunityReport' IN labels(m)
        RETURN id(r) AS id, type(r) AS type, properties(r) AS properties,
               id(n) AS source_id, id(m) AS target_id
        ORDER BY id(r)
        SKIP $skip LIMIT $page_size
        """

        relationships = []
        seen_rels: set = set()
        skip = 0
        while True:
            page_params = {**rel_params, "skip": skip, "page_size": PAGE_SIZE}
            page = await self._execute_query(
                rel_query_tpl, page_params, namespace=namespace, graph_name=graph_name, timeout=query_timeout
            )
            for record in page:
                r_id = str(record.get("id", ""))
                if r_id and r_id not in seen_rels:
                    seen_rels.add(r_id)
                    relationships.append({
                        "id": r_id,
                        "type": record.get("type", ""),
                        "properties": record.get("properties") or {},
                        "source_id": str(record.get("source_id", "")),
                        "target_id": str(record.get("target_id", "")),
                    })
            if len(page) < PAGE_SIZE:
                break
            skip += PAGE_SIZE

        logger.info(
            f"get_all_nodes_and_relationships: {len(nodes)} nodes, {len(relationships)} relationships "
            f"(graph='{graph_name or namespace}')"
        )
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
