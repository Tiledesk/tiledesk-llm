"""
Neo4j Repository for Knowledge Graph operations.
Provides CRUD operations for nodes and relationships with connection pooling.
"""

import json
import logging
from tilellm.shared.utility import get_service_config

try:
    import neo4j
    from neo4j import GraphDatabase, Driver, Session, Query
    from neo4j.time import DateTime, Date, Time, Duration
except ImportError:
    neo4j = None
    GraphDatabase = None
    Driver = None
    Session = None
    Query = None
    DateTime = None
    Date = None
    Time = None
    Duration = None
from typing import Optional, List, Dict, Any, cast
#from neo4j import GraphDatabase, Driver, Session, Query
from ..models import Node, Relationship

logger = logging.getLogger(__name__)


class GraphRepository:
    """
    Repository for interacting with Neo4j graph database.
    Uses connection pooling for optimal performance.
    """

    _driver: Optional[Driver] = None
    SEARCHABLE_PROPERTIES = ['text', 'content', 'name', 'title', 'description', 'summary', 'label', 'type']
    SEARCHABLE_REL_PROPERTIES = ['type', 'weight', 'context', 'description', 'source', 'target']

    def __init__(self):
        """
        Initialize Neo4j connection using environment variables.
        """
        if GraphRepository._driver is None:
            if neo4j is None:
                raise ImportError("Neo4j drivers not installed. Please install with 'poetry install -E graph'")
            
            service_config = get_service_config()
            neo4j_config = service_config.get("neo4j")

            if not neo4j_config:
                raise ValueError("Neo4j configuration not found in environment variables")

            uri = neo4j_config.get("uri")
            user = neo4j_config.get("user")
            password = neo4j_config.get("password")
            max_connection_pool_size = neo4j_config.get("max_connection_pool_size", 50) # Default to 50 if not specified

            if not all([uri, user, password]):
                raise ValueError("uri, user, and password are required for Neo4j configuration")

            GraphRepository._driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=max_connection_pool_size,
                connection_acquisition_timeout=60.0,  # 60 seconds timeout
                max_connection_lifetime=3600,  # 1 hour max connection lifetime
            )
            # Suppress Neo4j notification logs to avoid spam about existing indexes
            logging.getLogger('neo4j.notifications').setLevel(logging.WARNING)
            logger.info(f"Neo4j driver initialized with connection pool size: {max_connection_pool_size}")
            # Ensure indexes are created
            self.ensure_indexes()

    def _convert_neo4j_value(self, value) -> Any:
        """
        Convert Neo4j-specific types (DateTime, Date, etc.) to Python serializable types.
        """
        if value is None:
            return None
        
        # Check for Neo4j temporal types using isinstance (if types are available)
        # Only check if DateTime is not None (import succeeded)
        if DateTime is not None:
            # Create tuple of only non-None types (should be all if import succeeded)
            neo4j_types = tuple(t for t in (DateTime, Date, Time, Duration) if t is not None)
            if neo4j_types and isinstance(value, neo4j_types):
                try:
                    # Try iso_format first (most Neo4j temporal types have this)
                    if hasattr(value, 'iso_format'):
                        return value.iso_format()
                    # Try to_native as fallback
                    if hasattr(value, 'to_native'):
                        return value.to_native()
                    # Last resort: string representation
                    return str(value)
                except Exception:
                    # If all conversion methods fail, return string representation
                    return str(value)
        
        # Detect any Neo4j-specific type by module name
        if hasattr(value, '__class__') and hasattr(value.__class__, '__module__'):
            module = value.__class__.__module__
            if module.startswith('neo4j.'):
                # Generic Neo4j type conversion
                try:
                    if hasattr(value, 'iso_format'):
                        return value.iso_format()
                    if hasattr(value, 'to_native'):
                        return value.to_native()
                    if hasattr(value, 'isoformat'):
                        return value.isoformat()
                    if hasattr(value, 'strftime'):
                        # Try common datetime format
                        import datetime
                        if isinstance(value, datetime.datetime):
                            return value.isoformat()
                except Exception:
                    pass
                # Fallback to string representation
                return str(value)
        
        # Legacy detection for older drivers or other Neo4j types
        if hasattr(value, 'iso_format'):
            # Neo4j DateTime, Date, Time, Duration
            try:
                return value.iso_format()
            except (AttributeError, Exception):
                pass
        if hasattr(value, 'to_native'):
            # Convert to native Python type
            try:
                return value.to_native()
            except Exception:
                pass
        
        # Recursively handle containers
        if isinstance(value, dict):
            return {k: self._convert_neo4j_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._convert_neo4j_value(v) for v in value]
        
        # Return unchanged for other types
        return value

    def _convert_properties(self, props_dict) -> Dict[str, Any]:
        """
        Convert Neo4j properties dict to Python serializable dict.
        """
        if not props_dict:
            return {}
        if isinstance(props_dict, dict):
            converted = self._convert_neo4j_value(props_dict)
            # Ensure we return a dict (converted should be dict if props_dict is dict)
            if isinstance(converted, dict):
                return cast(Dict[str, Any], converted)
            # Should not happen, but fallback
            return {}
        # If props_dict is not a dict (should not happen), return empty dict
        return {}

    def ensure_indexes(self):
        """
        Ensure that the required full-text search indexes exist in the database.
        This method is idempotent and safe to run on startup.
        """
        with self._get_session() as session:
            # Create node index with fallback logic
            logger.info("Ensuring 'node_text_index' exists...")
            node_index_exists = False
            use_fallback = False
            
            # Try to check if index exists using SHOW INDEXES (supported in Neo4j 4.0+)
            try:
                query = Query("SHOW INDEXES WHERE name = 'node_text_index' AND type = 'FULLTEXT'")
                result = session.run(query)
                node_index_exists = result.single() is not None
                logger.debug(f"SHOW INDEXES check result: node_text_index exists = {node_index_exists}")
            except neo4j.exceptions.ClientError as e:
                # SHOW INDEXES not available, fall back to CREATE IF NOT EXISTS
                logger.warning(f"SHOW INDEXES not supported: {e}. Using CREATE IF NOT EXISTS fallback.")
                use_fallback = True
            
            if use_fallback:
                # Fallback: use CREATE IF NOT EXISTS (may produce notifications but they're suppressed)
                try:
                    query = Query("CREATE FULLTEXT INDEX node_text_index IF NOT EXISTS FOR (n:Document|Entity|CommunityReport|Person|Organization) ON EACH [n.text, n.content, n.name, n.title, n.description, n.summary, n.report]")
                    session.run(query)
                    logger.info("Successfully created or verified 'node_text_index' (fallback method).")
                except neo4j.exceptions.ClientError as e:
                    if e.code == 'Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists':
                        logger.warning("An equivalent node index to 'node_text_index' already exists. Skipping creation.")
                    else:
                        logger.error(f"Error creating node index: {e}", exc_info=True)
                        raise
            else:
                # SHOW INDEXES worked, create only if doesn't exist
                if not node_index_exists:
                    try:
                        query = Query("CREATE FULLTEXT INDEX node_text_index FOR (n:Document|Entity|CommunityReport|Person|Organization) ON EACH [n.text, n.content, n.name, n.title, n.description, n.summary, n.report]")
                        session.run(query)
                        logger.info("Successfully created 'node_text_index'.")
                    except neo4j.exceptions.ClientError as e:
                        if e.code == 'Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists':
                            logger.warning("An equivalent node index to 'node_text_index' already exists. Skipping creation.")
                        else:
                            logger.error(f"Error creating node index: {e}", exc_info=True)
                            raise
                else:
                    logger.info("'node_text_index' already exists. Skipping creation.")
            
            # Create relationship index with same fallback logic
            logger.info("Ensuring 'relationship_text_index' exists...")
            rel_index_exists = False
            use_fallback = False
            
            try:
                query = Query("SHOW INDEXES WHERE name = 'relationship_text_index' AND type = 'FULLTEXT'")
                result = session.run(query)
                rel_index_exists = result.single() is not None
                logger.debug(f"SHOW INDEXES check result: relationship_text_index exists = {rel_index_exists}")
            except neo4j.exceptions.ClientError as e:
                logger.warning(f"SHOW INDEXES not supported: {e}. Using CREATE IF NOT EXISTS fallback.")
                use_fallback = True
            
            if use_fallback:
                try:
                    query = Query("CREATE FULLTEXT INDEX relationship_text_index IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON EACH [r.description, r.context]")
                    session.run(query)
                    logger.info("Successfully created or verified 'relationship_text_index' (fallback method).")
                except neo4j.exceptions.ClientError as e:
                    if e.code == 'Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists':
                        logger.warning("An equivalent relationship index to 'relationship_text_index' already exists. Skipping creation.")
                    else:
                        logger.error(f"Error creating relationship index: {e}", exc_info=True)
                        raise
            else:
                if not rel_index_exists:
                    try:
                        query = Query("CREATE FULLTEXT INDEX relationship_text_index FOR ()-[r:RELATED_TO]-() ON EACH [r.description, r.context]")
                        session.run(query)
                        logger.info("Successfully created 'relationship_text_index'.")
                    except neo4j.exceptions.ClientError as e:
                        if e.code == 'Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists':
                            logger.warning("An equivalent relationship index to 'relationship_text_index' already exists. Skipping creation.")
                        else:
                            logger.error(f"Error creating relationship index: {e}", exc_info=True)
                            raise
                else:
                    logger.info("'relationship_text_index' already exists. Skipping creation.")

    @classmethod
    def close(cls):
        """Close the driver connection pool"""
        if cls._driver:
            cls._driver.close()
            cls._driver = None
            logger.info("Neo4j driver closed")

    def _get_session(self) -> Session:
        """Get a session from the connection pool"""
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self._driver.session()

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Returns:
            List of result records as dictionaries
        """
        with self._get_session() as session:
            result = session.run(query, **(parameters or {}))
            # Convert result to list of dicts with Neo4j type conversion
            records = []
            for record in result:
                record_dict = {}
                for key, value in record.items():
                    record_dict[key] = self._convert_neo4j_value(value)
                records.append(record_dict)
            return records

    # ==================== NODE OPERATIONS ====================

    def create_node(self, node: Node, namespace: Optional[str] = None, index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None, metadata_id: Optional[str] = None) -> Node:
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
        # Add partition properties if provided
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
        
        with self._get_session() as session:
            query = f"""
            CREATE (n:{node.label} $properties)
            RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
            """
            result = session.run(query, properties=properties)  # type: ignore
            record = result.single()

            if record:
                return Node(
                    id=record["id"],
                    label=record["labels"][0] if record["labels"] else node.label,
                    properties=self._convert_properties(dict(record["properties"]))  # type: ignore
                )
            raise RuntimeError("Failed to create node")

    def find_node_by_id(self, node_id: str) -> Optional[Node]:
        """
        Find a node by its internal Neo4j ID.

        Args:
            node_id: Internal Neo4j node ID

        Returns:
            Node if found, None otherwise
        """
        with self._get_session() as session:
            query = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
            """
            result = session.run(query, node_id=node_id)  # type: ignore
            record = result.single()

            if record:
                return Node(
                    id=str(record["id"]),
                    label=record["labels"][0] if record["labels"] else "Unknown",
                    properties=self._convert_properties(dict(record["properties"]))  # type: ignore
                )
            return None

    def find_nodes_by_label(self, label: str, limit: int = 100, namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
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
        with self._get_session() as session:
            # Build WHERE clause
            where_clauses = []
            params: Dict[str, Any] = {"limit": limit}
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
            RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
            LIMIT $limit
            """
            result = session.run(query, **params)  # type: ignore

            nodes = []
            for record in result:
                nodes.append(Node(
                    id=str(record["id"]),
                    label=record["labels"][0] if record["labels"] else label,
                    properties=dict(record["properties"])
                ))
            return nodes

    def find_nodes_by_property(self, label: str, property_key: str, property_value: Any, limit: int = 100, namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
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
        with self._get_session() as session:
            # Build WHERE clause
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
            RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
            LIMIT $limit
            """
            result = session.run(query, **params)  # type: ignore

            nodes = []
            for record in result:
                nodes.append(Node(
                    id=str(record["id"]),
                    label=record["labels"][0] if record["labels"] else label,
                    properties=dict(record["properties"])
                ))
            return nodes

    def update_node(self, node_id: str, label: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> Optional[Node]:
        """
        Update a node's label and/or properties.

        Args:
            node_id: Internal Neo4j node ID
            label: New label (optional)
            properties: New properties to set/update (optional)

        Returns:
            Updated node if found, None otherwise
        """
        with self._get_session() as session:
            # First check if node exists
            existing_node = self.find_node_by_id(node_id)
            if not existing_node:
                return None

            # Update properties if provided
            if properties:
                query = """
                MATCH (n)
                WHERE elementId(n) = $node_id
                SET n += $properties
                RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
                """
                result = session.run(query, node_id=node_id, properties=properties)  # type: ignore
                logger.debug(f"Properties update result: {result}")

            # Update label if provided (requires removing old label and adding new one)
            if label and label != existing_node.label:
                query = f"""
                MATCH (n)
                WHERE elementId(n) = $node_id
                REMOVE n:{existing_node.label}
                SET n:{label}
                RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
                """
                result = session.run(query, node_id=node_id)  # type: ignore
                logger.debug(f"Label update result: {result}")

            # Fetch updated node
            return self.find_node_by_id(node_id)

    def delete_node(self, node_id: str, detach: bool = True) -> bool:
        """
        Delete a node from the graph.

        Args:
            node_id: Internal Neo4j node ID
            detach: If True, delete all relationships before deleting node

        Returns:
            True if deleted, False if not found
        """
        with self._get_session() as session:
            if detach:
                query = """
                MATCH (n)
                WHERE elementId(n) = $node_id
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
            else:
                query = """
                MATCH (n)
                WHERE elementId(n) = $node_id
                DELETE n
                RETURN count(n) as deleted_count
                """

            result = session.run(query, node_id=node_id)  # type: ignore
            record = result.single()
            return record["deleted_count"] > 0 if record else False

    # ==================== RELATIONSHIP OPERATIONS ====================

    def create_relationship(self, relationship: Relationship, namespace: Optional[str] = None, index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None, metadata_id: Optional[str] = None) -> Relationship:
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
        # Add partition properties if provided
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
        
        with self._get_session() as session:
            query = f"""
            MATCH (source), (target)
            WHERE elementId(source) = $source_id AND elementId(target) = $target_id
            CREATE (source)-[r:{relationship.type} $properties]->(target)
            RETURN elementId(r) as id, type(r) as type, properties(r) as properties,
                   elementId(source) as source_id, elementId(target) as target_id
            """
            result = session.run(
                query,  # type: ignore
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                properties=properties
            )
            record = result.single()

            if record:
                return Relationship(
                    id=str(record["id"]),
                    source_id=str(record["source_id"]),
                    target_id=str(record["target_id"]),
                    type=record["type"],
                    properties=self._convert_properties(dict(record["properties"]))  # type: ignore
                )
            raise RuntimeError("Failed to create relationship. Check if source and target nodes exist.")

    def find_relationship_by_id(self, relationship_id: str) -> Optional[Relationship]:
        """
        Find a relationship by its internal Neo4j ID.

        Args:
            relationship_id: Internal Neo4j relationship ID

        Returns:
            Relationship if found, None otherwise
        """
        with self._get_session() as session:
            query = """
            MATCH (source)-[r]->(target)
            WHERE elementId(r) = $relationship_id
            RETURN elementId(r) as id, type(r) as type, properties(r) as properties,
                   elementId(source) as source_id, elementId(target) as target_id
            """
            result = session.run(query, relationship_id=relationship_id)  # type: ignore
            record = result.single()

            if record:
                return Relationship(
                    id=str(record["id"]),
                    source_id=str(record["source_id"]),
                    target_id=str(record["target_id"]),
                    type=record["type"],
                    properties=self._convert_properties(dict(record["properties"]))  # type: ignore
                )
            return None

    def find_relationships_by_node(self, node_id: str, direction: str = "both") -> List[Relationship]:
        """
        Find all relationships connected to a node.

        Args:
            node_id: Internal Neo4j node ID
            direction: "incoming", "outgoing", or "both"

        Returns:
            List of relationships
        """
        with self._get_session() as session:
            if direction == "outgoing":
                query = """
                MATCH (source)-[r]->(target)
                WHERE elementId(source) = $node_id
                RETURN elementId(r) as id, type(r) as type, properties(r) as properties,
                       elementId(source) as source_id, elementId(target) as target_id
                """
            elif direction == "incoming":
                query = """
                MATCH (source)-[r]->(target)
                WHERE elementId(target) = $node_id
                RETURN elementId(r) as id, type(r) as type, properties(r) as properties,
                       elementId(source) as source_id, elementId(target) as target_id
                """
            else:  # both
                query = """
                MATCH (n)-[r]-(other)
                WHERE elementId(n) = $node_id
                RETURN elementId(r) as id, type(r) as type, properties(r) as properties,
                       elementId(startNode(r)) as source_id, elementId(endNode(r)) as target_id
                """

            result = session.run(query, node_id=node_id)  # type: ignore

            relationships = []
            for record in result:
                relationships.append(Relationship(
                    id=str(record["id"]),
                    source_id=str(record["source_id"]),
                    target_id=str(record["target_id"]),
                    type=record["type"],
                    properties=self._convert_properties(dict(record["properties"]))  # type: ignore
                ))
            return relationships

    def update_relationship(self, relationship_id: str, rel_type: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> Optional[Relationship]:
        """
        Update a relationship's type and/or properties.
        Note: Neo4j doesn't support changing relationship type directly.
        To change type, you need to delete and recreate.

        Args:
            relationship_id: Internal Neo4j relationship ID
            rel_type: New relationship type (will recreate relationship)
            properties: New properties to set/update

        Returns:
            Updated relationship if found, None otherwise
        """
        with self._get_session() as session:
            # Check if relationship exists
            existing_rel = self.find_relationship_by_id(relationship_id)
            if not existing_rel:
                return None

            # If type needs to change, we need to recreate the relationship
            if rel_type and rel_type != existing_rel.type:
                # Delete old and create new
                self.delete_relationship(relationship_id)
                new_rel = Relationship(
                    source_id=existing_rel.source_id,
                    target_id=existing_rel.target_id,
                    type=rel_type,
                    properties=properties if properties else existing_rel.properties
                )
                return self.create_relationship(new_rel)

            # Otherwise just update properties
            if properties:
                query = """
                MATCH ()-[r]->()
                WHERE elementId(r) = $relationship_id
                SET r += $properties
                RETURN elementId(r) as id, type(r) as type, properties(r) as properties,
                       elementId(startNode(r)) as source_id, elementId(endNode(r)) as target_id
                """
                result = session.run(query, relationship_id=relationship_id, properties=properties)  # type: ignore
                logger.debug(f"Query result {result}")

            return self.find_relationship_by_id(relationship_id)

    def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a relationship from the graph.

        Args:
            relationship_id: Internal Neo4j relationship ID

        Returns:
            True if deleted, False if not found
        """
        with self._get_session() as session:
            query = """
            MATCH ()-[r]->()
            WHERE elementId(r) = $relationship_id
            DELETE r
            RETURN count(r) as deleted_count
            """
            result = session.run(query, relationship_id=relationship_id)  # type: ignore
            record = result.single()
            return record["deleted_count"] > 0 if record else False

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
        with self._get_session() as session:
            # Build WHERE clause dynamically based on provided parameters
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
                where_clauses.append("(exists(n.metadata_id) AND n.metadata_id = $metadata_id)")
                parameters["metadata_id"] = metadata_id
            
            # If no parameters provided, don't delete everything (safety)
            if not where_clauses:
                logger.warning("No metadata parameters provided, skipping delete to avoid deleting all nodes")
                return {"nodes_deleted": 0, "reports_deleted": 0}
            
            where_clause = " AND ".join(where_clauses)
            
            # Delete nodes and their relationships
            query = f"""
            MATCH (n)
            WHERE {where_clause}
            WITH n LIMIT 10000
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            
            result = session.run(query, parameters)
            record = result.single()
            deleted_count = record["deleted_count"] if record else 0
            
            logger.info(f"Deleted {deleted_count} nodes matching metadata: namespace={namespace}, "
                       f"index_name={index_name}, engine_name={engine_name}, "
                       f"engine_type={engine_type}, metadata_id={metadata_id}")
            
            return {"nodes_deleted": deleted_count}

    def delete_nodes_by_metadata_id(
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
        with self._get_session() as session:
            # Build WHERE clause dynamically based on provided parameters
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
                where_clauses.append("(exists(n.metadata_id) AND n.metadata_id = $metadata_id)")
                parameters["metadata_id"] = metadata_id

            # If no parameters provided, don't delete everything (safety)
            if not where_clauses:
                logger.warning("No metadata parameters provided, skipping delete to avoid deleting all nodes")
                return {"nodes_deleted": 0, "reports_deleted": 0}

            where_clause = " AND ".join(where_clauses)

            # Delete nodes and their relationships
            query = f"""
            MATCH (n)
            WHERE {where_clause}
            WITH n LIMIT 10000
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """

            result = session.run(query, parameters)
            record = result.single()
            deleted_count = record["deleted_count"] if record else 0

            logger.info(f"Deleted {deleted_count} nodes matching metadata: namespace={namespace}, "
                        f"index_name={index_name}, engine_name={engine_name}, "
                        f"engine_type={engine_type}, metadata_id={metadata_id}")

            return {"nodes_deleted": deleted_count}

    # ==================== UTILITY OPERATIONS ====================

    def verify_connection(self) -> bool:
        """
        Verify the connection to Neo4j is working.

        Returns:
            True if connection is successful
        """
        try:
            with self._get_session() as session:
                result = session.run(Query("RETURN 1 as test"))
                record = result.single()
                return record is not None and record["test"] == 1
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database.

        Returns:
            Dictionary with database statistics
        """
        with self._get_session() as session:
            query = """
            MATCH (n)
            OPTIONAL MATCH ()-[r]->()
            RETURN count(DISTINCT n) as node_count, count(r) as relationship_count
            """
            result = session.run(query)  # type: ignore
            record = result.single()

            if not record:
                return {
                    "node_count": 0,
                    "relationship_count": 0,
                    "connection_pool_size": 0
                }

            return {
                "node_count": record["node_count"],
                "relationship_count": record["relationship_count"],
                "connection_pool_size": self.max_connection_pool_size
            }
    
    def search_nodes_by_text(self, search_text: str, limit: int = 10, namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
        """
        Search nodes using a full-text index.
        Assumes a full-text index named 'node_text_index' exists.
        
        Args:
            search_text: Text to search for. Can include Lucene syntax.
            limit: Maximum number of nodes to return.
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes
            
        Returns:
            List of matching nodes.
        """
        with self._get_session() as session:
            # Build WHERE clause for partition filtering
            where_clauses = []
            params: Dict[str, Any] = {"search_text": search_text, "limit": limit}
            if namespace is not None:
                where_clauses.append("node.namespace = $namespace")
                params["namespace"] = namespace
            if index_name is not None:
                where_clauses.append("node.index_name = $index_name")
                params["index_name"] = index_name
            
            where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            query = f"""
            CALL db.index.fulltext.queryNodes("node_text_index", $search_text, {{limit: $limit}})
            YIELD node
            {where_clause}
            RETURN elementId(node) as id, labels(node) as labels, properties(node) as properties
            """
            result = session.run(query, **params)  # type: ignore
            
            nodes = []
            for record in result:
                nodes.append(Node(
                    id=str(record["id"]),
                    label=record["labels"][0] if record["labels"] else "Unknown",
                    properties=self._convert_properties(dict(record["properties"]))  # type: ignore
                ))
            return nodes
    
    def search_relationships_by_text(self, search_text: str, limit: int = 10, namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search relationships using a full-text index.
        Assumes a full-text index named 'relationship_text_index' exists.

        Args:
            search_text: Text to search for. Can include Lucene syntax.
            limit: Maximum number of relationships to return.
            namespace: Optional namespace to filter relationships
            index_name: Optional index_name (collection name) to filter relationships
            
        Returns:
            List of matching relationships with node info.
        """
        with self._get_session() as session:
            # Build WHERE clause for partition filtering
            where_clauses = []
            params: Dict[str, Any] = {"search_text": search_text, "limit": limit}
            if namespace is not None:
                where_clauses.append("relationship.namespace = $namespace")
                params["namespace"] = namespace
            if index_name is not None:
                where_clauses.append("relationship.index_name = $index_name")
                params["index_name"] = index_name
            
            where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            query = f"""
            CALL db.index.fulltext.queryRelationships("relationship_text_index", $search_text, {{limit: $limit}})
            YIELD relationship
            {where_clause}
            MATCH (source)-[relationship]->(target)
            RETURN elementId(relationship) as id, type(relationship) as type, properties(relationship) as properties,
                   elementId(source) as source_id, elementId(target) as target_id,
                   labels(source) as source_labels, labels(target) as target_labels,
                   properties(source) as source_props, properties(target) as target_props
            """
            result = session.run(query, **params)
            
            relationships = []
            for record in result:
                relationships.append({
                    "id": str(record["id"]),
                    "type": record["type"],
                    "properties": self._convert_properties(record["properties"]),
                    "source": {
                        "id": str(record["source_id"]),
                        "label": record["source_labels"][0] if record["source_labels"] else "Unknown",
                        "properties": self._convert_properties(record["source_props"])
                    },
                    "target": {
                        "id": str(record["target_id"]),
                        "label": record["target_labels"][0] if record["target_labels"] else "Unknown",
                        "properties": self._convert_properties(record["target_props"])
                    }
                })
            return relationships

    def search_community_reports(self, search_text: str, limit: int = 5) -> List[Node]:
        """
        Search community reports by text.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of reports to return
            
        Returns:
            List of matching community report nodes
        """
        with self._get_session() as session:
            # Assumes CommunityReport nodes with 'full_report', 'summary', or 'title' properties
            query = """
            MATCH (n:CommunityReport)
            WHERE 
                (n.full_report IS NOT NULL AND toLower(n.full_report) CONTAINS toLower($search_text)) OR
                (n.summary IS NOT NULL AND toLower(n.summary) CONTAINS toLower($search_text)) OR
                (n.title IS NOT NULL AND toLower(n.title) CONTAINS toLower($search_text))
            RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
            LIMIT $limit
            """
            result = session.run(query, search_text=search_text, limit=limit)
            
            nodes = []
            for record in result:
                nodes.append(Node(
                    id=str(record["id"]),
                    label=record["labels"][0] if record["labels"] else "CommunityReport",
                    properties=self._convert_properties(dict(record["properties"]))  # type: ignore
                ))
            return nodes

    def get_all_nodes_and_relationships(self, namespace: Optional[str] = None, index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch the entire graph from Neo4j, optionally filtered by namespace, index_name, engine_name and engine_type.
        
        Args:
            namespace: Optional namespace to filter nodes
            index_name: Optional index_name (collection name) to filter nodes
            engine_name: Optional engine name (e.g., 'pinecone', 'qdrant') to filter nodes
            engine_type: Optional engine type (e.g., 'pod', 'serverless') to filter nodes
            
        Returns:
            Dictionary with filtered nodes and relationships
        """
        logger.info(f"get_all_nodes_and_relationships -> query with namespace: {namespace}, index_name: {index_name}, engine_name: {engine_name}, engine_type: {engine_type}")
        with self._get_session() as session:
            # Build WHERE clause based on provided filters
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
            result = session.run(query, **params)
            
            nodes = {}
            relationships = []
            seen_nodes = set()
            seen_rels = set()
            
            for record in result:
                # Process source node
                n = record.get("n")
                if n:
                    n_id = n.element_id
                    if n_id not in seen_nodes:
                        seen_nodes.add(n_id)
                        nodes[n_id] = {
                            "id": n_id,
                            "label": list(n.labels)[0] if n.labels else "Unknown",
                            "properties": self._convert_properties(dict(n.items()))
                        }
                
                # Process target node
                m = record.get("m")
                if m:
                    m_id = m.element_id
                    if m_id not in seen_nodes:
                        seen_nodes.add(m_id)
                        nodes[m_id] = {
                            "id": m_id,
                            "label": list(m.labels)[0] if m.labels else "Unknown",
                            "properties": self._convert_properties(dict(m.items()))
                        }
                
                # Process relationship
                r = record.get("r")
                if r:
                    r_id = r.element_id
                    if r_id not in seen_rels:
                        seen_rels.add(r_id)
                        relationships.append({
                            "id": r_id,
                            "type": r.type,
                            "properties": self._convert_properties(dict(r.items())),
                            "source_id": r.start_node.element_id,
                            "target_id": r.end_node.element_id
                        })
            
            return {
                "nodes": list(nodes.values()),
                "relationships": relationships
            }

    def get_community_network(self, namespace: str, index_name: str, limit: int = 1000) -> Dict[str, Any]:
        """
        Fetch the community graph (nodes + BELONGS_TO_COMMUNITY relationships).
        
        Args:
            namespace: Namespace to filter
            index_name: index_name to filter
            limit: Maximum number of paths to return
            
        Returns:
            Dictionary with nodes and relationships
        """
        logger.info(f"get_community_network -> namespace: {namespace}, index_name: {index_name}, limit: {limit}")
        with self._get_session() as session:
            query = """
            MATCH p=(n)-[r:BELONGS_TO_COMMUNITY]->(m)
            WHERE n.namespace = $namespace AND m.namespace = $namespace 
              AND n.index_name = $index_name AND m.index_name = $index_name
            RETURN p
            LIMIT $limit
            """
            result = session.run(query, namespace=namespace, index_name=index_name, limit=limit)
            
            nodes = {}
            relationships = []
            seen_nodes = set()
            seen_rels = set()
            
            for record in result:
                path = record.get("p")
                if path:
                    # Process nodes in path
                    for node in path.nodes:
                        n_id = node.element_id
                        if n_id not in seen_nodes:
                            seen_nodes.add(n_id)
                            nodes[n_id] = {
                                "id": n_id,
                                "label": list(node.labels)[0] if node.labels else "Unknown",
                                "properties": self._convert_properties(dict(node.items()))
                            }
                    
                    # Process relationships in path
                    for rel in path.relationships:
                        r_id = rel.element_id
                        if r_id not in seen_rels:
                            seen_rels.add(r_id)
                            relationships.append({
                                "id": r_id,
                                "type": rel.type,
                                "properties": self._convert_properties(dict(rel.items())),
                                "source_id": rel.start_node.element_id,
                                "target_id": rel.end_node.element_id
                            })
            
            return {
                "nodes": list(nodes.values()),
                "relationships": relationships
            }

    def save_community_report(self, community_id: str, report: Dict[str, Any], level: int, namespace: Optional[str] = None, index_name: Optional[str] = None, engine_name: Optional[str] = None, engine_type: Optional[str] = None, metadata_id: Optional[str] = None) -> Optional[str]:
        """
        Save a community report as a node in Neo4j.
        
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
        with self._get_session() as session:
            # Create CommunityReport node
            query = """
            MERGE (c:CommunityReport {community_id: $community_id, level: $level})
            SET c += $properties,
                c.import_timestamp = datetime()
            RETURN elementId(c) as id
            """
            
            properties = {
                "title": report.get("title", ""),
                "summary": report.get("summary", ""),
                "rating": float(report.get("rating", 0.0)),
                "rating_explanation": report.get("rating_explanation", ""),
                "findings": json.dumps(report.get("findings", [])),
                "full_report": report.get("full_report", "") # Optional markdown version
            }
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
            
            result = session.run(query, community_id=community_id, level=level, properties=properties)
            record = result.single()
            
            report_node_id = str(record["id"]) if record else None
            
            # Link entities in the community to this report
            entities = report.get("entities", [])
            if entities and report_node_id:
                link_query = """
                UNWIND $entity_ids as entity_id
                MATCH (e) WHERE elementId(e) = entity_id
                MATCH (c) WHERE elementId(c) = $report_id
                MERGE (e)-[:BELONGS_TO_COMMUNITY]->(c)
                """
                session.run(link_query, entity_ids=entities, report_id=report_node_id)
                
            return report_node_id

    def expand_from_nodes(self, node_ids: List[str], max_hops: int = 2, limit: int = 50, namespace: Optional[str] = None, index_name: Optional[str] = None, query_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Expand graph from seed nodes, retrieving connected nodes and relationships.

        Args:
            node_ids: List of seed node IDs
            max_hops: Maximum number of hops to expand (1 = direct connections)
                      If query_type is provided, this is overridden by adaptive hop count
            limit: Maximum number of nodes to return
            namespace: Optional namespace to filter connected nodes and relationships
            index_name: Optional index_name (collection name) to filter connected nodes and relationships
            query_type: Optional query type for adaptive hop count ('technical'=1, 'exploratory'=2, 'relational'=3)

        Returns:
            Dictionary with nodes, relationships, and hops_executed
        """
        if not node_ids:
            return {"nodes": [], "relationships": [], "hops_executed": 0}

        # Adaptive hop count based on query type
        HOP_CONFIG = {
            'technical': 1,
            'exploratory': 2,
            'relational': 3
        }

        if query_type and query_type in HOP_CONFIG:
            max_hops = HOP_CONFIG[query_type]
            logger.info(f"Adaptive expansion: query_type={query_type}, max_hops={max_hops}")

        with self._get_session() as session:
            # Build WHERE clause for partition filtering
            where_clauses = []
            params: Dict[str, Any] = {"node_ids": node_ids, "limit": limit}

            if namespace is not None:
                where_clauses.append("connected.namespace = $namespace")
                params["namespace"] = namespace
            if index_name is not None:
                where_clauses.append("connected.index_name = $index_name")
                params["index_name"] = index_name

            # Build where clause for connected nodes
            connected_where = ""
            if where_clauses:
                connected_where = "WHERE " + " AND ".join(where_clauses)

            # Use variable-length path pattern for multi-hop expansion
            # Pattern: (start)-[*1..max_hops]-(connected)
            hop_pattern = f"*1..{max_hops}" if max_hops > 1 else ""

            query = f"""
            MATCH path = (start)-[{hop_pattern}]-(connected)
            WHERE elementId(start) IN $node_ids
            {connected_where}
            WITH path, start, connected, relationships(path) as rels
            LIMIT $limit
            RETURN DISTINCT
                elementId(start) as start_id,
                labels(start) as start_labels,
                properties(start) as start_props,
                elementId(connected) as connected_id,
                labels(connected) as connected_labels,
                properties(connected) as connected_props,
                [r IN rels | {{
                    id: elementId(r),
                    type: type(r),
                    properties: properties(r),
                    source_id: elementId(startNode(r)),
                    target_id: elementId(endNode(r))
                }}] as path_rels
            """

            result = session.run(query, **params)  # type: ignore

            nodes = []
            relationships = []
            seen_nodes = set()
            seen_rels = set()

            for record in result:
                start_id = record.get("start_id")
                connected_id = record.get("connected_id")
                path_rels = record.get("path_rels", [])

                # Add start node
                if start_id and start_id not in seen_nodes:
                    seen_nodes.add(start_id)
                    nodes.append({
                        "id": str(start_id),
                        "label": record.get("start_labels")[0] if record.get("start_labels") else "Unknown",
                        "properties": self._convert_properties(record.get("start_props", {}))
                    })

                # Add connected node
                if connected_id and connected_id not in seen_nodes:
                    seen_nodes.add(connected_id)
                    nodes.append({
                        "id": str(connected_id),
                        "label": record.get("connected_labels")[0] if record.get("connected_labels") else "Unknown",
                        "properties": self._convert_properties(record.get("connected_props", {}))
                    })

                # Add all relationships in the path
                for rel_data in path_rels:
                    rel_id = rel_data.get("id")
                    if rel_id and rel_id not in seen_rels:
                        seen_rels.add(rel_id)
                        relationships.append({
                            "id": str(rel_id),
                            "type": rel_data.get("type", "UNKNOWN"),
                            "properties": self._convert_properties(rel_data.get("properties", {})),
                            "source_id": str(rel_data.get("source_id")),
                            "target_id": str(rel_data.get("target_id"))
                        })

            logger.info(f"Expansion complete: {len(nodes)} nodes, {len(relationships)} relationships (max_hops={max_hops})")

            return {
                "nodes": nodes,
                "relationships": relationships,
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "hops_executed": max_hops
            }

    def find_nodes_by_source_id(self, source_id: str, limit: int = 10, namespace: Optional[str] = None, index_name: Optional[str] = None) -> List[Node]:
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
        with self._get_session() as session:
            # Build WHERE clause
            # Check if source_id matches chunk_id, metadata_id OR is in source_ids list
            condition = "($source_id IN n.source_ids)"
            where_clauses = [condition]
            
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
            RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
            LIMIT $limit
            """
            logger.debug(f"Searching for nodes with source_id: {source_id}, namespace: {namespace}")
            result = session.run(query, **params)  # type: ignore
            
            nodes = []
            for record in result:
                node_id = str(record["id"])
                label = record["labels"][0] if record["labels"] else "Unknown"
                props = self._convert_properties(dict(record["properties"]))  # type: ignore
                nodes.append(Node(
                    id=node_id,
                    label=label,
                    properties=props  # type: ignore
                ))
            logger.info(f"Total nodes found for source_id {source_id}: {len(nodes)}")
            return nodes
