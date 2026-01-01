"""
Dynamic graph expansion utilities for GraphRAG.

Provides adaptive multi-hop expansion based on query type and graph density.
"""

import logging
from typing import List, Dict, Any, Set, Optional
from ..models import Node, Relationship

logger = logging.getLogger(__name__)


class GraphExpander:
    """
    Adaptive graph expansion with dynamic hop count.
    """

    # Hop configuration by query type
    HOP_CONFIG = {
        'technical': 1,      # Precise queries need focused, 1-hop context
        'exploratory': 2,    # Broad queries benefit from wider, 2-hop exploration
        'relational': 3      # Relationship queries need deeper, 3-hop paths
    }

    # Early stop thresholds
    MIN_NEW_NODES_PER_HOP = 3  # Stop if fewer than N new nodes found
    MAX_NODES_ABSOLUTE = 200   # Hard limit on total nodes

    def __init__(self, repository):
        """
        Initialize graph expander.

        Args:
            repository: GraphRepository instance for Neo4j queries
        """
        self.repository = repository

    async def expand_from_seeds(
        self,
        seed_node_ids: List[str],
        query_type: str = 'exploratory',
        max_nodes: int = 50,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        relationship_types: Optional[List[str]] = None,
        min_relationship_weight: float = 0.0
    ) -> Dict[str, Any]:
        """
        Expand graph from seed nodes with adaptive hop count.

        Args:
            seed_node_ids: Starting node IDs
            query_type: Query type for hop count determination
            max_nodes: Maximum total nodes to retrieve
            namespace: Optional namespace filter
            index_name: Optional index filter
            relationship_types: Optional list of relationship types to follow
            min_relationship_weight: Minimum weight threshold for relationships

        Returns:
            Dictionary with expanded nodes and relationships:
            {
                "nodes": List[Node],
                "relationships": List[Relationship],
                "hops_executed": int,
                "expansion_stats": Dict
            }
        """
        if not seed_node_ids:
            logger.warning("No seed nodes provided for expansion")
            return {
                "nodes": [],
                "relationships": [],
                "hops_executed": 0,
                "expansion_stats": {}
            }

        # Determine max hops based on query type
        max_hops = self.HOP_CONFIG.get(query_type, 2)
        logger.info(f"Starting graph expansion with query_type={query_type}, max_hops={max_hops}, "
                   f"max_nodes={max_nodes}, seed_nodes={len(seed_node_ids)}")

        # Initialize tracking sets
        expanded_node_ids: Set[str] = set(seed_node_ids)
        all_nodes: Dict[str, Node] = {}
        all_relationships: List[Relationship] = []

        # Fetch seed nodes
        for seed_id in seed_node_ids:
            node = self.repository.find_node_by_id(seed_id)
            if node:
                all_nodes[seed_id] = node

        current_layer = list(seed_node_ids)
        hops_executed = 0

        # Expansion statistics
        stats = {
            "seed_nodes": len(seed_node_ids),
            "nodes_by_hop": [len(seed_node_ids)],
            "relationships_by_hop": []
        }

        # Multi-hop expansion
        for hop in range(max_hops):
            if not current_layer:
                logger.info(f"No more nodes to expand from at hop {hop}")
                break

            if len(expanded_node_ids) >= self.MAX_NODES_ABSOLUTE:
                logger.warning(f"Reached absolute node limit ({self.MAX_NODES_ABSOLUTE}) at hop {hop}")
                break

            logger.info(f"Hop {hop+1}/{max_hops}: Expanding from {len(current_layer)} nodes")

            # Build Cypher query for next hop
            query, params = self._build_expansion_query(
                current_nodes=current_layer,
                already_expanded=list(expanded_node_ids),
                max_nodes=max_nodes - len(expanded_node_ids),
                namespace=namespace,
                index_name=index_name,
                relationship_types=relationship_types,
                min_weight=min_relationship_weight
            )

            # Execute expansion
            try:
                results = self.repository.execute_query(query, params)
            except Exception as e:
                logger.error(f"Error executing expansion query at hop {hop}: {e}")
                break

            # Process results
            new_nodes_this_hop = []
            new_relationships_this_hop = []

            for record in results:
                # Extract target node data (fields are returned as scalars, not nested objects)
                target_id = record.get('target_id')
                label = record.get('label')
                properties = record.get('properties', {})

                if not target_id:
                    continue

                # Create or get target node
                if target_id not in all_nodes:
                    target_node = Node(
                        id=target_id,
                        label=label or 'Unknown',
                        properties=properties
                    )
                    all_nodes[target_id] = target_node
                    new_nodes_this_hop.append(target_id)
                    expanded_node_ids.add(target_id)

                # Create relationship
                rel_id = record.get('rel_id')
                if rel_id:
                    relationship = Relationship(
                        id=rel_id,
                        type=record.get('rel_type', 'RELATED_TO'),
                        source_id=record.get('source_id'),
                        target_id=record.get('target_id_2'),  # target_id from relationship
                        properties=record.get('rel_properties', {})
                    )
                    all_relationships.append(relationship)
                    new_relationships_this_hop.append(relationship)

            # Update statistics
            stats["nodes_by_hop"].append(len(new_nodes_this_hop))
            stats["relationships_by_hop"].append(len(new_relationships_this_hop))

            logger.info(f"Hop {hop+1} results: {len(new_nodes_this_hop)} new nodes, "
                       f"{len(new_relationships_this_hop)} relationships")

            # Early stopping conditions
            if len(new_nodes_this_hop) < self.MIN_NEW_NODES_PER_HOP:
                logger.info(f"Early stop: Found only {len(new_nodes_this_hop)} new nodes "
                           f"(threshold: {self.MIN_NEW_NODES_PER_HOP})")
                hops_executed = hop + 1
                break

            if len(expanded_node_ids) >= max_nodes:
                logger.info(f"Reached max_nodes limit ({max_nodes})")
                hops_executed = hop + 1
                break

            # Prepare next layer
            current_layer = new_nodes_this_hop
            hops_executed = hop + 1

        # Final statistics
        stats["total_nodes"] = len(all_nodes)
        stats["total_relationships"] = len(all_relationships)
        stats["hops_executed"] = hops_executed

        logger.info(f"Graph expansion complete: {stats['total_nodes']} nodes, "
                   f"{stats['total_relationships']} relationships in {hops_executed} hops")

        return {
            "nodes": list(all_nodes.values()),
            "relationships": all_relationships,
            "hops_executed": hops_executed,
            "expansion_stats": stats
        }

    def _build_expansion_query(
        self,
        current_nodes: List[str],
        already_expanded: List[str],
        max_nodes: int,
        namespace: Optional[str],
        index_name: Optional[str],
        relationship_types: Optional[List[str]],
        min_weight: float
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build Cypher query for graph expansion.

        Returns:
            Tuple of (query_string, parameters)
        """
        # Base query
        query_parts = ["MATCH (n)-[r]-(m)"]

        # Conditions - use elementId() instead of .id property
        conditions = ["elementId(n) IN $current_nodes", "NOT elementId(m) IN $already_expanded"]

        if namespace:
            conditions.append("m.namespace = $namespace")

        if index_name:
            conditions.append("m.index_name = $index_name")

        if relationship_types:
            rel_types = '|'.join(relationship_types)
            conditions.append(f"type(r) IN $relationship_types")

        if min_weight > 0:
            conditions.append("r.weight >= $min_weight")

        # Combine conditions
        query_parts.append("WHERE " + " AND ".join(conditions))

        # Return - use elementId() for node identifiers
        query_parts.append("""
            RETURN
                elementId(m) as target_id,
                labels(m)[0] as label,
                properties(m) as properties,
                elementId(r) as rel_id,
                type(r) as rel_type,
                elementId(n) as source_id,
                elementId(m) as target_id_2,
                properties(r) as rel_properties
            ORDER BY COALESCE(r.weight, 0) DESC
        """)

        query_parts.append(f"LIMIT {max_nodes}")

        query = "\n".join(query_parts)

        # Parameters
        params = {
            "current_nodes": current_nodes,
            "already_expanded": already_expanded,
        }

        if namespace:
            params["namespace"] = namespace
        if index_name:
            params["index_name"] = index_name
        if relationship_types:
            params["relationship_types"] = relationship_types
        if min_weight > 0:
            params["min_weight"] = min_weight

        return query, params

    def filter_nodes_by_relevance(
        self,
        nodes: List[Node],
        query_keywords: List[str],
        top_k: int = 50
    ) -> List[Node]:
        """
        Filter and rank nodes by relevance to query keywords.

        Args:
            nodes: List of nodes to filter
            query_keywords: Keywords from user query
            top_k: Number of top nodes to return

        Returns:
            Filtered and ranked list of nodes
        """
        if not query_keywords:
            return nodes[:top_k]

        # Score each node
        scored_nodes = []
        for node in nodes:
            score = self._calculate_node_relevance(node, query_keywords)
            scored_nodes.append((node, score))

        # Sort by score descending
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        return [node for node, score in scored_nodes[:top_k]]

    def _calculate_node_relevance(self, node: Node, keywords: List[str]) -> float:
        """
        Calculate relevance score for a node based on keywords.

        Args:
            node: Node to score
            keywords: Query keywords

        Returns:
            Relevance score (higher is better)
        """
        score = 0.0

        # Check node properties
        for prop_value in node.properties.values():
            if isinstance(prop_value, str):
                prop_lower = prop_value.lower()
                for keyword in keywords:
                    if keyword.lower() in prop_lower:
                        score += 1.0

        return score
