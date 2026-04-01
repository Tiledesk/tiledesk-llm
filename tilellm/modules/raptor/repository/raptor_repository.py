"""
Repository for RAPTOR tree storage and retrieval.
Handles persistence of tree structure and embeddings.
"""

import json
import logging
from typing import Dict, List, Optional, Any

from tilellm.modules.raptor.models.models import (
    RaptorTree,
    RaptorNode,
)

logger = logging.getLogger(__name__)


class RaptorRepository:
    """
    Repository for RAPTOR tree persistence.
    
    Stores:
    - Tree structure in Redis (as JSON)
    - Summary embeddings in vector store (alongside original chunks)
    - Node metadata in Redis hash
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize RAPTOR repository.

        Stores:
        - Tree structure in Redis (as JSON)
        - Summary embeddings in SAME vector store as application (with -raptor suffix)
        - Node metadata in Redis hash

        Args:
            redis_client: Redis client for tree structure storage
        """
        self.redis_client = redis_client
        self._tree_prefix = "raptor:tree:"
        self._node_prefix = "raptor:node:"
        # Use same vector store with dedicated namespace: {original}-raptor
        self._raptor_namespace_suffix = "-raptor"

    def _get_tree_key(self, tree_id: str) -> str:
        """Get Redis key for tree structure."""
        return f"{self._tree_prefix}{tree_id}"

    def _get_node_key(self, node_id: str) -> str:
        """Get Redis key for node metadata."""
        return f"{self._node_prefix}{node_id}"

    def get_raptor_namespace(self, namespace: str) -> str:
        """
        Get vector store namespace for RAPTOR nodes.

        Uses the SAME vector store as the application, with a dedicated namespace.
        Example: if original namespace is "pippo", RAPTOR namespace is "pippo-raptor"

        Args:
            namespace: Original document namespace

        Returns:
            RAPTOR namespace (original + "-raptor")
        """
        return f"{namespace}{self._raptor_namespace_suffix}"
    
    async def save_tree(self, tree: RaptorTree) -> bool:
        """
        Save RAPTOR tree to Redis.
        
        Args:
            tree: RaptorTree instance to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tree_key = self._get_tree_key(tree.tree_id)
            
            # Serialize tree to JSON
            tree_data = tree.model_dump(mode='json')

            # Store in Redis (use json.dumps, not str() which produces invalid JSON)
            if self.redis_client:
                await self.redis_client.set(tree_key, json.dumps(tree_data, default=str))
                logger.info(f"Saved RAPTOR tree {tree.tree_id} to Redis")
            
            # Save individual nodes
            for node_id, node in tree.nodes.items():
                await self.save_node(node)
            
            logger.info(f"Saved {len(tree.nodes)} nodes for tree {tree.tree_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving RAPTOR tree: {e}", exc_info=True)
            return False
    
    async def save_node(self, node: RaptorNode) -> bool:
        """
        Save individual node metadata to Redis.

        Args:
            node: RaptorNode to save

        Returns:
            True if successful
        """
        try:
            node_key = self._get_node_key(node.node_id)
            node_data = node.model_dump(mode='json')

            if self.redis_client:
                # Store as JSON string (consistent with save_tree), not hash
                # This avoids issues with complex values (lists, dicts) in hset
                await self.redis_client.set(node_key, json.dumps(node_data, default=str))

            logger.debug(f"Saved node {node.node_id} at level {node.level}")
            return True

        except Exception as e:
            logger.error(f"Error saving node: {e}", exc_info=True)
            return False
    
    async def get_tree(self, tree_id: str) -> Optional[RaptorTree]:
        """
        Retrieve RAPTOR tree from Redis.
        
        Args:
            tree_id: Tree identifier
            
        Returns:
            RaptorTree instance or None if not found
        """
        try:
            tree_key = self._get_tree_key(tree_id)
            
            if self.redis_client:
                tree_data = await self.redis_client.get(tree_key)
                if tree_data:
                    # Parse JSON and reconstruct RaptorTree
                    import json
                    tree_dict = json.loads(tree_data)
                    return RaptorTree(**tree_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving RAPTOR tree: {e}", exc_info=True)
            return None
    
    async def get_node(self, node_id: str) -> Optional[RaptorNode]:
        """
        Retrieve individual node from Redis.

        Args:
            node_id: Node identifier

        Returns:
            RaptorNode or None
        """
        try:
            node_key = self._get_node_key(node_id)

            if self.redis_client:
                # Retrieve from JSON string (consistent with save_node)
                raw = await self.redis_client.get(node_key)
                if raw:
                    node_dict = json.loads(raw)
                    return RaptorNode(**node_dict)

            return None

        except Exception as e:
            logger.error(f"Error retrieving node: {e}", exc_info=True)
            return None
    
    async def get_tree_by_doc_id(self, doc_id: str, namespace: str) -> Optional[RaptorTree]:
        """
        Find RAPTOR tree for a specific document.
        
        Args:
            doc_id: Document ID
            namespace: Namespace
            
        Returns:
            RaptorTree or None
        """
        try:
            # Search for tree with matching doc_id and namespace
            tree_key_pattern = f"{self._tree_prefix}*"

            if self.redis_client:
                # Use scan_iter instead of keys() to avoid blocking Redis
                keys = [k async for k in self.redis_client.scan_iter(tree_key_pattern)]
                for key in keys:
                    tree_data = await self.redis_client.get(key)
                    if tree_data:
                        import json
                        tree_dict = json.loads(tree_data)
                        if (tree_dict.get("doc_id") == doc_id and 
                            tree_dict.get("namespace") == namespace):
                            return RaptorTree(**tree_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding tree for document: {e}", exc_info=True)
            return None
    
    async def delete_tree(self, tree_id: str) -> bool:
        """
        Delete RAPTOR tree and all associated nodes.
        
        Args:
            tree_id: Tree identifier
            
        Returns:
            True if successful
        """
        try:
            # Get tree first to know which nodes to delete
            tree = await self.get_tree(tree_id)
            if not tree:
                logger.warning(f"Tree {tree_id} not found for deletion")
                return False
            
            # Delete all nodes
            for node_id in tree.nodes.keys():
                await self.delete_node(node_id)
            
            # Delete tree structure
            tree_key = self._get_tree_key(tree_id)
            if self.redis_client:
                await self.redis_client.delete(tree_key)
            
            logger.info(f"Deleted RAPTOR tree {tree_id} with {len(tree.nodes)} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting RAPTOR tree: {e}", exc_info=True)
            return False
    
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete individual node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if successful
        """
        try:
            node_key = self._get_node_key(node_id)
            if self.redis_client:
                await self.redis_client.delete(node_key)
            logger.debug(f"Deleted node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting node: {e}", exc_info=True)
            return False
    
    async def list_trees(self, namespace: str) -> List[str]:
        """
        List all RAPTOR trees in a namespace.
        
        Args:
            namespace: Namespace to search
            
        Returns:
            List of tree IDs
        """
        try:
            tree_ids = []
            tree_key_pattern = f"{self._tree_prefix}*"

            if self.redis_client:
                # Use scan_iter instead of keys() to avoid blocking Redis
                keys = [k async for k in self.redis_client.scan_iter(tree_key_pattern)]
                for key in keys:
                    tree_data = await self.redis_client.get(key)
                    if tree_data:
                        import json
                        tree_dict = json.loads(tree_data)
                        if tree_dict.get("namespace") == namespace:
                            tree_ids.append(tree_dict.get("tree_id"))
            
            return tree_ids
            
        except Exception as e:
            logger.error(f"Error listing trees: {e}", exc_info=True)
            return []
    
    async def get_summary_namespace(self, namespace: str) -> str:
        """
        Get the vector store namespace for RAPTOR nodes.
        Deprecated: use get_raptor_namespace() instead.

        Args:
            namespace: Original namespace

        Returns:
            RAPTOR namespace (original + "-raptor")
        """
        return self.get_raptor_namespace(namespace)
    
    async def index_summary_embedding(
        self,
        node: RaptorNode,
        namespace: str,
        repo: Any,
        engine: Any,
        llm_embeddings: Any,
        sparse_encoder: Any = None,
    ) -> bool:
        """
        Index RAPTOR node embedding in vector store.

        Uses the SAME vector store as the application with namespace: {original}-raptor
        Example: if original namespace is "pippo", RAPTOR nodes are stored in "pippo-raptor"

        Args:
            node: RaptorNode with summary content
            namespace: Original document namespace (RAPTOR namespace will be derived)
            repo: Vector store repository instance (same as application uses)
            engine: Engine configuration object
            llm_embeddings: Embeddings model instance
            sparse_encoder: Optional sparse encoder for hybrid search

        Returns:
            True if successful
        """
        try:
            from langchain_core.documents import Document

            # Create document with summary content
            child_ids_str = ",".join(node.child_ids) if node.child_ids else ""
            doc = Document(
                page_content=node.summary or node.content,
                metadata={
                    **node.metadata,
                    "node_id": node.node_id,
                    "level": node.level.value,
                    "is_summary": node.level.value > 0,  # Level 0 = original chunk
                    "is_raptor_node": True,
                    "child_ids": child_ids_str,  # Serialize list to string for metadata compat
                    "parent_id": node.parent_id or "",
                    "doc_id": node.metadata.get("doc_id", ""),
                    "raptor_tree_id": node.metadata.get("raptor_tree_id", ""),
                }
            )

            # Use SAME vector store with dedicated RAPTOR namespace
            raptor_namespace = self.get_raptor_namespace(namespace)

            logger.info(
                f"Indexing RAPTOR node {node.node_id} (L{node.level.value}) "
                f"in {raptor_namespace}"
            )

            # Add to vector store via aadd_documents (handles embedding internally)
            await repo.aadd_documents(
                engine=engine,
                documents=[doc],
                namespace=raptor_namespace,
                embedding_model=llm_embeddings,
                sparse_encoder=sparse_encoder,
                skip_delete=True,
                metadata_id=node.node_id,
            )

            return True

        except Exception as e:
            logger.error(f"Error indexing RAPTOR node embedding: {e}", exc_info=True)
            return False
    
    async def get_summaries_for_level(
        self,
        namespace: str,
        level: int,
        repo: Any,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve RAPTOR nodes from a specific level.

        Args:
            namespace: Original document namespace
            level: RAPTOR level to search (0 = chunks, 1+ = summaries)
            repo: Vector store repository (same as application uses)
            query_embedding: Query embedding for similarity search
            top_k: Number of results

        Returns:
            List of RAPTOR node documents with scores
        """
        try:
            # Use SAME vector store with dedicated RAPTOR namespace
            raptor_namespace = self.get_raptor_namespace(namespace)

            # Query vector store filtered by level
            results = await repo.similarity_search_with_score(
                namespace=raptor_namespace,
                query_embedding=query_embedding,
                k=top_k,
                filter={"level": str(level)},
            )

            # Format results
            formatted = []
            for doc, score in results:
                formatted.append({
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                    "level": doc.metadata.get("level", 0),
                    "is_summary": doc.metadata.get("is_summary", False),
                    "node_id": doc.metadata.get("node_id"),
                })

            return formatted

        except Exception as e:
            logger.error(f"Error retrieving RAPTOR nodes for level {level}: {e}", exc_info=True)
            return []
