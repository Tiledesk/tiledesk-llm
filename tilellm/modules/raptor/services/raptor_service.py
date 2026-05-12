"""
RAPTOR Service - Core business logic for hierarchical summarization.

Implements:
- Phase 2: Clustering and Summarization
- Phase 4: Collapsed tree indexing
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from tilellm.modules.raptor.models.models import (
    RaptorTree,
    RaptorNode,
    RaptorLevel,
    RaptorConfig,
    RaptorResponse,
)
from tilellm.modules.raptor.repository import RaptorRepository
from tilellm.modules.raptor.prompts import (
    RAPTOR_SUMMARY_PROMPT,
    RAPTOR_HIERARCHICAL_SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


class RaptorService:
    """
    Service for RAPTOR hierarchical summarization.
    
    Handles:
    - Chunk retrieval from vector store
    - Clustering chunks into groups
    - LLM-based summary generation
    - Recursive summarization (higher levels)
    - Tree structure management
    - Summary embedding and indexing
    """
    
    def __init__(self, repo: RaptorRepository):
        """
        Initialize RAPTOR service.
        
        Args:
            repo: RaptorRepository instance
        """
        self.repo = repo
    
    async def build_raptor_tree(
        self,
        chunks: List[Document],
        namespace: str,
        doc_id: str,
        llm: Any,
        embeddings: Any,
        vector_repo: Any,
        engine: Any,
        config: Optional[RaptorConfig] = None,
        sparse_encoder: Any = None,
    ) -> RaptorResponse:
        """
        Build complete RAPTOR tree from chunks.

        Args:
            chunks: List of chunk documents from vector store
            namespace: Namespace identifier
            doc_id: Document ID
            llm: LLM instance for summarization
            embeddings: Embeddings model (dense)
            vector_repo: Vector store repository
            engine: Engine configuration object
            config: Optional configuration overrides
            sparse_encoder: Optional sparse encoder for hybrid search

        Returns:
            RaptorResponse with tree statistics
        """
        start_time = time.time()
        config = config or RaptorConfig()

        try:
            logger.info(f"Building RAPTOR tree for doc {doc_id} with {len(chunks)} chunks")
            
            if sparse_encoder:
                logger.info("Sparse encoder provided, will index sparse vectors for hybrid search")

            if len(chunks) < config.cluster_size:
                logger.info(f"Not enough chunks ({len(chunks)}) for RAPTOR summarization")
                return RaptorResponse(
                    success=False,
                    error=f"Not enough chunks: {len(chunks)} < {config.cluster_size}",
                    total_chunks=len(chunks),
                )

            # Initialize tree structure
            tree_id = f"raptor_{doc_id}_{int(time.time())}"
            tree = RaptorTree(
                tree_id=tree_id,
                namespace=namespace,
                doc_id=doc_id,
                config=config,
                created_at=datetime.utcnow().isoformat(),
            )

            # Level 0: Create leaf nodes from chunks
            leaf_nodes = await self._create_leaf_nodes(chunks, doc_id, tree_id)
            tree.leaf_ids = [node.node_id for node in leaf_nodes]
            tree.levels[0] = [node.node_id for node in leaf_nodes]

            for node in leaf_nodes:
                tree.nodes[node.node_id] = node

            # Index leaf node embeddings in RAPTOR namespace (batch)
            logger.info(f"Indexing {len(leaf_nodes)} leaf chunks in RAPTOR namespace")
            await self.repo.batch_index_nodes(
                nodes=leaf_nodes,
                namespace=namespace,
                repo=vector_repo,
                engine=engine,
                llm_embeddings=embeddings,
                sparse_encoder=sparse_encoder,
            )
            
            # Level 1+: Recursive summarization
            current_level_nodes = leaf_nodes
            current_level = RaptorLevel.LEVEL_0
            
            while (current_level.value < config.max_levels and 
                   len(current_level_nodes) >= config.cluster_size):
                
                next_level = RaptorLevel(current_level.value + 1)
                
                # Cluster nodes into groups
                clusters = await self._cluster_nodes(
                    current_level_nodes,
                    config.cluster_size,
                    embeddings
                )
                
                # Generate summaries for all clusters in parallel
                summary_nodes = list(await asyncio.gather(*[
                    self._generate_summary_for_cluster(
                        cluster=cluster,
                        level=next_level,
                        llm=llm,
                        config=config,
                        doc_id=doc_id,
                        namespace=namespace,
                        tree_id=tree_id,
                    )
                    for cluster in clusters
                ]))

                # Batch-index all summary embeddings in one vector store call
                nodes_to_index = [n for n in summary_nodes if n.summary]
                if nodes_to_index:
                    await self.repo.batch_index_nodes(
                        nodes=nodes_to_index,
                        namespace=namespace,
                        repo=vector_repo,
                        engine=engine,
                        llm_embeddings=embeddings,
                        sparse_encoder=sparse_encoder,
                    )
                
                # Add summary nodes to tree
                tree.levels[next_level.value] = [
                    node.node_id for node in summary_nodes
                ]
                
                for node in summary_nodes:
                    tree.nodes[node.node_id] = node
                
                # Prepare for next iteration
                current_level_nodes = summary_nodes
                current_level = next_level
            
            # Set root nodes (highest level)
            tree.root_ids = tree.levels.get(current_level.value, [])
            tree.total_nodes = len(tree.nodes)
            
            # Save tree to Redis
            await self.repo.save_tree(tree)
            
            processing_time = time.time() - start_time
            
            # Build level statistics
            level_stats = {
                level: len(nodes) for level, nodes in tree.levels.items()
            }
            
            logger.info(
                f"RAPTOR tree built successfully: {tree.total_nodes} nodes, "
                f"{len(tree.levels)} levels in {processing_time:.2f}s"
            )
            
            return RaptorResponse(
                success=True,
                tree_id=tree_id,
                total_chunks=len(leaf_nodes),
                total_summaries=tree.total_nodes - len(leaf_nodes),
                levels_created=len(tree.levels),
                level_stats=level_stats,
                processing_time_seconds=processing_time,
                model_used=getattr(llm, 'model_name', None),
                tree=tree,
            )
            
        except Exception as e:
            logger.error(f"Error building RAPTOR tree: {e}", exc_info=True)
            return RaptorResponse(
                success=False,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )
    
    async def _create_leaf_nodes(
        self,
        chunks: List[Document],
        doc_id: str,
        tree_id: str,
    ) -> List[RaptorNode]:
        """
        Create leaf nodes (Level 0) from chunks.

        Args:
            chunks: List of chunk documents
            doc_id: Document ID
            tree_id: RAPTOR tree ID

        Returns:
            List of RaptorNode instances
        """
        leaf_nodes = []

        for idx, chunk in enumerate(chunks):
            node_id = f"chunk_{doc_id}_{idx}_{uuid.uuid4().hex[:8]}"

            node = RaptorNode(
                node_id=node_id,
                level=RaptorLevel.LEVEL_0,
                content=chunk.page_content,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", 0),
                    "raptor_tree_id": tree_id,  # Reference to tree
                },
                created_at=datetime.utcnow().isoformat(),
            )

            leaf_nodes.append(node)

        logger.debug(f"Created {len(leaf_nodes)} leaf nodes")
        return leaf_nodes
    
    async def _cluster_nodes(
        self,
        nodes: List[RaptorNode],
        cluster_size: int,
        embeddings: Any
    ) -> List[List[RaptorNode]]:
        """
        Cluster nodes into groups for summarization.

        Uses either:
        - RAPTOR clustering (UMAP + GMM) for semantic grouping
        - Sequential grouping as fallback

        Args:
            nodes: Nodes to cluster
            cluster_size: Target cluster size
            embeddings: Embeddings model

        Returns:
            List of clusters (each cluster is a list of nodes)
        """
        if len(nodes) <= cluster_size:
            # All nodes fit in one cluster
            return [nodes]

        # Try RAPTOR clustering (UMAP + GMM)
        try:
            from tilellm.modules.raptor.utils.clustering import (
                RAPTORClustering,
                RAPTOR_CLUSTERING_AVAILABLE,
            )

            if RAPTOR_CLUSTERING_AVAILABLE:
                logger.info("Using RAPTOR clustering (UMAP + GMM)")

                # Store embeddings in node metadata for clustering
                texts = [node.content for node in nodes]
                node_embeddings = await embeddings.aembed_documents(texts)

                for node, emb in zip(nodes, node_embeddings):
                    node.metadata["embedding_default"] = emb

                # Run CPU-bound UMAP+GMM in thread pool to avoid blocking event loop
                clustering = RAPTORClustering()
                clusters = await asyncio.to_thread(
                    lambda: clustering.perform_clustering(
                        nodes=nodes,
                        embedding_model_name="default",
                        max_length_in_cluster=cluster_size * 500,
                        verbose=True,
                    )
                )

                # Validate clustering result is not empty
                if not clusters:
                    logger.warning("RAPTOR clustering returned empty result, falling back to sequential")
                    raise ValueError("Empty clustering result")

                logger.info(f"RAPTOR clustering created {len(clusters)} clusters")
                return clusters

        except Exception as e:
            logger.warning(f"RAPTOR clustering failed: {e}, falling back to sequential")

        # Fallback: sequential grouping
        logger.info("Using sequential clustering (fallback)")
        clusters = []
        for i in range(0, len(nodes), cluster_size):
            clusters.append(nodes[i:i + cluster_size])
        return clusters

    async def _generate_summary_for_cluster(
        self,
        cluster: List[RaptorNode],
        level: RaptorLevel,
        llm: Any,
        config: RaptorConfig,
        doc_id: str,
        namespace: str,
        tree_id: str,
    ) -> RaptorNode:
        """
        Generate summary for a cluster of nodes.
        
        Args:
            cluster: List of nodes to summarize
            level: Target level for summary
            llm: LLM instance
            config: Configuration
            doc_id: Document ID
            namespace: Namespace
            
        Returns:
            RaptorNode with summary
        """
        # Prepare context from cluster
        context_parts = []
        child_ids = []
        
        for node in cluster:
            context_parts.append(node.content)
            child_ids.append(node.node_id)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Choose prompt based on level
        if level.value == 1:
            prompt_text = RAPTOR_SUMMARY_PROMPT.format(context=context)
        else:
            prompt_text = RAPTOR_HIERARCHICAL_SUMMARY_PROMPT.format(context=context)
        
        # Generate summary with LLM
        try:
            summary_response = await llm.ainvoke(prompt_text)
            summary_text = summary_response.content.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback: truncate context to approximate token limit (~4 chars per token)
            summary_text = context[:config.summary_max_tokens * 4]
        
        # Create summary node
        node_id = f"summary_{doc_id}_L{level.value}_{uuid.uuid4().hex[:8]}"
        
        # Determine parent (will be set in next iteration)
        parent_id = None  # Will be updated if this node gets a parent
        
        summary_node = RaptorNode(
            node_id=node_id,
            level=level,
            content=summary_text,
            summary=summary_text,
            child_ids=child_ids,
            parent_id=parent_id,
            metadata={
                "doc_id": doc_id,
                "namespace": namespace,
                "num_children": len(child_ids),
                "cluster_theme": f"Summary of {len(child_ids)} items",
                "raptor_tree_id": tree_id,
            },
            created_at=datetime.utcnow().isoformat(),
            model_used=getattr(llm, 'model_name', None),
        )
        
        logger.debug(
            f"Generated {level.name} summary ({len(summary_text)} chars) "
            f"from {len(cluster)} nodes"
        )
        
        return summary_node
    
    
    async def retrieve_collapsed_tree(
        self,
        query: str,
        namespace: str,
        vector_repo: Any,
        embeddings: Any,
        top_k: int = 5,
        top_k_per_level: int = 3,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from collapsed tree (all levels in same vector space).

        Uses the SAME vector store as application with namespace: {namespace}-raptor
        All RAPTOR nodes (chunks + summaries) are stored together.

        Args:
            query: Search query
            namespace: Namespace to search (original document namespace)
            vector_repo: Vector store repository (same as application uses)
            embeddings: Embeddings model
            top_k: Total results to return
            top_k_per_level: Max results per level
            doc_id: Optional document filter

        Returns:
            List of retrieved nodes with scores
        """
        try:
            # Generate query embedding
            query_embedding = embeddings.embed_query(query)

            # Use SAME vector store with dedicated RAPTOR namespace
            # Example: "pippo" -> "pippo-raptor"
            raptor_namespace = self.repo.get_raptor_namespace(namespace)

            # Build filter
            filter_dict = None
            if doc_id:
                filter_dict = {"doc_id": doc_id}

            # Retrieve from RAPTOR namespace (contains all levels: chunks + summaries)
            results = await vector_repo.similarity_search_with_score(
                namespace=raptor_namespace,
                query_embedding=query_embedding,
                k=top_k,
                filter=filter_dict,
            )

            # Format results
            all_results = []
            seen_ids = set()

            for doc, score in results:
                node_id = doc.metadata.get("node_id")
                if node_id and node_id not in seen_ids:
                    seen_ids.add(node_id)
                    all_results.append({
                        "content": doc.page_content,
                        "score": float(score),
                        "metadata": doc.metadata,
                        "level": doc.metadata.get("level", 0),
                        "is_summary": doc.metadata.get("is_summary", False),
                        "is_raptor_node": doc.metadata.get("is_raptor_node", False),
                    })

            # Sort by score and limit
            all_results.sort(key=lambda x: x["score"])
            return all_results[:top_k]

        except Exception as e:
            logger.error(f"Error in collapsed tree retrieval: {e}", exc_info=True)
            return []
