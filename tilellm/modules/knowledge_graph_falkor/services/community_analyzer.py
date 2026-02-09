"""
Document Community Analyzer
Analyzes collections of documents to find communities and topics.
"""

import logging
from typing import Dict, Any, List, Optional
from tilellm.models import Engine
from tilellm.modules.knowledge_graph.services.clustering import ClusterService

logger = logging.getLogger(__name__)

class DocumentCommunityAnalyzer:
    """
    Analyzes document collections using community detection.
    """

    def __init__(self, cluster_service: ClusterService):
        self.cluster_service = cluster_service

    async def analyze_collection(
        self,
        namespace: str,
        engine: Engine,
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run community detection on a document collection.
        """
        logger.info(f"Analyzing collection in namespace {namespace}")
        
        # 1. Build Document Similarity Graph (Simulated or via existing relationships)
        # In our architecture, documents are linked via shared entities.
        # So we can run community detection on the entity graph or a projected document graph.
        
        # We'll use the existing hierarchical clustering which works on the entity graph
        # This effectively groups documents if they share many entities.
        
        cluster_stats = await self.cluster_service.perform_clustering_leiden(
            level=0,
            namespace=namespace,
            index_name=engine.index_name if hasattr(engine, 'index_name') else None,
            engine_name=engine.name,
            engine_type=engine.type if hasattr(engine, 'type') else None
        )
        
        return {
            "status": "success",
            "communities": cluster_stats.get("communities_detected", 0),
            "reports": cluster_stats.get("reports_created", 0),
            "details": cluster_stats
        }
