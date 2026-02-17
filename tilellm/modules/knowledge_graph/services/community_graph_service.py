"""
Community Graph Service for Global Search and Community Reports.
Provides functionality to create community reports from vector store documents
and perform global search on them using DuckDB.
"""

import asyncio
import logging
import tempfile
import shutil
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_core.documents import Document
from pydantic import Field

from tilellm.models import Engine, ChatEntry
from tilellm.models.llm import TEIConfig, PineconeRerankerConfig
from tilellm.models.schemas import RepositoryItems, RepositoryNamespace
from tilellm.models import QuestionAnswer # Aggiunto per Hybrid Search
from tilellm.controller.controller_utils import fetch_question_vectors # Aggiunto per Hybrid Search
from tilellm.tools.sparse_encoders import TiledeskSparseEncoders
from ..utils import format_chat_history
from ..utils.rrf import reciprocal_rank_fusion_with_metadata
from ..utils.query_analysis import detect_query_type_with_llm, get_weight_adjustments
from ..utils.graph_expansion import GraphExpander
from ..utils.synthetic_qa import (
    enrich_reports_with_synthetic_qa,
    format_report_with_questions_for_indexing
)

from .services import GraphRAGService, GraphService
from .clustering import ClusterService
from .minio_storage import MinIOStorageService, get_minio_storage_service
from tilellm.tools.reranker import TileReranker

logger = logging.getLogger(__name__)

# Optional DuckDB for efficient Parquet querying
DUCKDB_AVAILABLE = None

def _check_duckdb_available():
    """Check if DuckDB is available."""
    global DUCKDB_AVAILABLE
    if DUCKDB_AVAILABLE is None:
        try:
            import duckdb
            DUCKDB_AVAILABLE = True
        except ImportError:
            DUCKDB_AVAILABLE = False
    return DUCKDB_AVAILABLE

def _load_parquet_with_duckdb(file_path: str, columns: Optional[List[str]] = None) -> Any:
    """Load Parquet file using DuckDB for efficient querying."""
    if not _check_duckdb_available():
        raise ImportError("DuckDB not available")
    
    import duckdb

    # Create a temporary DuckDB database in memory
    conn = duckdb.connect(database=':memory:')
    
    # Build query
    if columns:
        cols_str = ', '.join(columns)
        query = f"SELECT {cols_str} FROM '{file_path}'"
    else:
        query = f"SELECT * FROM '{file_path}'"
    
    # Execute query and fetch result as Arrow table
    result = conn.execute(query).fetch_arrow_table()
    
    # Convert to pandas DataFrame

    df = result.to_pandas()
    
    conn.close()
    return df

def _load_parquet_efficiently(file_path: str, columns: Optional[List[str]] = None):
    """Load Parquet file efficiently, using DuckDB if available, otherwise pandas."""
    if _check_duckdb_available():
        try:
            return _load_parquet_with_duckdb(file_path, columns)
        except Exception as e:
            logger.warning(f"DuckDB failed to load {file_path}: {e}. Falling back to pandas.")
    
    # Fallback to pandas
    import pandas as pd
    if columns:
        return pd.read_parquet(file_path, columns=columns)
    else:
        return pd.read_parquet(file_path)


class CommunityGraphService:
    """
    Service for creating community reports and performing global search.
    
    Combines GraphRAG extraction, clustering, and MinIO storage for
    efficient community-based search.
    """
    
    def __init__(
        self,
        graph_service: Optional[GraphService] = None,
        graph_rag_service: Optional[GraphRAGService] = None,
        minio_storage_service: Optional[MinIOStorageService] = None,
        llm: Optional[Any] = None
    ):
        """
        Initialize Community Graph service. 
        
        Args:
            graph_service: GraphService instance for Neo4j operations
            graph_rag_service: GraphRAGService instance for extraction
            minio_storage_service: MinIOStorageService instance for artifact storage
        """
        self.graph_service = graph_service
        self.graph_rag_service = graph_rag_service
        self.minio_storage_service = minio_storage_service
        self.llm = llm
        self.db = duckdb.connect(database=':memory:')
        self._setup_duckdb()
        # Initialize MinIO storage if not provided
        if self.minio_storage_service is None:
            try:
                self.minio_storage_service = get_minio_storage_service()
            except Exception as e:
                logger.warning(f"MinIO storage service not available: {e}")
                self.minio_storage_service = None

    def _setup_duckdb(self):
        """Configura DuckDB per leggere da S3/MinIO se necessario."""
        self.db.execute("INSTALL httpfs; LOAD httpfs;")
        # Se hai MinIO, qui configureresti le credenziali S3
        # self.db.execute("SET s3_endpoint='...'" ) 

    async def create_community_graph(
        self,
        namespace: str,
        engine: Engine,
        vector_store_repo,
        llm=None,
        llm_embeddings: Optional[Any] = None,
        sparse_encoder: Union[str, TEIConfig, None] = None,
        limit: int = 100,
        index_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        overwrite: bool = False,
        import_to_neo4j: bool = True,
        save_to_minio: bool = True,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a knowledge graph with community reports from vector store chunks.
        """
        logger.info(f"Creating community graph from namespace: {namespace}, limit: {limit}")
        
        # Normalize sparse_encoder (empty string treated as None)
        if isinstance(sparse_encoder, str) and sparse_encoder == "":
            sparse_encoder = None
        
        # 1. Create temporary working directory if output_dir not provided
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix=f"community_graph_{namespace}_")
            output_dir = temp_dir
            cleanup_temp = True
        else:
            cleanup_temp = False
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        try:
            # 2. Import documents from vector store using GraphRAG service
            if self.graph_rag_service is None:
                raise RuntimeError("GraphRAGService not initialized")
            
            # Set vector store repository if needed
            if self.graph_rag_service.vector_store_repository is None:
                self.graph_rag_service.set_vector_store_repository(vector_store_repo)
            
            # Import documents (extracts entities/relationships and imports to Neo4j)
            import_stats = {}
            if import_to_neo4j:
                # Determine engine_type for metadata filtering
                engine_type = None
                if hasattr(engine, 'type') and engine.type is not None:
                    engine_type = engine.type
                elif hasattr(engine, 'deployment'):
                    engine_type = engine.deployment
                import_stats = await self.graph_rag_service.import_from_vector_store(
                    namespace=namespace,
                    vector_store_repo=vector_store_repo,
                    engine=engine,
                    limit=limit,
                    llm=llm,
                    index_name=index_name,
                    overwrite=overwrite,
                    engine_type=engine_type
                )
                logger.info(f"Imported {import_stats.get('nodes_created', 0)} nodes and "
                           f"{import_stats.get('relationships_created', 0)} relationships")
            else:
                logger.info("Skipping Neo4j import (import_to_neo4j=False)")
            
            # 3-7. Generate community reports and export (Hierarchical 0-2)
            cluster_stats = await self.generate_hierarchical_reports(
                namespace=namespace,
                index_name=index_name,
                sparse_encoder=sparse_encoder,
                output_dir=output_dir,
                save_to_minio=save_to_minio,
                timestamp=timestamp,
                llm=llm,
                vector_store_repo=vector_store_repo,
                llm_embeddings=llm_embeddings,
                engine=engine
            )
            
            # Combine stats
            result = {
                "namespace": namespace,
                "chunks_processed": limit,
                "nodes_created": import_stats.get("nodes_created", 0),
                "relationships_created": import_stats.get("relationships_created", 0),
                **cluster_stats,
                "status": "success",
                "cleanup_temp": cleanup_temp
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create community graph: {e}")
            if cleanup_temp:
                shutil.rmtree(output_dir, ignore_errors=True)
            raise

    async def generate_hierarchical_reports(
        self,
        namespace: str,
        index_name: Optional[str] = None,
        sparse_encoder: Union[str, TEIConfig, None] = None,
        output_dir: Optional[str] = None,
        save_to_minio: bool = True,
        timestamp: Optional[str] = None,
        llm=None,
        vector_store_repo=None,
        llm_embeddings=None,
        engine: Optional[Engine] = None,
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """
        Generate hierarchical community reports (Level 0, 1, 2) using Leiden clustering with varying resolution.
        Also, embeds and indexes the reports in the vector store for semantic search.

        Args:
            overwrite: If True, clears existing reports before generating new ones (default: True)
            :param overwrite:
            :param engine:
            :param llm_embeddings:
            :param vector_store_repo:
            :param llm:
            :param timestamp:
            :param save_to_minio:
            :param output_dir:
             :param sparse_encoder:
             :param index_name:
             :param namespace:
         """
        # Normalize sparse_encoder (empty string treated as None)
        if isinstance(sparse_encoder, str) and sparse_encoder == "":
            sparse_encoder = None
        
        # Clear existing reports if overwrite is enabled
        if overwrite and vector_store_repo and engine:
            await self._clear_existing_reports(namespace, vector_store_repo, engine)
        if self.graph_service is None:
            raise RuntimeError("GraphService not initialized")
        
        repo = self.graph_service._get_repository()
        llm_to_use = llm or (self.graph_rag_service.llm if self.graph_rag_service else None) or self.llm
        
        if llm_to_use is None:
             raise RuntimeError("LLM not provided")
             
        cluster_service = ClusterService(repository=repo, llm=llm_to_use)
        
        levels_config = { 0: 1.2, 1: 0.8, 2: 0.5 }
        
        all_reports = []
        total_communities = 0
        
        for level, resolution in levels_config.items():
            logger.info(f"Generating reports for Level {level} (Resolution {resolution})")
            stats = await cluster_service.perform_clustering_leiden(
                level=level,
                namespace=namespace,
                index_name=index_name,
                engine_name=engine.name,
                engine_type=engine.type, #if engine.name=="pinecone" else engine.deployment,
                resolution=resolution
            )
            if stats.get("reports"):
                all_reports.extend(stats["reports"])
                total_communities += stats.get("communities_detected", 0)

        # Index reports in vector store for semantic search with Synthetic QA
        print(f"all report: {len(all_reports)}, vector {vector_store_repo} embedding: {llm_embeddings}, engine: {engine}")
        if all_reports and vector_store_repo and llm_embeddings and engine:
            logger.info(f"Indexing {len(all_reports)} reports for semantic search...")
            try:
                # ENHANCEMENT: Enrich reports with synthetic questions
                enriched_reports = all_reports
                if llm_to_use:
                    try:
                        logger.info(f"Generating synthetic questions for {len(all_reports)} reports...")
                        enriched_reports = await enrich_reports_with_synthetic_qa(
                            reports=all_reports,
                            llm=llm_to_use,
                            num_questions_per_report=3  # Generate 3 questions per report
                        )
                        logger.info(f"Successfully enriched {len(enriched_reports)} reports with synthetic questions")
                    except Exception as e:
                        logger.warning(f"Failed to generate synthetic questions: {e}, continuing without them")
                        enriched_reports = all_reports

                # Create documents with enhanced content (including synthetic questions)
                report_docs = []
                for r in enriched_reports:
                    # Use format_report_with_questions_for_indexing to create optimized content
                    page_content = format_report_with_questions_for_indexing(r)

                    report_docs.append(
                        Document(
                            page_content=page_content,
                            metadata={
                                "doc_type": "community_report",
                                "level": r.get("level"),
                                "rating": r.get("rating"),
                                "community_id": r.get("community_id"),
                                "full_report": r.get("full_report", ""),
                                "title": r.get("title", ""),
                                "summary": r.get("summary", ""),
                                "synthetic_questions": r.get("synthetic_questions", [])  # Store questions
                            }
                        )
                    )

                # Use a dedicated namespace for reports
                report_namespace = f"{namespace}-reports"

                # Add documents to the vector store (using aadd_documents)
                await vector_store_repo.aadd_documents(
                    engine=engine,
                    documents=report_docs,
                    namespace=report_namespace,
                    sparse_encoder=sparse_encoder,
                    embedding_model=llm_embeddings
                )
                logger.info(f"Successfully indexed {len(report_docs)} enriched reports in namespace '{report_namespace}'")

            except Exception as e:
                logger.error(f"Failed to index community reports in vector store: {e}")

        combined_stats = {
            "communities_detected": total_communities,
            "reports_created": len(all_reports),
            "reports": all_reports
        }
        
        return await self._process_reports_and_export(
            combined_stats, namespace, index_name, output_dir, save_to_minio, timestamp, repo, engine
        )

    async def _clear_existing_reports(
        self,
        namespace: str,
        vector_store_repo=None,
        engine=None
    ):
        """
        Clear existing community reports from vector store.

        Args:
            namespace: Base namespace (reports are stored in {namespace}-reports)
            vector_store_repo: Vector store repository instance
            engine: Engine configuration
        """
        if not vector_store_repo or not engine:
            logger.warning("Cannot clear existing reports: vector_store_repo or engine not provided")
            return

        report_namespace = f"{namespace}-reports"
        logger.info(f"Clearing existing reports from namespace: {report_namespace}")

        namespace_to_delete = RepositoryNamespace(
            namespace=report_namespace,
            engine=engine
        )

        try:
            # Attempt to delete all documents from the report namespace
            # Different vector stores have different methods, so we try common approaches
            if hasattr(vector_store_repo, 'delete_namespace'):
                await vector_store_repo.delete_namespace(namespace_to_delete=namespace_to_delete)
                logger.info(f"Successfully deleted namespace: {report_namespace}")
            else:
                logger.warning(f"Vector store repository does not support namespace deletion. "
                             f"Old reports in '{report_namespace}' may remain.")
        except Exception as e:
            logger.warning(f"Failed to clear existing reports from '{report_namespace}': {e}. "
                         f"Continuing with report generation.")

    async def generate_community_reports(
        self,
        namespace: str,
        index_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_to_minio: bool = True,
        timestamp: Optional[str] = None,
        llm=None,
        vector_store_repo=None,
        llm_embeddings=None,
        engine=None,
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """
        Perform clustering, generate reports, and export to Parquet/MinIO.
        Can be called independently on an existing graph.

        Args:
            overwrite: If True, clears existing reports before generating new ones (default: True)
            :param overwrite:
            :param engine:
            :param llm_embeddings:
            :param vector_store_repo:
            :param llm:
            :param timestamp:
            :param save_to_minio:
            :param namespace:
            :param output_dir:
            :param index_name:
        """
        # Clear existing reports if overwrite is enabled
        if overwrite and vector_store_repo and engine:
            await self._clear_existing_reports(namespace, vector_store_repo, engine)
        # 3. Perform clustering to generate community reports
        if self.graph_service is None:
            raise RuntimeError("GraphService not initialized")
        
        # Get repository from graph service
        repo = self.graph_service._get_repository()
        if repo is None:
            raise RuntimeError("Graph repository not available")
        
        # Determine which LLM to use
        llm_to_use = llm or (self.graph_rag_service.llm if self.graph_rag_service else None) or self.llm
        if llm_to_use is None:
            raise RuntimeError("LLM not provided for community report generation")
        
        # Initialize cluster service
        cluster_service = ClusterService(repository=repo, llm=llm_to_use)
        
        # Perform clustering
        cluster_stats = await cluster_service.perform_clustering(
            level=0,  # Top-level communities
            namespace=namespace,
            index_name= index_name,
            engine_name=engine.name,
            engine_type=engine.type #if engine.name=="pinecone" else engine.deployment
        )
        
        return await self._process_reports_and_export(
            cluster_stats, namespace, index_name, output_dir, save_to_minio, timestamp, repo, None
        )

    async def generate_community_reports_leiden(
        self,
        namespace: str,
        index_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_to_minio: bool = True,
        timestamp: Optional[str] = None,
        llm=None,
        vector_store_repo=None,
        llm_embeddings=None,
        engine=None,
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """
        Perform Leiden clustering, generate reports, and export.

        Args:
            overwrite: If True, clears existing reports before generating new ones (default: True)
            :param overwrite:
            :param engine:
            :param llm_embeddings:
            :param vector_store_repo:
            :param llm:
            :param timestamp:
            :param save_to_minio:
            :param output_dir:
            :param index_name:
            :param namespace:
        """
        # Clear existing reports if overwrite is enabled
        if overwrite and vector_store_repo and engine:
            await self._clear_existing_reports(namespace, vector_store_repo, engine)
        if self.graph_service is None:
            raise RuntimeError("GraphService not initialized")
        
        repo = self.graph_service._get_repository()
        llm_to_use = llm or (self.graph_rag_service.llm if self.graph_rag_service else None) or self.llm
        
        if llm_to_use is None:
             raise RuntimeError("LLM not provided")
             
        cluster_service = ClusterService(repository=repo, llm=llm_to_use)
        
        # Use Leiden clustering
        cluster_stats = await cluster_service.perform_clustering_leiden(
            level=0,
            namespace=namespace,
            index_name=index_name,
            engine_name=engine.name,
            engine_type=engine.type #if engine.name =="pinecone" else engine.deployment

        )
        
        return await self._process_reports_and_export(
            cluster_stats, namespace, index_name, output_dir, save_to_minio, timestamp, repo, None
        )

    async def _process_reports_and_export(
        self, cluster_stats, namespace, index_name, output_dir, save_to_minio, timestamp, repo, engine=None
    ):
        """Helper to process reports, save parquet, upload MinIO"""
        created_temp_dir = False
        if output_dir is None:
             import tempfile
             output_dir = tempfile.mkdtemp(prefix=f"cr_{namespace}_")
             created_temp_dir = True

        try:
            logger.info(f"Clustering completed: {cluster_stats.get('communities_detected', 0)} communities, "
                        f"{cluster_stats.get('reports_created', 0)} reports generated")
            
            reports_data = cluster_stats.get("reports", [])
            
            parquet_files = {}
            if reports_data:
                import pandas as pd
                import json
                processed_reports = []
                for r in reports_data:
                    report_dict = {
                        "community_id": r.get("community_id"),
                        "level": r.get("level"),
                        "title": r.get("title", ""),
                        "summary": r.get("summary", ""),
                        "full_report": r.get("full_report", ""),
                        "rating": r.get("rating", 0.0),
                        "rating_explanation": r.get("rating_explanation", ""),
                        "findings": str(r.get("findings")),
                        "import_timestamp": r.get("timestamp"),
                        "namespace": namespace,
                        "index_name": index_name
                    }
                    
                    # Convert list fields to JSON strings for Parquet/DuckDB compatibility
                    if "entities" in r:
                        try:
                            report_dict["entities"] = json.dumps(r["entities"])
                        except Exception as ex:
                            report_dict["entities"] = "[]"
                            
                    processed_reports.append(report_dict)
                
                community_reports_df = pd.DataFrame(processed_reports)
                
                reports_path = Path(output_dir) / "community_reports.parquet"
                community_reports_df.to_parquet(reports_path, index=False)
                parquet_files["community_reports"] = str(reports_path)
            
            # Export entities and relationships
            entities_df = self._export_entities_to_dataframe(repo, namespace, index_name)
            if not entities_df.empty:
                entities_path = Path(output_dir) / "entities.parquet"
                entities_df.to_parquet(entities_path, index=False)
                parquet_files["entities"] = str(entities_path)
            
            relationships_df = self._export_relationships_to_dataframe(repo, namespace, index_name)
            if not relationships_df.empty:
                relationships_path = Path(output_dir) / "relationships.parquet"
                relationships_df.to_parquet(relationships_path, index=False)
                parquet_files["relationships"] = str(relationships_path)
            
            # 6. Optionally upload to MinIO
            minio_files = {}
            if save_to_minio and self.minio_storage_service and parquet_files:
                try:
                    # Determine index_name and index_type for MinIO storage structure
                    upload_index_name = index_name
                    upload_index_type = None
                    
                    if engine is not None:
                        # Use engine's index_name if not provided
                        if upload_index_name is None:
                            upload_index_name = engine.index_name
                        # Determine index_type based on engine type
                        # MODIFIED -
                        upload_index_type = engine.type
                        #if engine.name == "pinecone":
                        #    upload_index_type = engine.type  # "serverless" or "pod"
                        #elif engine.name == "qdrant":
                        #    upload_index_type = engine.deployment  # "local" or "cloud"
                        #else:
                        #    logger.warning(f"Unsupported engine name: {engine.name}, using legacy structure")
                    
                    for file_name, file_path in parquet_files.items():
                        object_key = self.minio_storage_service.upload_parquet_file(
                            namespace=namespace,
                            file_name=f"{file_name}.parquet",
                            file_path=file_path,
                            timestamp=timestamp,
                            index_name=upload_index_name,
                            index_type=upload_index_type
                        )
                        minio_files[file_name] = object_key
                except Exception as e:
                    logger.error(f"Failed to upload to MinIO: {e}")
                    minio_files = {"error": str(e)}
            
            return {
                "status": "success",
                "communities_detected": cluster_stats.get("communities_detected", 0),
                "reports_created": cluster_stats.get("reports_created", 0),
                "parquet_files": list(parquet_files.keys()),
                "minio_files": minio_files,
                "output_directory": output_dir
            }
        except Exception as e:
            logger.error(f"Error in _process_reports_and_export: {e}")
            if created_temp_dir and output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
            raise

    def _export_community_reports_to_dataframe(self, repository, namespace: Optional[str] = None, index_name: Optional[str] = None):
        """Export community reports from Neo4j to pandas DataFrame."""
        try:
            # Query community reports from Neo4j
            # This is a simplified query - adjust based on your actual schema
            query = """
            MATCH (c:CommunityReport)
            WHERE ($namespace IS NULL OR c.namespace = $namespace)
            AND ($index_name IS NULL OR c.index_name = $index_name)
            RETURN 
                elementId(c) as report_id,
                c.community_id as community_id,
                c.title as title,
                c.summary as summary,
                c.rating as rating,
                c.rating_explanation as rating_explanation,
                c.full_report as full_report,
                c.import_timestamp as import_timestamp,
                c.level as level
            """
            
            with repository._get_session() as session:
                result = session.run(query, namespace=namespace, index_name=index_name)
                records = [dict(record) for record in result]
            
            import pandas as pd
            df = pd.DataFrame(records)
            
            # Add namespace and index_name columns if not present
            if 'namespace' not in df.columns and namespace:
                df['namespace'] = namespace
            if 'index_name' not in df.columns and index_name:
                df['index_name'] = index_name
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to export community reports: {e}")
            import pandas as pd
            return pd.DataFrame()
    
    def _export_entities_to_dataframe(self, repository, namespace: Optional[str] = None, index_name: Optional[str] = None):
        """Export entities (nodes) from Neo4j to pandas DataFrame."""
        try:
            # Query nodes from Neo4j
            query = """
            MATCH (n)
            WHERE ($namespace IS NULL OR n.namespace = $namespace)
            AND ($index_name IS NULL OR n.index_name = $index_name)
            AND NOT n:CommunityReport  // Exclude community report nodes
            RETURN 
                elementId(n) as node_id,
                labels(n) as labels,
                properties(n) as properties
            """
            
            with repository._get_session() as session:
                result = session.run(query, namespace=namespace, index_name=index_name)
                records = []
                for record in result:
                    props = dict(record["properties"])
                    props["node_id"] = record["node_id"]
                    props["labels"] = list(record["labels"])
                    records.append(props)
            
            import pandas as pd
            df = pd.DataFrame(records)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to export entities: {e}")
            import pandas as pd
            return pd.DataFrame()
    
    def _export_relationships_to_dataframe(self, repository, namespace: Optional[str] = None, index_name: Optional[str] = None):
        """Export relationships from Neo4j to pandas DataFrame."""
        try:
            # Query relationships from Neo4j
            query = """
            MATCH (s)-[r]->(t)
            WHERE ($namespace IS NULL OR s.namespace = $namespace AND t.namespace = $namespace)
            AND ($index_name IS NULL OR s.index_name = $index_name AND t.index_name = $index_name)
            RETURN 
                elementId(r) as relationship_id,
                elementId(s) as source_id,
                elementId(t) as target_id,
                type(r) as relationship_type,
                properties(r) as properties
            """
            
            with repository._get_session() as session:
                result = session.run(query, namespace=namespace, index_name=index_name)
                records = []
                for record in result:
                    props = dict(record["properties"])
                    props["relationship_id"] = record["relationship_id"]
                    props["source_id"] = record["source_id"]
                    props["target_id"] = record["target_id"]
                    props["relationship_type"] = record["relationship_type"]
                    records.append(props)
            
            import pandas as pd
            df = pd.DataFrame(records)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to export relationships: {e}")
            import pandas as pd
            return pd.DataFrame()
    
    async def query_with_global_search_old(
        self,
        question: str,
        namespace: str,
        search_type: str = "community",
        llm=None,
        timestamp: Optional[str] = None,
        use_minio: bool = True,
        local_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform global search on community reports.
        
        Args:
            question: Question to answer
            namespace: Namespace used for community reports
            search_type: Type of search - "community" (search community reports),
                        "hybrid" (combine with vector search)
            llm: Optional LLM instance for answer generation (if None, uses simple text concatenation)
            timestamp: Optional timestamp for MinIO artifacts (default: 'latest')
            use_minio: Whether to load Parquet files from MinIO
            local_dir: Optional local directory containing Parquet files
            
        Returns:
            Dictionary with search results
        """
        logger.info(f"Global search query: {question}, namespace: {namespace}, search_type: {search_type}")
        
        # Determine where to load Parquet files from
        parquet_dir = local_dir
        downloaded_from_minio = False
        
        if parquet_dir is None and use_minio and self.minio_storage_service:
            # Try to download from MinIO
            try:
                temp_dir = tempfile.mkdtemp(prefix=f"global_search_{namespace}_")
                downloaded = self.minio_storage_service.download_graphrag_outputs(
                    namespace=namespace,
                    timestamp=timestamp or "latest",
                    local_dir=temp_dir
                )
                
                if downloaded:
                    parquet_dir = temp_dir
                    downloaded_from_minio = True
                    logger.info(f"Downloaded Parquet files from MinIO to {temp_dir}")
                else:
                    raise FileNotFoundError(f"No Parquet files found in MinIO for namespace {namespace}")
            except Exception as e:
                logger.warning(f"Failed to load from MinIO: {e}")
                if downloaded_from_minio and parquet_dir:
                    shutil.rmtree(parquet_dir, ignore_errors=True)
                raise
        
        if parquet_dir is None:
            raise FileNotFoundError(
                f"No Parquet files available for namespace {namespace}. "
                f"Please create community graph first."
            )
        
        try:
            # Load community reports Parquet file
            reports_path = Path(parquet_dir) / "community_reports.parquet"
            if not reports_path.exists():
                raise FileNotFoundError(f"Community reports file not found: {reports_path}")
            
            # Load efficiently using DuckDB if available
            reports_df = _load_parquet_efficiently(str(reports_path))
            logger.info(f"Loaded {len(reports_df)} community reports")
            
            # Simple keyword search in community reports
            # In a real implementation, you would use more sophisticated search
            # (e.g., embedding similarity, BM25, etc.)
            matching_reports = []
            for _, row in reports_df.iterrows():
                # Check if question keywords appear in report content
                content = f"{row.get('title', '')} {row.get('summary', '')} {row.get('full_report', '')}".lower()
                question_lower = question.lower()
                
                # Simple keyword matching
                if any(keyword in content for keyword in question_lower.split()[:5]):
                    matching_reports.append(dict(row))
            
            # Load entities for context
            entities_df = None
            entities_path = Path(parquet_dir) / "entities.parquet"
            if entities_path.exists():
                entities_df = _load_parquet_efficiently(str(entities_path))
                logger.info(f"Loaded {len(entities_df)} entities for context")
            
            # Generate answer based on matching reports
            answer = self._generate_answer_from_reports(question, matching_reports, entities_df, llm)
            
            result = {
                "answer": answer,
                "matching_reports": matching_reports,
                "total_reports_searched": len(reports_df),
                "reports_found": len(matching_reports),
                "search_type": search_type,
                "namespace": namespace,
                "data_source": "MinIO" if downloaded_from_minio else "local"
            }
            
            return result
            
        finally:
            # Clean up temporary directory if we downloaded from MinIO
            if downloaded_from_minio and parquet_dir:
                shutil.rmtree(parquet_dir, ignore_errors=True)

    async def query_with_global_search(
            self,
            question: str,
            namespace: str,
            sparse_encoder: Union[str, TEIConfig, None] = 'splade',
            user_language: str = "the same language as the user's question",
            llm=None,
            vector_store_repo=None,
            llm_embeddings=None,
            engine: Optional[Engine] = None,
            chat_history_dict: Optional[Dict[str, Any]] = None,
            reranking_config: Optional[Union[str, TEIConfig, PineconeRerankerConfig]] = None,
            use_rrf: bool = True,
            use_reranking: bool = True,
            top_k_initial: int = 20,
            top_k_reranked: int = 5
    ) -> Dict[str, Any]:
        """
        Global Search with Semantic Retrieval on Community Reports.

        Enhanced with:
        - RRF (Reciprocal Rank Fusion) for combining dense+sparse results
        - Cross-Encoder Reranking before MAP-REDUCE
        - Query Type Detection for optimized retrieval

        Args:
            question: User's question
            namespace: Namespace for reports
            sparse_encoder: Sparse encoder configuration
            user_language: Language for response
            llm: LLM for answer generation
            vector_store_repo: Vector store repository
            llm_embeddings: Embedding model
            engine: Engine configuration
            chat_history_dict: Chat history
            reranking_config: Configuration for reranker
            use_rrf: Whether to use RRF for combining results (default: True)
            use_reranking: Whether to use cross-encoder reranking (default: True)
            top_k_initial: Number of reports to retrieve before reranking (default: 20)
            top_k_reranked: Number of reports to use after reranking (default: 5)
        """
        logger.info(f"Enhanced Global Search for: {question} in namespace: {namespace}")
        logger.info(f"RRF: {use_rrf}, Reranking: {use_reranking}, Initial: {top_k_initial}, Final: {top_k_reranked}")

        # Validate dependencies
        llm_to_use = llm or self.llm
        if not llm_to_use:
             raise ValueError("LLM is required for global search")
        if not all([vector_store_repo, llm_embeddings, engine]):
            raise ValueError("Vector store, embeddings, and engine are required for semantic search.")

        try:
            # 0. QUERY TYPE DETECTION (optional enhancement)
            query_type = None
            try:
                query_type = await detect_query_type_with_llm(question, llm_to_use, fallback_to_heuristic=True)
                logger.info(f"Detected query type: {query_type}")

                # Adjust weights based on query type
                weight_adjustments = get_weight_adjustments(query_type)
                logger.info(f"Weight adjustments for {query_type}: {weight_adjustments}")
            except Exception as e:
                logger.warning(f"Query type detection failed: {e}, continuing without adjustments")

            # 1. HYBRID SEMANTIC RETRIEVAL of reports from Vector Store
            report_namespace = f"{namespace}-reports"
            logger.info(f"Performing hybrid semantic search in report namespace: '{report_namespace}'")

            # Build QuestionAnswer object for hybrid search
            qa_for_reports = QuestionAnswer(
                question=question,
                namespace=report_namespace,
                sparse_encoder = sparse_encoder,
                engine=engine,
                search_type="hybrid", # We want hybrid search for reports
                top_k=top_k_initial, # Retrieve more initially for RRF/reranking
                llm="openai", model="gpt-4", temperature=0.0 # Placeholders
            )

            emb_dimension, sparse_encoder, index = await vector_store_repo.initialize_embeddings_and_index(qa_for_reports,
                                                                                              llm_embeddings)
            # Validate sparse encoder if present
            if sparse_encoder is not None and not hasattr(sparse_encoder, 'encode_queries'):
                logger.error(f"sparse_encoder is not a proper encoder instance: {type(sparse_encoder)}")
                raise ValueError(f"Invalid sparse encoder returned: {type(sparse_encoder)}")

            # Fetch dense and sparse vectors for the question
            dense_vector, sparse_vector = await fetch_question_vectors(qa_for_reports, sparse_encoder, llm_embeddings)

            # Perform hybrid search (returns combined results)
            search_results = await vector_store_repo.search_community_report(
                qa_for_reports, index, dense_vector, sparse_vector
            )

            if not search_results.get('matches'):
                return {"answer": f"I could not find any semantically relevant community reports for '{question}'.", "status": "empty"}

            # Convert to Document objects for reranking
            report_documents = []
            for match in search_results['matches']:
                metadata = match.get('metadata', {}) if isinstance(match, dict) else {}

                # Create Document for reranking
                doc = Document(
                    page_content=metadata.get("full_report", ""),
                    metadata={
                        "title": metadata.get("title", ""),
                        "summary": metadata.get("summary", ""),
                        "full_report": metadata.get("full_report", ""),
                        "level": metadata.get("level", 0),
                        "rating": metadata.get("rating", 0.0),
                        "score": match.get("score", 0.0),  # Original hybrid score
                        "id": metadata.get("id", match.get("id", ""))
                    }
                )
                report_documents.append(doc)

            logger.info(f"Retrieved {len(report_documents)} reports from hybrid search")

            # 2. CROSS-ENCODER RERANKING (if enabled)
            if use_reranking and reranking_config:
                try:
                    logger.info(f"Reranking {len(report_documents)} reports with cross-encoder...")
                    reranker = TileReranker(reranking_config)

                    # Rerank documents using async method (runs in thread pool)
                    reranked_docs = await reranker.arerank_documents(
                        query=question,
                        documents=report_documents,
                        top_k=top_k_reranked,
                        batch_size=8
                    )

                    logger.info(f"Reranking complete: {len(reranked_docs)} top reports selected")
                    report_documents = reranked_docs

                except Exception as e:
                    logger.error(f"Reranking failed: {e}, using original hybrid results")
                    # Fall back to top_k_reranked from hybrid results
                    report_documents = report_documents[:top_k_reranked]
            else:
                # No reranking, just take top_k
                report_documents = report_documents[:top_k_reranked]
                logger.info(f"Reranking disabled, using top {top_k_reranked} hybrid results")

            # Convert to report tuples for MAP-REDUCE
            relevant_reports = []
            for doc in report_documents:
                relevant_reports.append((
                    doc.metadata.get("title", ""),
                    doc.metadata.get("summary", ""),
                    doc.metadata.get("full_report", ""),
                    doc.metadata.get("level", 0),
                    doc.metadata.get("rating", 0.0)
                ))

            logger.info(f"Processing {len(relevant_reports)} final reports for MAP-REDUCE")

            # 3. MAP PHASE: Analyze reports in parallel (only on reranked top-k)
            tasks = [self._map_report_to_answer(question, r, user_language, llm_to_use) for r in relevant_reports]
            partial_answers = await asyncio.gather(*tasks)

            # 4. REDUCE PHASE: Synthesize final answer
            final_answer = await self._reduce_answers(question, partial_answers, user_language, llm_to_use, chat_history_dict)

            # Update Chat History
            updated_history = chat_history_dict or {}
            next_key = str(len(updated_history))
            updated_history[next_key] = ChatEntry(question=question, answer=final_answer)

            return {
                "answer": final_answer,
                "reports_used": len(relevant_reports),
                "total_reports_found": len(search_results.get('matches', [])),
                "reports_after_reranking": len(report_documents),
                "language": user_language,
                "query_type": query_type,
                "used_rrf": use_rrf,
                "used_reranking": use_reranking,
                "chat_history_dict": updated_history
            }

        except Exception as e:
            logger.error(f"Enhanced global search error: {e}", exc_info=True)
            return {"answer": f"An error occurred during enhanced global search: {str(e)}", "status": "error"}

    async def _map_report_to_answer(self, question, report, lang, llm):
        """Analyze a single report to extract relevant info."""
        title, summary, full_text, level, rating = report
        
        # Use simple invoke #Formulate the answer in {lang}.
        prompt = f"""
        Analyze the following community report (Level {level}, Rating {rating}) and extract information relevant to the user's question.

        USER QUESTION: "{question}"

        COMMUNITY REPORT CONTENT:
        Title: {title}
        Summary: {summary}
        Full Details: {full_text}

        INSTRUCTIONS:
        1. Based *only* on the report content provided, extract the information that directly answers the user's question.
        2. Formulate the answer in {lang}.
        3. If the report does not contain any relevant information to answer the question, respond with the exact phrase 'NOT_RELEVANT'.
        """
        try:
            if hasattr(llm, 'ainvoke'):
                response = await llm.ainvoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                # Synchronous fallback (should be avoided in async loop but handled)
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            
            return content
        except Exception as e:
            logger.warning(f"Error in map phase: {e}")
            return "NOT_RELEVANT"

    async def _reduce_answers(self, question, answers, lang, llm, chat_history_dict=None):
        """Combine partial answers into final response."""
        print(answers)
        filtered_answers = [a for a in answers if a and "NOT_RELEVANT" not in a and len(a) > 10]
        
        if not filtered_answers:
            # If no relevant context found, ask LLM to generate a polite "not found" message
            # in the requested language.
            not_found_prompt = f"""
                        The user asked the following question: "{question}"
                        No relevant information was found in the knowledge base.
                        Please write a polite, concise message in {lang} to inform the user that you could not find a specific answer.
                        """
            try:
                if hasattr(llm, 'ainvoke'):
                    response = await llm.ainvoke(not_found_prompt)
                    return response.content if hasattr(response, 'content') else str(response)
                else:
                    response = llm.invoke(not_found_prompt)
                    return response.content if hasattr(response, 'content') else str(response)
            except Exception:
                # Fallback in case of error
                return "I couldn't find sufficient relevant information in the community reports to answer your question."
            
        context = "\n---\n".join(filtered_answers)

        # Format Chat History
        history_text = format_chat_history(chat_history_dict)

        # in {lang}
        prompt = f"""
        You are an expert assistant. Synthesize the following analysis fragments to answer the user's question, taking into account the conversation history.

        QUESTION: {question}

        CHAT HISTORY:
        {history_text}
        
        ANALYSIS CONTEXT:
        {context}

        FINAL ANSWER:
        Generate a structured, professional, and comprehensive answer.
        If the question was in English, answer in English. If it was in Italian, answer in Italian. If it was in French, answer in French. If it was in Spanish, answer in Spanish, and so on, regardless of the context language.
        Cite the information source implicitly by synthesizing the facts.
        """
        
        try:
            if hasattr(llm, 'ainvoke'):
                response = await llm.ainvoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            return content
        except Exception as e:
            logger.error(f"Error in reduce phase: {e}")
            return "Error generating final summary."

    async def _search_relevant_reports(self, question: str, namespace: str, limit: int = 10) -> List[Any]:
        """
        Helper to search relevant reports using DuckDB.
        Handles downloading from MinIO if needed.
        """
        temp_dir = None
        reports_path = None
        
        try:
            # Check if we have MinIO service and try download
            if self.minio_storage_service:
                temp_dir = tempfile.mkdtemp(prefix=f"gs_{namespace}_")
                local_path = Path(temp_dir) / "community_reports.parquet"
                try:
                    downloaded_path = self.minio_storage_service.download_parquet_file(
                        namespace=namespace,
                        file_name="community_reports.parquet",
                        local_path=str(local_path)
                    )
                    reports_path = downloaded_path
                except Exception as e:
                    logger.warning(f"Could not download reports from MinIO: {e}")
            
            if not reports_path or not os.path.exists(reports_path):
                 # Fallback logic or return empty
                 if temp_dir: shutil.rmtree(temp_dir, ignore_errors=True)
                 return []

            # DuckDB Query
            safe_question = question.replace("'", "''")
            keywords = [w for w in safe_question.split() if len(w) > 3]
            if not keywords: keywords = [safe_question]

            conditions = []
            params = [reports_path]
            keyword_conditions = []
            for kw in keywords[:5]:
                keyword_conditions.append("(title ILIKE ? OR summary ILIKE ?)")
                params.extend([f"%{kw}%", f"%{kw}%"])
            
            where_clause = " OR ".join(keyword_conditions) if keyword_conditions else "1=1"
            
            search_query = f"""
                SELECT title, summary, full_report, level, rating
                FROM read_parquet(?) 
                WHERE {where_clause}
                ORDER BY rating DESC, level ASC
                LIMIT {limit}
            """
            
            conn = duckdb.connect(database=':memory:')
            results = conn.execute(search_query, params).fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching reports: {e}")
            return []
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def context_fusion_search(
        self,
        question: str,
        namespace: str,
        search_type: str,
        sparse_encoder_injected: Optional[Union[str, TEIConfig, None]] ,
        reranking_injected: Optional[Union[str, TEIConfig, PineconeRerankerConfig]],
        engine: Engine,
        vector_store_repo,
        llm,
        llm_embeddings,
        user_language: str = "the same language as the user's question",
        query_type: Optional[str] = None,
        max_results: int = 15,
        chat_history_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Context Fusion Search: The Ultimate Hybrid Method.
        
        Pipeline:
        1. Query Analysis (Type Detection & Weighting)
        2. Parallel Execution:
           - Global Search (Semantic Search on Community Reports)
           - Local Search (Hybrid Vector + Keyword Search)
        3. Graph Expansion (from Local Search seeds)
        4. Cross-Encoder Reranking (Global + Local + Graph)
        5. LLM Synthesis
        """
        logger.info(f"Starting Context Fusion Search (Ultimate Hybrid) for: {question}")
        
        # 0. Dependencies Check
        if not self.graph_rag_service:
            raise RuntimeError("GraphRAGService required for query analysis and expansion.")
        
        # 1. Query Analysis
        if not query_type:
            query_type = self.graph_rag_service._detect_query_type(question)
        
        # Get Weights from Matrix of Balancing
        weights = self.graph_rag_service._adjust_weights_by_query_type(
            vector_weight=1.0, keyword_weight=1.0, graph_weight=1.0, query_type=query_type
        )
        logger.info(f"Query Type: {query_type}, Weights: {weights}")

        # 2. Pre-initialization: Create shared resources to prevent duplicate Pinecone sessions
        # Get embedding dimension
        embedding_model_name = getattr(llm_embeddings, 'model_name', None) or "text-embedding-ada-002"
        emb_dimension = await vector_store_repo.get_embeddings_dimension(embedding_model_name)
        
        # Create sparse encoder if needed (based on repository type and search_type)
        sparse_encoder_obj = None
        if hasattr(vector_store_repo, 'sparse_enabled') and vector_store_repo.sparse_enabled and search_type == "hybrid" and sparse_encoder_injected is not None:
            from tilellm.tools.sparse_encoders import TiledeskSparseEncoders
            sparse_encoder_obj = TiledeskSparseEncoders(sparse_encoder_injected)
        
        # Create shared CachedVectorStore wrapper with a unique cache suffix
        # This ensures both tasks use the same PineconeAsyncio client
        cached_wrapper = None
        cache_suffix = f"context_fusion_{namespace}"
        
        if hasattr(vector_store_repo, 'create_index_cache_wrapper'):
            cached_wrapper = await vector_store_repo.create_index_cache_wrapper(
                engine, llm_embeddings, emb_dimension, embedding_config_key=None, cache_suffix=cache_suffix
            )
        else:
            # Fallback for non-Pinecone repositories or older versions
            # Create a dummy QuestionAnswer for initialization
            qa_dummy = QuestionAnswer(
                question=question,
                namespace=namespace,
                engine=engine,
                search_type=search_type,
                embedding=embedding_model_name,
                sparse_encoder=sparse_encoder_injected
            )
            # We'll still initialize embeddings and index but can't share wrapper
            # Tasks will call initialize_embeddings_and_index separately
            # This may still create duplicate sessions but is a fallback
            pass
        
        # Keep reference to wrapper to prevent premature closure
        # Each task will get its own index from the wrapper
        
        # Task A: Global Search (if weight > 0 or exploratory)
        async def run_global_retrieval():
            if weights["graph"] < 0.2 and query_type == "technical":
                return []
            
            index = None
            try:
                report_namespace = f"{namespace}-reports"
                qa_reports = QuestionAnswer(
                    question=question,
                    namespace=report_namespace,
                    engine=engine,
                    search_type=search_type,
                    sparse_encoder=sparse_encoder_injected,
                    top_k=5, # Top 5 reports
                    llm="openai", model="gpt-4"
                )

                if cached_wrapper is not None:
                    # Use shared wrapper (each task gets its own index object)
                    sparse_enc = sparse_encoder_obj
                    index = await cached_wrapper.get_index()
                else:
                    # Fallback: initialize embeddings and index separately
                    _, sparse_enc, index = await vector_store_repo.initialize_embeddings_and_index(
                        qa_reports, llm_embeddings, cache_suffix=cache_suffix
                    )
                
                if qa_reports.search_type == "hybrid":
                    dense, sparse = await fetch_question_vectors(qa_reports, sparse_enc, llm_embeddings)
                else:
                    dense = await llm_embeddings.aembed_query(qa_reports.question)
                    sparse = None

                results = await vector_store_repo.search_community_report(qa_reports, index, dense, sparse)

                reports = []

                if results and results.get('matches'):
                    for m in results['matches']:
                        meta = m.get('metadata', {})
                        reports.append({
                            "title": meta.get("title", ""),
                            "summary": meta.get("summary", ""),
                            "content": meta.get("full_report", "") or meta.get("page_content", ""),
                            "rating": meta.get("rating", 0)
                        })
                return reports
            except Exception as e:
                logger.warning(f"Global retrieval failed: {e}")
                return []


        # Task B: Local Search (Vector + Keyword)
        async def run_local_retrieval():
            index = None
            try:
                # Calculate alpha for RRF
                total_w = weights["vector"] + weights["keyword"]
                alpha = weights["vector"] / total_w if total_w > 0 else 0.5
                
                qa_local = QuestionAnswer(
                    question=question,
                    namespace=namespace,
                    engine=engine,
                    top_k=max_results,
                    alpha=alpha,
                    search_type=search_type,
                    llm="openai", model="gpt-3.5-turbo",
                    embedding=llm_embeddings.model_name if hasattr(llm_embeddings, 'model_name') else "text-embedding-ada-002",
                    sparse_encoder=sparse_encoder_injected,
                    reranking=False # We rerank later globally
                )

                if cached_wrapper is not None:
                    # Use shared wrapper (each task gets its own index object)
                    sparse_enc = sparse_encoder_obj
                    index = await cached_wrapper.get_index()
                else:
                    # Fallback: initialize embeddings and index separately
                    _, sparse_enc, index = await vector_store_repo.initialize_embeddings_and_index(
                        qa_local, llm_embeddings, cache_suffix=cache_suffix
                    )
                
                if qa_local.search_type == "hybrid":
                    dense, sparse = await fetch_question_vectors(qa_local, sparse_enc, llm_embeddings)
                else:
                    logger.debug(f"search type: {qa_local.search_type }")
                    dense = await llm_embeddings.aembed_query(qa_local.question)
                    sparse = None

                results = await vector_store_repo.perform_hybrid_search(qa_local, index, dense, sparse)

                chunks = []
                if results and results.get('matches'):
                    for m in results['matches']:
                        chunks.append({
                            "id": m.get("id"),
                            "text": m.get("metadata", {}).get("text", "") or m.get("page_content", ""),
                            "score": m.get("score", 0)
                        })
                return chunks
            except Exception as e:
                logger.error(f"Local retrieval failed: {e}")
                return []


        # Execute Parallel
        global_task = asyncio.create_task(run_global_retrieval())
        local_task = asyncio.create_task(run_local_retrieval())
        
        global_reports, local_chunks = await asyncio.gather(global_task, local_task)
        
        # 3. Graph Expansion with Adaptive Multihop Strategy
        graph_nodes = []
        graph_rels = []

        if local_chunks and weights["graph"] > 0.3:
            try:
                chunk_ids = [c["id"] for c in local_chunks[:10]] # Expand from top 10 chunks
                graph_repo = self.graph_service._get_repository()

                # Find seed nodes
                seed_ids = []
                for cid in chunk_ids:
                    nodes = graph_repo.find_nodes_by_source_id(cid, limit=2, namespace=namespace, index_name=engine.index_name)
                    seed_ids.extend([n.id for n in nodes])

                seed_ids = list(set(seed_ids))

                if seed_ids:
                    # Use adaptive GraphExpander with dynamic hop count based on query_type
                    graph_expander = GraphExpander(repository=graph_repo)

                    # Scale expansion limit based on graph weight
                    expansion_limit = int(30 * weights["graph"])  # Dynamic limit based on graph weight

                    expansion_result = await graph_expander.expand_from_seeds(
                        seed_node_ids=seed_ids,
                        query_type=query_type,  # Determines max hops: technical=1, exploratory=2, relational=3
                        max_nodes=expansion_limit,
                        namespace=namespace,
                        index_name=engine.index_name,
                        min_relationship_weight=0.0
                    )

                    # Convert to expected format
                    graph_nodes = [
                        {
                            "id": node.id,
                            "label": node.label,
                            "properties": node.properties
                        } for node in expansion_result["nodes"]
                    ]

                    graph_rels = [
                        {
                            "id": rel.id,
                            "type": rel.type,
                            "source_id": rel.source_id,
                            "target_id": rel.target_id,
                            "properties": rel.properties
                        } for rel in expansion_result["relationships"]
                    ]

                    logger.info(f"Adaptive graph expansion complete: {len(graph_nodes)} nodes, "
                               f"{len(graph_rels)} relationships in {expansion_result['hops_executed']} hops "
                               f"(query_type: {query_type})")
            except Exception as e:
                logger.warning(f"Graph expansion failed: {e}")

        # 4. Context Preparation & Reranking
        all_docs = []
        
        # Add Reports
        for r in global_reports:
            txt = f"[Global Report] {r['title']}: {r['summary']}"
            all_docs.append(Document(page_content=txt, metadata={"type": "global", "rating": r['rating']}))
            
        # Add Chunks
        for c in local_chunks:
            txt = f"[Document] {c['text']}"
            all_docs.append(Document(page_content=txt, metadata={"type": "local", "id": c['id']}))
            
        # Add Graph Info
        for n in graph_nodes:
            props = n.get("properties", {})
            name = props.get("name", "Unknown")
            desc = props.get("description", "")
            txt = f"[Graph Entity] {name} ({n.get('label')}): {desc}"
            all_docs.append(Document(page_content=txt, metadata={"type": "graph", "id": n.get("id")}))

        # Rerank
        final_context = []
        if all_docs:
            try:
                from tilellm.tools.reranker import TileReranker
                reranker = TileReranker(model_name=reranking_injected)
                reranked = reranker.rerank_documents(question, all_docs, top_k=20) # Top 20 relevant pieces
                final_context = [d.page_content for d in reranked]
            except ImportError:
                logger.warning("TileReranker not available. Using top results without reranking.")
                final_context = [d.page_content for d in all_docs[:20]]
            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using raw results.")
                final_context = [d.page_content for d in all_docs[:20]]

        # 5. Synthesis
        context_str = "\n\n".join(final_context)
        
        # Format Chat History using utility
        history_text = format_chat_history(chat_history_dict)
        
        prompt = f"""
        You are an intelligent assistant using a Hybrid RAG system (Global Reports + Local Docs + Knowledge Graph).
        
        QUESTION: {question}
        
        CHAT HISTORY:
        {history_text}
        
        RETRIEVED CONTEXT:
        {context_str}
        
        INSTRUCTIONS:
        1. Answer the question comprehensively using the provided context and chat history.
        2. If Global Reports are present, use them for high-level summary.
        3. If Local Documents/Graph Entities are present, use them for specific details.
        4. Answer in {user_language}.
        5. If no relevant information is found, admit it politely.
        """
        
        answer = ""
        try:
            if hasattr(llm, 'ainvoke'):
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=prompt)]
                response = await llm.ainvoke(messages)
                answer = response.content if hasattr(response, 'content') else str(response)
            else:
                response = await llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            answer = "I'm sorry, I encountered an error generating the answer."

        # Update Chat History
        updated_history = chat_history_dict or {}
        next_key = str(len(updated_history))
        updated_history[next_key] = ChatEntry(question=question, answer=answer)
        return {
            "answer": answer,
            "entities": [{"id": n.get("id"), "label": n.get("label"), "properties": n.get("properties")} for n in graph_nodes],
            "relationships": [
                {
                    "id": r.get("id"), 
                    "type": r.get("type"), 
                    "properties": r.get("properties"), 
                    "source_id": r.get("source_id"), 
                    "target_id": r.get("target_id")
                } for r in graph_rels
            ],
            "retrieval_strategy": f"integrated_hybrid_{query_type}",
            "scores": {
                "global_reports": len(global_reports),
                "local_chunks": len(local_chunks),
                "graph_nodes": len(graph_nodes),
                "query_type": query_type
            },
            "expanded_nodes": graph_nodes,
            "expanded_relationships": graph_rels,
            "chat_history_dict": updated_history
        }


    def export_to_parquet(self, data: List[Dict], output_path: str):
        """Esporta dati in Parquet usando PyArrow senza passare da Pandas."""
        # Trasforma la lista di dizionari in una Tabella Arrow
        table = pa.Table.from_pylist(data)
        pq.write_table(table, output_path)

        logger.info(f"File salvato con successo in: {output_path}")

    def _generate_answer_from_reports(self, question: str, matching_reports: List[Dict], entities_df=None, llm=None) -> str:
        """Generate answer from matching community reports."""
        if not matching_reports:
            return f"I couldn't find specific information about '{question}' in the community reports."
        
        # Try to use LLM if provided
        if llm is not None:
            try:
                # Build context from reports
                context_parts = []
                for i, report in enumerate(matching_reports[:5]):  # Limit to 5 reports for token limits
                    title = report.get('title', 'Untitled Report')
                    summary = report.get('summary', '')
                    rating = report.get('rating', 0.0)
                    context_parts.append(f"Report {i+1}: {title} (rating: {rating}/5)\nSummary: {summary}")
                
                context = "\n\n".join(context_parts)
                prompt = f"""Question: {question}

Context from community reports:
{context}

Based on the above community reports, please answer the question. If the reports don't contain relevant information, state that clearly."""
                
                # Use LLM (assuming it has invoke method)
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                logger.info("Generated answer using LLM")
                return answer
            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}. Falling back to simple answer.")
        
        # Simple answer generation (fallback)
        report_summaries = []
        for i, report in enumerate(matching_reports[:3]):  # Limit to top 3
            title = report.get('title', 'Untitled Report')
            summary = report.get('summary', '')
            rating = report.get('rating', 0.0)
            
            summary_text = f"Report '{title}' (rating: {rating}/5): {summary}"
            if len(summary) > 200:
                summary_text = summary_text[:200] + "..."
            
            report_summaries.append(summary_text)
        
        # Combine into answer
        answer = f"Based on {len(matching_reports)} community reports related to '{question}':\n\n"
        answer += "\n".join([f"- {summary}" for summary in report_summaries])
        
        if len(matching_reports) > 3:
            answer += f"\n\n... and {len(matching_reports) - 3} more reports."
        
        return answer
    



# Singleton instance
_community_graph_service: Optional[CommunityGraphService] = None

def get_community_graph_service(
    graph_service: Optional[GraphService] = None,
    graph_rag_service: Optional[GraphRAGService] = None,
    minio_storage_service: Optional[MinIOStorageService] = None
) -> CommunityGraphService:
    """Get or create a singleton instance of CommunityGraphService."""
    global _community_graph_service
    if _community_graph_service is None:
        _community_graph_service = CommunityGraphService(
            graph_service=graph_service,
            graph_rag_service=graph_rag_service,
            minio_storage_service=minio_storage_service
        )
    return _community_graph_service
