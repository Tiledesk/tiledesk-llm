"""
Multimodal PDF Search Service
Implements hybrid search combining text, tables, and images.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from tilellm.models import Engine
from tilellm.modules.knowledge_graph.services.community_graph_service import CommunityGraphService
from tilellm.modules.knowledge_graph.services.table_qa_service import TableQAService

logger = logging.getLogger(__name__)

class MultimodalPDFSearch:
    """
    Multimodal search service that fuses results from text, tables, and images.
    """

    def __init__(
        self,
        community_service: CommunityGraphService,
        llm: Any = None
    ):
        self.community_service = community_service
        self.llm = llm
        self.table_qa_service = TableQAService(llm=llm)

    async def search(
        self,
        query: str,
        namespace: str,
        engine: Engine,
        vector_store_repo: Any,
        llm: Any,
        llm_embeddings: Any,
        search_types: List[str] = ["text", "table", "image"],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Execute multimodal search.
        """
        logger.info(f"Multimodal search for '{query}' in {namespace} types={search_types}")
        
        # 1. Text Search (Hybrid + Graph) - always run as it provides context
        text_task = self.community_service.context_fusion_search(
            question=query,
            namespace=namespace,
            search_type="hybrid",
            sparse_encoder_injected=None,
            reranking_injected=None,
            engine=engine,
            vector_store_repo=vector_store_repo,
            llm=llm,
            llm_embeddings=llm_embeddings
        )
        
        # Parallel execution of other types if we had independent searchers
        # For now, we rely on text search to find table references, 
        # then we'll do deep table QA in the synthesis phase.
        
        text_result = await text_task
        
        # 2. Extract potential tables from text result
        # Text chunks might have metadata['related_tables']
        candidate_table_ids = set()
        
        # Check source documents from local retrieval (if exposed in text_result)
        # Currently text_result returns 'answer', 'entities', 'scores'. 
        # We might need to look at 'sources' if context_fusion_search returns them.
        # It currently returns 'expanded_nodes'.
        
        # Let's try to fetch table nodes from the expanded graph nodes
        # If text search expanded to Table nodes, we can use them.
        expanded_nodes = text_result.get("expanded_nodes", [])
        for node in expanded_nodes:
            if node.get("label") == "Table":
                candidate_table_ids.add(node.get("id"))
                
        # Also check if the answer mentions looking at a table (heuristic)
        
        # 3. Table QA (if requested and candidates found)
        table_qa_result = None
        if "table" in search_types and candidate_table_ids:
            logger.info(f"Found candidate tables for QA: {candidate_table_ids}")
            tables_data = await self._fetch_tables_data(list(candidate_table_ids))
            if tables_data:
                table_qa_result = await self.table_qa_service.answer_query(query, tables_data)

        # 4. Image Search (Placeholder/Future)
        image_results = []

        # 5. Final Synthesis
        final_response = await self._fuse_and_synthesize(query, text_result, table_qa_result, image_results, llm)
        
        return final_response

    async def _fetch_tables_data(self, table_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch table metadata from Graph."""
        repo = self.community_service.graph_service._get_repository()
        if not repo:
            return []
            
        tables = []
        for tid in table_ids:
            node = repo.get_node(tid)
            if node:
                tables.append({
                    "id": node.id,
                    "parquet_path": node.properties.get("parquet_path"),
                    "columns": node.properties.get("columns", []),
                    "description": node.properties.get("description", "")
                })
        return tables

    async def _fuse_and_synthesize(
        self, 
        query: str, 
        text_result: Dict[str, Any], 
        table_result: Optional[Dict[str, Any]], 
        image_results: List[Any],
        llm: Any
    ) -> Dict[str, Any]:
        """
        Fuse results and generate a multimodal answer.
        """
        text_answer = text_result.get("answer", "")
        
        final_answer = text_answer
        sources = {
            "text": text_result,
            "tables": [],
            "images": image_results
        }
        
        if table_result and table_result.get("answer"):
            sources["tables"] = [table_result]
            
            # Combine answers
            prompt = f"""Combine the following information to answer the user's question: "{query}""

Text Analysis:
{text_answer}

Table Data Analysis:
{table_result['answer']}
(Derived from SQL: {table_result.get('sql')})

Provide a unified, comprehensive answer.
"""
            try:
                response = await llm.ainvoke(prompt)
                final_answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"Fusion failed: {e}")
                final_answer = f"{text_answer}\n\nAdditionally, from table analysis: {table_result['answer']}"
        
        return {
            "answer": final_answer,
            "sources": sources
        }