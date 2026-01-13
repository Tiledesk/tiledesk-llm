"""
Table Semantic Linker
Links tables to text descriptions and generates semantic metadata/questions.
"""

import logging
import json
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class TableSemanticLinker:
    """
    Links tables to their context in the document and generates
    enhanced semantic metadata.
    """

    def __init__(self, graph_repository: Optional[Any] = None):
        self.graph_repository = graph_repository

    async def link_table_to_context(
        self,
        table_data: Dict[str, Any],
        text_elements: List[Dict[str, Any]],
        llm: Any,
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Enhance table metadata by linking to surrounding text and generating questions.
        """
        table_id = table_data.get('id')
        page = table_data.get('page', 0)
        df = table_data.get('data')
        
        if df is None or df.empty:
            return table_data

        # 1. Find text referencing this table
        referencing_elements = self._find_references(table_id, text_elements)
        ref_texts = [e.get('text', '') for e in referencing_elements]
        context_text = " ".join(ref_texts)
        
        # 2. Generate questions the table can answer
        questions = await self._generate_table_questions(df, context_text, llm)
        table_data['answerable_questions'] = questions
        
        # 3. Create semantic description (extended)
        semantic_desc = await self._generate_enhanced_description(df, context_text, questions, llm)
        table_data['semantic_description'] = semantic_desc

        # 4. Update Knowledge Graph if repository is available
        if self.graph_repository:
            try:
                from tilellm.modules.knowledge_graph.models import Relationship
                
                # Link referencing paragraphs to table
                for elem in referencing_elements:
                    rel = Relationship(
                        source_id=elem.get('id'),
                        target_id=table_id,
                        type="REFERENCES",
                        properties={"doc_id": doc_id}
                    )
                    self.graph_repository.create_relationship(rel)
                
                # Update table node with new properties
                self.graph_repository.update_node(
                    node_id=table_id,
                    properties={
                        "questions": questions,
                        "semantic_description": semantic_desc
                    }
                )
            except Exception as e:
                logger.error(f"Error updating graph for table {table_id}: {e}")

        return table_data

    def _find_references(self, table_id: str, text_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find text elements that reference the table ID or number."""
        # table_id is usually like doc_id_table_page_index
        # We look for "Table X" where X is the index or some match
        
        import re
        # Try to extract the table number from the ID or caption
        # Simple heuristic: last part of ID
        try:
            table_num = table_id.split('_')[-1]
        except:
            table_num = None
            
        refs = []
        patterns = [rf"Table\s+{table_num}\b", rf"Tbl\.?\s+{table_num}\b"]
        if not table_num:
             patterns = [r"Table\s+\d+", r"Tbl\.?\s+\d+"]

        for elem in text_elements:
            text = elem.get('text', '')
            for p in patterns:
                if re.search(p, text, re.IGNORECASE):
                    refs.append(elem)
                    break
                    
        return refs

    async def _generate_table_questions(self, df: pd.DataFrame, context: str, llm: Any) -> List[str]:
        """Generate questions that can be answered by the table."""
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Convert first few rows to string for prompt
        table_snippet = df.head(10).to_string()
        
        system_prompt = """You are a helpful AI assistant specialized in analyzing tabular data from documents.
Generate specific questions that can be answered by table data."""
        
        human_prompt = f"""Analyze this table and generate 5 specific questions that can be answered by the data.

Context from document: {context}

Table Snippet:
{table_snippet}

Generate questions that:
1. Are specific and answerable directly from the data
2. Cover comparisons, trends, or specific values
3. Are useful for a user searching through this document

Return only a JSON list of strings."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

        try:
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback: split by lines
                lines = [l.strip().strip('123456789. -') for l in content.split('\n') if '?' in l]
                return lines[:5]
        except Exception as e:
            logger.error(f"Error generating table questions: {e}")
            return []

    async def _generate_enhanced_description(self, df: pd.DataFrame, context: str, questions: List[str], llm: Any) -> str:
        """Generate a rich semantic description of the table."""
        from langchain_core.messages import HumanMessage, SystemMessage
        
        table_snippet = df.head(5).to_string()
        cols = ", ".join(df.columns.tolist())
        
        system_prompt = """You are a helpful AI assistant specialized in analyzing tabular data from documents.
Provide comprehensive semantic descriptions that explain table purpose and structure."""
        
        human_prompt = f"""Provide a comprehensive semantic description of this table.        
Context: {context}
Columns: {cols}
Sample Data:
{table_snippet}

Sample questions it answers:
{chr(10).join([str(q) for q in questions])}

The description should explain the table's purpose, what the rows and columns represent, and why it's important in the context of the document.
Keep it under 150 words."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

        try:
            response = await llm.ainvoke(messages)
            return response.content.strip() if hasattr(response, 'content') else str(response).strip()
        except Exception as e:
            logger.error(f"Error generating enhanced table description: {e}")
            return f"Table with columns: {cols}"
