"""
Agentic QA Service for FalkorDB Knowledge Graph.
Implements "The Graph Specialist" agent that autonomously queries the graph using Cypher.
"""
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional

from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# We need the repository from graph_service
from .services import GraphService

logger = logging.getLogger(__name__)

class AgenticQAService:
    def __init__(self, graph_service: GraphService, llm: Any):
        self.graph_service = graph_service
        self.llm = llm
        self.repo = graph_service._get_repository()
        
    async def process_query(
        self, 
        question: str, 
        namespace: str, 
        chat_history_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language query using the Agentic Graph Specialist.
        """
        
        # --- TOOL DEFINITION ---
        
        async def query_credit_graph(cypher_query: str) -> str:
            """
            Executes a Cypher query on the debt recovery knowledge graph.
            Use this tool to answer questions about debtors, guarantors, protests, loans, legal actions, and their relationships.
            
            The database has the following schema:
            - Labels: Person, Organization, Loan, Guarantee, Protest, LegalProceeding, Payment
            - Relationships: 
              - (:Person|Organization)-[:OBLIGATED_UNDER|HAS_LOAN]->(:Loan)
              - (:Person|Organization)-[:GUARANTEES]->(:Loan)
              - (:Loan)<-[:SECURED_BY]-(:Guarantee)
              - (:Person|Organization)-[:PROTESTATO_IL]->(:Protest)
              - (:Loan)-[:HAS_LEGAL_ACTION]->(:LegalProceeding)
              - (:LegalProceeding)-[:NEXT_STEP]->(:LegalProceeding)
            
            The tool returns a list of records as a JSON string.
            If the result is very large, it handles summarization automatically.
            """
            try:
                # Security check: Limit query types (read-only)
                if not cypher_query.strip().upper().startswith("MATCH") and \
                   not cypher_query.strip().upper().startswith("WITH") and \
                   not cypher_query.strip().upper().startswith("CALL") and \
                   not cypher_query.strip().upper().startswith("RETURN"): # Allow simple RETURN
                     return "Error: Only read operations (MATCH, WITH, CALL, RETURN) are allowed."
                
                logger.info(f"Agent executing Cypher: {cypher_query} on namespace: {namespace}")
                
                # Execute query via repository
                # We assume the repository method returns a list of dictionaries
                results = await self.repo._execute_query(cypher_query, {}, namespace=namespace)
                
                if not results:
                    return "No results found."

                # --- MAP-REDUCE LOGIC FOR LARGE RESULTS ---
                MAP_REDUCE_THRESHOLD = 50  # Items
                
                if len(results) > MAP_REDUCE_THRESHOLD:
                    logger.info(f"Large result set ({len(results)} items). Applying Map-Reduce summarization.")
                    return await self._apply_map_reduce(results, question)

                return json.dumps(results, default=str)
                
            except Exception as e:
                logger.error(f"Agent Cypher execution failed: {e}")
                # Provide the error to the agent so it can self-correct
                return f"Error executing Cypher query: {str(e)}"

        # Create the LangChain tool
        tools = [
            StructuredTool.from_function(
                coroutine=query_credit_graph,
                name="query_credit_graph",
                description="Executes a Cypher query on the debt recovery knowledge graph. Use this to find debtors, guarantors, loans, protests, etc."
            )
        ]

        # --- PROMPT DEFINITION ---
        
        system_prompt = """You are 'The Graph Specialist', an expert agent in debt recovery (recupero crediti).
Your goal is to answer user questions by querying the Knowledge Graph using the `query_credit_graph` tool.

### Graph Schema
- **Nodes**: 
  - `Person`, `Organization`: Debtors, Guarantors, Creditors. Properties: `name`, `tax_id`.
  - `Loan`: Credit files. Properties: `id`, `amount`, `status`, `principal_amount`.
  - `Guarantee`: Securities. Properties: `type`, `amount`, `value`.
  - `Protest`: Protest events. Properties: `amount`, `date`.
  - `LegalProceeding`: Legal actions. Properties: `type`, `date`, `status`, `court`.
  - `Payment`: Payments made. Properties: `amount`, `date`.

- **Relationships**:
  - `(Debtor)-[:OBLIGATED_UNDER]->(Loan)` (Debtor connects to Loan)
  - `(Creditor)-[:HAS_LOAN]->(Loan)` (Bank connects to Loan)
  - `(Guarantor)-[:GUARANTEES]->(Loan)` (CRITICAL: Guarantors are linked to the LOAN, not directly to the debtor)
  - `(Loan)<-[:SECURED_BY]-(Guarantee)` (Loan secured by Guarantee)
  - `(Debtor)-[:PROTESTATO_IL]->(Protest)` (Debtor has Protest)
  - `(Loan)-[:HAS_LEGAL_ACTION]->(LegalProceeding)` (Loan has Legal Action)
  - `(LegalProceeding)-[:NEXT_STEP]->(LegalProceeding)` (for timeline)

### Instructions
1. **Analyze the Request**: Understand specific intent (Guarantors, Protests, Timeline, Exposure).
2. **Generate Cypher**: Write a precise Cypher query.
   - **Guarantors**: `MATCH (p)-[:OBLIGATED_UNDER|HAS_LOAN]->(l)<-[:GUARANTEES]-(g) WHERE ... RETURN g.name, labels(g), l.id`
   - **Protests**: `MATCH (p)-[:PROTESTATO_IL]->(pr) WHERE ... RETURN pr`
   - **Timeline**: `MATCH (p)-[:OBLIGATED_UNDER|HAS_LOAN]->(l)-[:HAS_LEGAL_ACTION|HAS_PAYMENT|RESULTED_IN]->(event) ... RETURN event.date, type(event), event.description ORDER BY event.date`
   - Use `toUpper(n.name) CONTAINS toUpper('NAME')` for case-insensitive matching.
3. **Execute**: Call `query_credit_graph`.
4. **Synthesize**: Present the answer clearly to the user based on the tool output.
   - If the tool says "No results found", say so.
   - If the tool returns a summary, use it to construct the final answer.

### Constraint
- **Self-Correction**: If the query fails, analyze the error and try a corrected query.
- **Precision**: Do not guess. Only use data returned by the tool.
"""

        # --- AGENT CREATION & EXECUTION (LANGGRAPH) ---
        
        # Create the LangGraph agent
        app = create_react_agent(self.llm, tools)
        
        # Prepare input messages
        # We prepend the system prompt to the messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]

        try:
            # Execute the graph
            result = await app.ainvoke({"messages": messages})
            
            # Extract the final response
            # result["messages"] contains the full conversation history including tool calls
            final_message = result["messages"][-1]
            answer = final_message.content
            
            return {
                "answer": answer,
                "query_used": question,
                "retrieval_strategy": "agentic_qa"
            }
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "retrieval_strategy": "agentic_qa_error"
            }

    async def _apply_map_reduce(self, results: List[Dict], question: str) -> str:
        """
        Applies Map-Reduce to summarize large result sets.
        """
        # Chunking
        CHUNK_SIZE = 20
        chunks = [results[i:i + CHUNK_SIZE] for i in range(0, len(results), CHUNK_SIZE)]
        
        logger.info(f"Splitting {len(results)} results into {len(chunks)} chunks for summarization.")
        
        # Map Step (Parallel Summarization)
        summaries = []
        for chunk in chunks:
            # We use a simple prompt for the map step
            chunk_str = json.dumps(chunk, default=str)
            map_prompt = f"""
            Analyze these partial graph results relevant to the question: "{question}"
            
            Data:
            {chunk_str}
            
            Extract key events, entities, and facts. 
            Format as a concise list of items (Date - Event - Details).
            If there are dates, ensure they are preserved.
            """
            
            try:
                if hasattr(self.llm, 'ainvoke'):
                    response = await self.llm.ainvoke(map_prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                else:
                    response = self.llm.invoke(map_prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                summaries.append(content)
            except Exception as e:
                logger.error(f"Map step failed: {e}")
                
        # Reduce Step
        combined_summaries = "\n".join(summaries)
        return f"Found {len(results)} records. Summarized content:\n{combined_summaries}"
