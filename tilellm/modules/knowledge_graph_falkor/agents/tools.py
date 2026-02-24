"""
Tool factory functions for Graph Specialist Agent.

These tools are maintained for potential future use with ReAct-style agents
or for integration with other LangChain-based workflows.
"""

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


def create_cypher_executor_tool(repository: Any, namespace: str, graph_name: str = None) -> StructuredTool:
    """
    Factory function for Cypher executor tool.

    This creates a LangChain tool that executes read-only Cypher queries
    on the knowledge graph via the repository.

    Note: In the current StateGraph implementation, query execution happens
    in the executor_node. This tool is kept for potential future use with
    ReAct agents or hybrid approaches.

    Args:
        repository: AsyncFalkorGraphRepository instance
        namespace: Graph namespace for query execution

    Returns:
        StructuredTool instance for Cypher query execution
    """

    async def query_graph(cypher_query: str) -> str:
        """
        Executes a read-only Cypher query on the knowledge graph.

        Args:
            cypher_query: Cypher query string

        Returns:
            JSON string of results or error message
        """
        # Security check: Only read operations
        query_upper = cypher_query.strip().upper()
        if not (
            query_upper.startswith("MATCH")
            or query_upper.startswith("WITH")
            or query_upper.startswith("CALL")
            or query_upper.startswith("RETURN")
        ):
            return "Error: Only read operations (MATCH, WITH, CALL, RETURN) are allowed."

        try:
            logger.info(f"Tool executing Cypher: {cypher_query} on namespace: {namespace}")

            results = await repository._execute_query(cypher_query, {}, namespace=namespace, graph_name=graph_name)

            if not results:
                return "No results found."

            return json.dumps(results, default=str)

        except Exception as e:
            logger.error(f"Tool Cypher execution failed: {e}")
            return f"Error executing Cypher query: {str(e)}"

    return StructuredTool.from_function(
        coroutine=query_graph,
        name="query_graph",
        description="""Executes a Cypher query on the knowledge graph.

The database contains nodes and relationships for debt recovery:
- Nodes: Person, Organization, Loan, Guarantee, Protest, LegalProceeding, Payment
- Relationships: OBLIGATED_UNDER, HAS_LOAN, GUARANTEES, SECURED_BY, PROTESTATO_IL, HAS_LEGAL_ACTION, NEXT_STEP

Use this tool to find information about debtors, guarantors, loans, protests, legal actions, etc.
Only read operations are allowed (MATCH, WITH, CALL, RETURN).""",
    )
