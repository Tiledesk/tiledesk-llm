"""
Graph Specialist Agent - LangGraph StateGraph implementation.

This agent orchestrates a multi-step workflow for answering questions about
a knowledge graph using Cypher queries with self-correction capabilities.
"""

import logging
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END

from tilellm.modules.knowledge_graph_falkor.agents.state import GraphSpecialistState
from tilellm.modules.knowledge_graph_falkor.agents.nodes import create_nodes

logger = logging.getLogger(__name__)


class GraphSpecialistAgent:
    """
    Graph Specialist Agent using LangGraph StateGraph.

    This agent implements a workflow with the following nodes:
    1. query_generator: Generates Cypher query from natural language
    2. executor: Executes the query via repository
    3. validator: Validates results and determines routing
    4. responder: Generates final answer (with map-reduce for large results)
    5. fail_safe: Handles failure cases

    The workflow includes self-correction: if a query fails or returns empty results,
    the validator routes back to query_generator with error feedback, up to max_retries.
    """

    def __init__(self, repository, llm):
        """
        Initialize the Graph Specialist Agent.

        Args:
            repository: AsyncFalkorGraphRepository instance
            llm: LLM instance with ainvoke and with_structured_output support
        """
        self.repository = repository
        self.llm = llm

        # Create node functions with dependency injection via closure
        self.nodes = create_nodes(repository, llm)

        # Build and compile the graph
        self.graph = self._build_graph()

        logger.info("GraphSpecialistAgent initialized with StateGraph")

    def _build_graph(self) -> Any:
        """
        Builds the LangGraph StateGraph workflow.

        Returns:
            Compiled StateGraph application
        """
        workflow = StateGraph(GraphSpecialistState)

        # Add nodes (MVP: 5 core nodes)
        workflow.add_node("query_generator", self.nodes["query_generator"])
        workflow.add_node("executor", self.nodes["executor"])
        workflow.add_node("validator", self.nodes["validator"])
        workflow.add_node("responder", self.nodes["responder"])
        workflow.add_node("fail_safe", self.nodes["fail_safe"])

        # Set entry point
        workflow.set_entry_point("query_generator")

        # Sequential edges
        workflow.add_edge("query_generator", "executor")
        workflow.add_edge("executor", "validator")

        # Conditional routing after validation
        def route_after_validation(state: GraphSpecialistState) -> str:
            """
            Routes based on validation status.

            Returns:
                - "respond": Query succeeded, proceed to response generation
                - "retry": Query failed but can retry (error or empty results)
                - "fail_safe": Max retries exceeded, go to fail-safe
            """
            status = state["validation_status"]

            if status == "success":
                return "respond"
            elif status == "max_retries":
                return "fail_safe"
            else:  # "syntax_error" or "empty_results"
                # Increment retry counter before retry
                state["retry_count"] = state.get("retry_count", 0) + 1
                return "retry"

        workflow.add_conditional_edges(
            "validator",
            route_after_validation,
            {
                "respond": "responder",
                "retry": "query_generator",  # Self-correction loop
                "fail_safe": "fail_safe",
            },
        )

        # Terminal edges
        workflow.add_edge("responder", END)
        workflow.add_edge("fail_safe", END)

        # Compile and return
        return workflow.compile()

    async def process_query(
        self,
        question: str,
        namespace: str,
        chat_history_dict: Optional[Dict[str, Any]] = None,
        creation_prompt: Optional[str] = None,
        max_retries: int = 3,
        max_history_messages: int = 10,
        summary_text: Optional[str] = None,
        graph_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language query against the knowledge graph.

        This is the main entry point for the agent. It initializes the state
        and executes the LangGraph workflow.

        Args:
            question: Natural language question
            namespace: Graph namespace for multi-tenancy
            chat_history_dict: Chat history for context-aware queries
            creation_prompt: Domain identifier (e.g., "debt_recovery", "generic").
                           If None, uses "generic" domain.
            max_retries: Maximum number of query generation retries
            max_history_messages: Maximum number of history turns to consider
            summary_text: Optional summary of older conversation
            graph_name: Graph name

        Returns:
            Dictionary with keys:
                - answer: Final answer to the question
                - query_used: Cypher query that was executed
                - retrieval_strategy: "agentic_qa"
                - metadata: Trace and execution metadata
        """
        # Initialize state
        initial_state: GraphSpecialistState = {
            "question": question,
            "namespace": namespace,
            "graph_name": graph_name,
            "chat_history": chat_history_dict,
            "summary_text": summary_text,
            "max_history_messages": max_history_messages,
            "creation_prompt": creation_prompt,
            "graph_schema": None,
            "cypher_query": None,
            "cypher_explanation": None,
            "query_results": None,
            "result_count": 0,
            "retry_count": 0,
            "max_retries": max_retries,
            "error_message": None,
            "validation_status": "pending",
            "answer": None,
            "metadata": {"trace": [], "domain": creation_prompt or "generic"},
        }

        try:
            logger.info(f"Processing query: {question}")

            # Execute the graph
            result = await self.graph.ainvoke(initial_state)

            logger.info(
                f"Query processed successfully. Answer length: {len(result.get('answer', ''))}"
            )

            # Update chat history with current turn in project standard format
            from tilellm.models import ChatEntry
            updated_history = result.get("chat_history") or {}
            next_key = str(len(updated_history))
            updated_history[next_key] = ChatEntry(
                question=question, 
                answer=result.get("answer", "No answer generated")
            )

            # Return standardized response
            return {
                "answer": result.get("answer", "No answer generated"),
                "query_used": result.get("cypher_query"),
                "retrieval_strategy": "agentic_qa",
                "metadata": result.get("metadata", {}),
                "chat_history_dict": updated_history
            }

        except Exception as e:
            logger.error(f"Graph specialist execution failed: {e}", exc_info=True)

            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "query_used": None,
                "retrieval_strategy": "agentic_qa_error",
                "metadata": {"error": str(e)},
            }
