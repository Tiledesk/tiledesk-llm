"""
Agentic QA Service for FalkorDB Knowledge Graph.
Implements "The Graph Specialist" agent that autonomously queries the graph using Cypher.

Refactored to use LangGraph StateGraph for better observability and control.
"""
import logging
from typing import Dict, Any, Optional

from tilellm.modules.knowledge_graph_falkor.agents import GraphSpecialistAgent
from tilellm.modules.knowledge_graph_falkor.services.services import GraphService

logger = logging.getLogger(__name__)


class AgenticQAService:
    """
    Agentic QA Service using LangGraph StateGraph.

    This service is a thin wrapper around GraphSpecialistAgent,
    maintaining backward compatibility with the existing API.
    """

    def __init__(self, graph_service: GraphService, llm: Any):
        """
        Initialize the service.

        Args:
            graph_service: GraphService instance for accessing repository
            llm: LLM instance for the agent
        """
        self.graph_service = graph_service
        self.llm = llm
        self.repo = graph_service._get_repository()

        # Initialize the Graph Specialist Agent
        self.agent = GraphSpecialistAgent(repository=self.repo, llm=self.llm)

        logger.info("AgenticQAService initialized with StateGraph-based agent")

    async def process_query(
        self,
        question: str,
        namespace: str,
        chat_history_dict: Optional[Dict[str, Any]] = None,
        creation_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a natural language query using the Graph Specialist Agent.

        Args:
            question: Natural language question
            namespace: Graph namespace for multi-tenancy
            chat_history_dict: Chat history for context-aware queries
            creation_prompt: Domain identifier (e.g., "debt_recovery", "generic").
                           If None, uses "generic" domain to match the graph schema.

        Returns:
            Dictionary with answer, query_used, retrieval_strategy, and metadata
        """
        logger.info(
            f"Processing agentic query (domain: {creation_prompt or 'generic'}): {question[:100]}..."
        )

        # Delegate to the agent with domain context
        result = await self.agent.process_query(
            question=question,
            namespace=namespace,
            chat_history_dict=chat_history_dict,
            creation_prompt=creation_prompt,
        )

        return result
