"""
Agentic QA Service for FalkorDB Knowledge Graph.
Implements "The Graph Specialist" agent that autonomously queries the graph using Cypher.

Refactored to use LangGraph StateGraph for better observability and control.
"""
import logging
from typing import Dict, Any, Optional

from tilellm.modules.knowledge_graph_falkor.agents import GraphSpecialistAgent
from tilellm.modules.knowledge_graph_falkor.services.services import GraphService
from tilellm.controller.controller_utils import summarize_history, create_contextualize_query
from tilellm.models import QuestionAnswer, Engine

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
        engine: Optional[Engine] = None,
        # New Hybrid History Flags
        contextualize_prompt: bool = False,
        include_history_in_prompt: bool = True,
        max_history_messages: int = 10,
        conversation_summary: bool = False,
        graph_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language query using the Graph Specialist Agent.

        Args:
            question: Natural language question
            namespace: Graph namespace for multi-tenancy
            chat_history_dict: Chat history for context-aware queries
            creation_prompt: Domain identifier (e.g., "debt_recovery", "generic").
                           If None, uses "generic" domain to match the graph schema.
            engine: engine to use
            contextualize_prompt: Enable query rewriting for retrieval
            include_history_in_prompt: Include history in final synthesis
            max_history_messages: Limit history turns
            conversation_summary: Enable history summarization

        Returns:
            Dictionary with answer, query_used, retrieval_strategy, and metadata
        """
        logger.info(
            f"Processing agentic query (domain: {creation_prompt or 'generic'}): {question[:100]}..."
        )

        # 0. HYBRID HISTORY PREPARATION
        retrieval_query = question
        summary_text = ""
        
        if chat_history_dict and (contextualize_prompt or conversation_summary):

            
            # Since we don't have engine/vector_store easily here, we might skip contextualization 
            # if it requires vector store init, but create_contextualize_query handles just the LLM call.
            # However, it needs a QuestionAnswer object. We'll provide a mock one.
            qa_mock = QuestionAnswer(
                question=question,
                namespace=namespace,
                chat_history_dict=chat_history_dict,
                contextualize_prompt=contextualize_prompt,
                max_history_messages=max_history_messages,
                engine=engine # Minimal mock
            )

            if contextualize_prompt:
                try:
                    retrieval_query = await create_contextualize_query(self.llm, qa_mock)
                    logger.info(f"Contextualized query for agent: {retrieval_query}")
                except Exception as e:
                    logger.warning(f"Query contextualization failed: {e}")
            
            if conversation_summary:
                sorted_keys = sorted(chat_history_dict.keys(), key=lambda x: int(x))
                if len(sorted_keys) > max_history_messages:
                    from ..utils import format_chat_history
                    old_keys = sorted_keys[:-max_history_messages]
                    old_history_dict = {k: chat_history_dict[k] for k in old_keys}
                    old_history_text = format_chat_history(old_history_dict)
                    summary_text = await summarize_history(old_history_text, self.llm)

        # Delegate to the agent with domain context and potentially contextualized query
        # Note: We pass the ORIGINAL question to maintain the intent, but use retrieval_query for internal search if needed.
        # However, the GraphSpecialistAgent handles its own history. We'll pass the summary if available.
        
        result = await self.agent.process_query(
            question=retrieval_query, # Use contextualized query
            namespace=namespace,
            chat_history_dict=chat_history_dict,
            creation_prompt=creation_prompt,
            max_history_messages=max_history_messages,
            summary_text=summary_text,
            graph_name=graph_name
        )

        # Add contextualization info to result
        if contextualize_prompt:
            result["query_contextualized"] = retrieval_query
            result["original_question"] = question

        return result
