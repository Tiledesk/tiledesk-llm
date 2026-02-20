"""
State model for GraphSpecialist Agent.
Defines the state structure used by the LangGraph StateGraph.
"""

from typing import TypedDict, Optional, Any, List, Dict


class GraphSpecialistState(TypedDict):
    """
    State for the Graph Specialist Agent workflow.

    This state is passed through the LangGraph nodes and updated at each step.
    Each node returns a dictionary of updates that LangGraph merges into the state.
    """

    # Input
    question: str
    namespace: str
    chat_history: Optional[Dict[str, Any]]  # Chat history for context
    summary_text: Optional[str]  # Summarized conversation context
    max_history_messages: int  # Limit history turns
    creation_prompt: Optional[str]  # Domain identifier (e.g., "debt_recovery", "generic")

    # Schema context (optional future enhancement)
    graph_schema: Optional[Dict[str, Any]]

    # Query generation
    cypher_query: Optional[str]
    cypher_explanation: Optional[str]

    # Execution
    query_results: Optional[List[Dict]]
    result_count: int

    # Control flow
    retry_count: int
    max_retries: int
    error_message: Optional[str]
    validation_status: str  # "success", "syntax_error", "empty_results", "max_retries"

    # Final output
    answer: Optional[str]

    # Tracing & metadata
    metadata: Dict[str, Any]  # {"trace": [...], "execution_time": ..., etc.}
