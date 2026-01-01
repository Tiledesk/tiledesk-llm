"""
Advanced query analysis utilities for GraphRAG.

Provides LLM-based query type detection and intent classification
to optimize retrieval strategies.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query type classification."""
    EXPLORATORY = "exploratory"  # Broad, overview questions
    TECHNICAL = "technical"      # Specific technical questions
    RELATIONAL = "relational"    # Questions about relationships/connections


# Query type detection prompt
QUERY_TYPE_DETECTION_PROMPT = """Classify the following question into ONE of these types:

1. EXPLORATORY: Broad questions seeking overview or general understanding
   - Examples: "Tell me about...", "What is...", "Explain...", "Give me an overview..."
   - Characteristics: Open-ended, high-level, informational

2. TECHNICAL: Specific technical questions with precise requirements
   - Examples: "How do I fix error X?", "What's the syntax for...", "Step-by-step guide..."
   - Characteristics: Specific, actionable, often includes technical terms or error codes

3. RELATIONAL: Questions about relationships, connections, or comparisons
   - Examples: "How does X affect Y?", "What's the relationship between...", "Compare X and Y..."
   - Characteristics: Focuses on connections, influence, correlation, causation

Question: "{question}"

Analyze the question and respond with ONLY one word - the classification: EXPLORATORY, TECHNICAL, or RELATIONAL"""


async def detect_query_type_with_llm(
    question: str,
    llm: Any,
    fallback_to_heuristic: bool = True
) -> str:
    """
    Detect query type using LLM for accurate classification.

    Args:
        question: User's question
        llm: LLM instance for classification
        fallback_to_heuristic: If True, fall back to rule-based detection on error

    Returns:
        Query type as string: "exploratory", "technical", or "relational"
    """
    try:
        prompt = QUERY_TYPE_DETECTION_PROMPT.format(question=question)

        # Try async invocation
        if hasattr(llm, 'ainvoke'):
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
        elif hasattr(llm, 'invoke'):
            # Fallback to sync
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
        else:
            raise AttributeError("LLM does not have ainvoke or invoke method")

        # Parse response
        query_type = content.strip().lower()

        # Validate response
        valid_types = [QueryType.EXPLORATORY.value, QueryType.TECHNICAL.value, QueryType.RELATIONAL.value]
        if query_type not in valid_types:
            logger.warning(f"LLM returned invalid query type: '{query_type}'. Falling back to heuristic.")
            if fallback_to_heuristic:
                return detect_query_type_heuristic(question)
            return QueryType.EXPLORATORY.value  # Safe default

        logger.info(f"LLM detected query type: {query_type} for question: '{question[:50]}...'")
        return query_type

    except Exception as e:
        logger.error(f"Error detecting query type with LLM: {e}")
        if fallback_to_heuristic:
            logger.info("Falling back to heuristic-based query type detection")
            return detect_query_type_heuristic(question)
        return QueryType.EXPLORATORY.value


def detect_query_type_heuristic(question: str) -> str:
    """
    Detect query type using rule-based heuristics.
    Fallback method when LLM is not available or fails.

    Args:
        question: User's question

    Returns:
        Query type as string
    """
    question_lower = question.lower()

    # Exploratory queries: broad, asking about concepts
    exploratory_phrases = [
        "tell me about", "what is", "what are", "explain", "describe",
        "overview", "introduction", "summarize", "give me", "show me",
        "parlami di", "cos'è", "cos'è", "spiega", "descrivi"  # Italian
    ]
    if any(phrase in question_lower for phrase in exploratory_phrases):
        return QueryType.EXPLORATORY.value

    # Technical queries: specific errors, codes, technical details
    technical_phrases = [
        "error", "code", "bug", "issue", "fix", "problem",
        "how to", "tutorial", "step by step", "configure",
        "install", "setup", "syntax", "command", "function",
        "errore", "codice", "problema", "come fare", "configurare"  # Italian
    ]
    if any(phrase in question_lower for phrase in technical_phrases):
        return QueryType.TECHNICAL.value

    # Relational queries: connections, relationships, influence
    relational_phrases = [
        "relation", "relationship", "connect", "connection", "influence",
        "impact", "affect", "correlate", "between", "vs", "versus",
        "compare", "comparison", "difference", "similar",
        "relazione", "connessione", "influenza", "impatto", "confronto"  # Italian
    ]
    if any(phrase in question_lower for phrase in relational_phrases):
        return QueryType.RELATIONAL.value

    # Check for question words that might indicate type
    if any(word in question_lower for word in ["how does", "why does", "what causes"]):
        return QueryType.RELATIONAL.value

    # Default to exploratory for general questions
    return QueryType.EXPLORATORY.value


def get_weight_adjustments(query_type: str) -> Dict[str, float]:
    """
    Get weight multipliers based on query type.

    This implements the "Balancing Matrix" for different query types:
    - Exploratory: High vector, Low keyword, Medium graph, High community
    - Technical: Low vector, High keyword, Low graph, Low community
    - Relational: Medium vector, Medium keyword, Very High graph, High community

    Args:
        query_type: Query type classification

    Returns:
        Dictionary with weight multipliers for each retrieval method
    """
    weight_matrix = {
        QueryType.EXPLORATORY.value: {
            "vector": 1.5,      # High - semantic understanding important
            "keyword": 0.5,     # Low - exact matches less critical
            "graph": 1.0,       # Medium - relationships provide context
            "community": 1.8    # High - community reports great for overview
        },
        QueryType.TECHNICAL.value: {
            "vector": 0.5,      # Low - exact terms matter more
            "keyword": 1.8,     # High - specific terms/codes critical
            "graph": 0.5,       # Low - relationships less relevant
            "community": 0.3    # Low - need specific details not summaries
        },
        QueryType.RELATIONAL.value: {
            "vector": 1.0,      # Medium - semantic understanding helps
            "keyword": 1.0,     # Medium - both concepts important
            "graph": 2.5,       # Very High - relationships are core
            "community": 1.5    # High - community structure shows patterns
        }
    }

    return weight_matrix.get(query_type, weight_matrix[QueryType.EXPLORATORY.value])


def apply_weight_adjustments(
    base_weights: Dict[str, float],
    query_type: str
) -> Dict[str, float]:
    """
    Apply query type adjustments to base weights.

    Args:
        base_weights: Base weights dictionary (e.g., {"vector": 1.0, "keyword": 1.0, "graph": 1.0})
        query_type: Query type classification

    Returns:
        Adjusted weights dictionary
    """
    adjustments = get_weight_adjustments(query_type)
    adjusted_weights = {}

    for key in base_weights:
        if key in adjustments:
            adjusted_weights[key] = base_weights[key] * adjustments[key]
        else:
            adjusted_weights[key] = base_weights[key]

    logger.info(f"Applied {query_type} adjustments: {base_weights} -> {adjusted_weights}")

    return adjusted_weights
