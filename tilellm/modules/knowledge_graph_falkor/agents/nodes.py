"""
Node implementations for Graph Specialist Agent.
Uses closure pattern for dependency injection of repository and llm.
"""

import logging
import json
import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from tilellm.modules.knowledge_graph_falkor.agents.state import GraphSpecialistState
from tilellm.modules.knowledge_graph_falkor.agents.prompts import (
    QUERY_GENERATOR_SYSTEM_PROMPT,
    QUERY_GENERATOR_CORRECTION_PROMPT,
    RESPONDER_SYSTEM_PROMPT,
    SUMMARIZER_MAP_PROMPT,
)

logger = logging.getLogger(__name__)


# --- Pydantic Models for Structured Output ---


class CypherQuery(BaseModel):
    """Structured output for query generator."""

    query: str = Field(description="The Cypher query to execute")
    explanation: str = Field(description="Brief explanation of what the query does")


# --- Node Factory Function ---


def create_nodes(repository, llm):
    """
    Factory function that creates node functions with access to repository and llm via closure.

    Args:
        repository: AsyncFalkorGraphRepository instance
        llm: LLM instance (with ainvoke and with_structured_output support)

    Returns:
        Dictionary mapping node names to node functions
    """

    # --- Node 1: Query Generator ---

    async def query_generator_node(state: GraphSpecialistState) -> Dict[str, Any]:
        """
        Generates a Cypher query from the user's natural language question.
        Includes self-correction feedback if retry_count > 0.
        """
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        error_message = state.get("error_message")

        # Build prompt with optional correction feedback
        system_prompt = QUERY_GENERATOR_SYSTEM_PROMPT
        if retry_count > 0 and error_message:
            system_prompt += QUERY_GENERATOR_CORRECTION_PROMPT.format(
                error_message=error_message
            )

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]

        try:
            # Use structured output
            structured_llm = llm.with_structured_output(CypherQuery)
            result = await structured_llm.ainvoke(messages)

            cypher_query = result.query
            explanation = result.explanation

            logger.info(f"Generated Cypher query: {cypher_query}")

            # Update trace
            trace = list(state.get("metadata", {}).get("trace", []))
            trace.append(
                {
                    "step": "query_generator",
                    "cypher_query": cypher_query,
                    "explanation": explanation,
                    "retry_count": retry_count,
                    "timestamp": time.time(),
                }
            )

            return {
                "cypher_query": cypher_query,
                "cypher_explanation": explanation,
                "metadata": {**state.get("metadata", {}), "trace": trace},
            }

        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Return error state
            trace = list(state.get("metadata", {}).get("trace", []))
            trace.append(
                {
                    "step": "query_generator",
                    "error": str(e),
                    "timestamp": time.time(),
                }
            )
            return {
                "error_message": f"Query generation failed: {str(e)}",
                "validation_status": "max_retries",  # Force fail-safe
                "metadata": {**state.get("metadata", {}), "trace": trace},
            }

    # --- Node 2: Executor ---

    async def executor_node(state: GraphSpecialistState) -> Dict[str, Any]:
        """
        Executes the Cypher query via repository.
        Includes security validation (read-only operations).
        """
        cypher_query = state["cypher_query"]
        namespace = state["namespace"]

        # Security check: Only read operations
        query_upper = cypher_query.strip().upper()
        if not (
            query_upper.startswith("MATCH")
            or query_upper.startswith("WITH")
            or query_upper.startswith("CALL")
            or query_upper.startswith("RETURN")
        ):
            error_msg = "Error: Only read operations (MATCH, WITH, CALL, RETURN) are allowed."
            logger.warning(f"Security violation: {cypher_query}")

            trace = list(state.get("metadata", {}).get("trace", []))
            trace.append(
                {
                    "step": "executor",
                    "security_violation": True,
                    "timestamp": time.time(),
                }
            )

            return {
                "query_results": None,
                "result_count": 0,
                "error_message": error_msg,
                "metadata": {**state.get("metadata", {}), "trace": trace},
            }

        try:
            logger.info(f"Executing Cypher: {cypher_query} on namespace: {namespace}")

            # Execute query via repository
            results = await repository._execute_query(cypher_query, {}, namespace=namespace)

            result_count = len(results) if results else 0

            logger.info(f"Query returned {result_count} results")

            # Update trace
            trace = list(state.get("metadata", {}).get("trace", []))
            trace.append(
                {
                    "step": "executor",
                    "result_count": result_count,
                    "timestamp": time.time(),
                }
            )

            return {
                "query_results": results,
                "result_count": result_count,
                "metadata": {**state.get("metadata", {}), "trace": trace},
            }

        except Exception as e:
            logger.error(f"Query execution failed: {e}")

            trace = list(state.get("metadata", {}).get("trace", []))
            trace.append(
                {
                    "step": "executor",
                    "error": str(e),
                    "timestamp": time.time(),
                }
            )

            return {
                "query_results": None,
                "result_count": 0,
                "error_message": f"Query execution failed: {str(e)}",
                "metadata": {**state.get("metadata", {}), "trace": trace},
            }

    # --- Node 3: Validator ---

    async def validator_node(state: GraphSpecialistState) -> Dict[str, Any]:
        """
        Validates query results and determines next step.
        Returns validation_status: "success", "syntax_error", "empty_results", or "max_retries".
        """
        query_results = state.get("query_results")
        result_count = state.get("result_count", 0)
        error_message = state.get("error_message")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        # Check for errors
        if error_message:
            # Check if we've exceeded max retries
            if retry_count >= max_retries:
                status = "max_retries"
                logger.warning(f"Max retries ({max_retries}) exceeded")
            else:
                status = "syntax_error"
                logger.info(f"Syntax error detected, retry {retry_count + 1}/{max_retries}")
        # Check for empty results
        elif result_count == 0:
            if retry_count >= max_retries:
                status = "max_retries"
                logger.warning(f"Max retries ({max_retries}) exceeded with empty results")
            else:
                status = "empty_results"
                logger.info(f"Empty results, retry {retry_count + 1}/{max_retries}")
        else:
            status = "success"
            logger.info(f"Validation successful with {result_count} results")

        # Update trace
        trace = list(state.get("metadata", {}).get("trace", []))
        trace.append(
            {
                "step": "validator",
                "validation_status": status,
                "retry_count": retry_count,
                "timestamp": time.time(),
            }
        )

        return {
            "validation_status": status,
            "metadata": {**state.get("metadata", {}), "trace": trace},
        }

    # --- Node 4: Responder ---

    async def responder_node(state: GraphSpecialistState) -> Dict[str, Any]:
        """
        Generates final answer for the user.
        Includes map-reduce logic for large result sets (>50 items).
        """
        query_results = state.get("query_results", [])
        result_count = state.get("result_count", 0)
        question = state["question"]

        # Map-Reduce for large results
        MAP_REDUCE_THRESHOLD = 50

        if result_count > MAP_REDUCE_THRESHOLD:
            logger.info(
                f"Large result set ({result_count} items). Applying Map-Reduce summarization."
            )

            try:
                summary = await _apply_map_reduce(llm, query_results, question)
                # Use summary instead of raw results
                context = f"Summary of {result_count} records:\n{summary}"
            except Exception as e:
                logger.error(f"Map-reduce failed: {e}")
                # Fallback: Use first 50 results
                context = json.dumps(query_results[:50], default=str)
                context += f"\n\n(Showing first 50 of {result_count} results)"
        else:
            # Small result set - use directly
            context = json.dumps(query_results, default=str)

        # Generate final answer
        messages = [
            {"role": "system", "content": RESPONDER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nGraph Results:\n{context}\n\nProvide a clear answer to the user's question based on these results.",
            },
        ]

        try:
            response = await llm.ainvoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)

            logger.info(f"Generated answer (length: {len(answer)})")

            # Update trace
            trace = list(state.get("metadata", {}).get("trace", []))
            trace.append(
                {
                    "step": "responder",
                    "answer_length": len(answer),
                    "used_map_reduce": result_count > MAP_REDUCE_THRESHOLD,
                    "timestamp": time.time(),
                }
            )

            return {
                "answer": answer,
                "metadata": {**state.get("metadata", {}), "trace": trace},
            }

        except Exception as e:
            logger.error(f"Response generation failed: {e}")

            # Fallback answer
            answer = f"I found {result_count} results but encountered an error generating the response: {str(e)}"

            trace = list(state.get("metadata", {}).get("trace", []))
            trace.append(
                {
                    "step": "responder",
                    "error": str(e),
                    "timestamp": time.time(),
                }
            )

            return {
                "answer": answer,
                "metadata": {**state.get("metadata", {}), "trace": trace},
            }

    # --- Node 5: Fail Safe ---

    async def fail_safe_node(state: GraphSpecialistState) -> Dict[str, Any]:
        """
        Handles failure cases (max retries exceeded, unrecoverable errors).
        Provides user-friendly error message.
        """
        error_message = state.get("error_message", "Unknown error")
        retry_count = state.get("retry_count", 0)

        answer = f"I apologize, but I was unable to answer your question after {retry_count} attempts. The last error was: {error_message}"

        logger.warning(f"Fail-safe activated: {error_message}")

        # Update trace
        trace = list(state.get("metadata", {}).get("trace", []))
        trace.append(
            {
                "step": "fail_safe",
                "error_message": error_message,
                "timestamp": time.time(),
            }
        )

        return {
            "answer": answer,
            "metadata": {**state.get("metadata", {}), "trace": trace},
        }

    # Return node dictionary
    return {
        "query_generator": query_generator_node,
        "executor": executor_node,
        "validator": validator_node,
        "responder": responder_node,
        "fail_safe": fail_safe_node,
    }


# --- Helper Function for Map-Reduce ---


async def _apply_map_reduce(llm, results: List[Dict], question: str) -> str:
    """
    Applies Map-Reduce to summarize large result sets.

    Args:
        llm: LLM instance
        results: List of query results
        question: Original user question

    Returns:
        Summarized content as string
    """
    CHUNK_SIZE = 20
    chunks = [results[i : i + CHUNK_SIZE] for i in range(0, len(results), CHUNK_SIZE)]

    logger.info(f"Splitting {len(results)} results into {len(chunks)} chunks for summarization.")

    # Map Step (Sequential for simplicity in MVP)
    summaries = []
    for i, chunk in enumerate(chunks):
        chunk_str = json.dumps(chunk, default=str)

        map_prompt = SUMMARIZER_MAP_PROMPT.format(question=question, chunk_data=chunk_str)

        try:
            response = await llm.ainvoke(map_prompt)
            content = response.content if hasattr(response, "content") else str(response)
            summaries.append(content)
            logger.debug(f"Processed chunk {i+1}/{len(chunks)}")
        except Exception as e:
            logger.error(f"Map step failed for chunk {i+1}: {e}")
            summaries.append(f"[Error processing chunk {i+1}]")

    # Reduce Step
    combined_summaries = "\n\n".join(summaries)
    return f"Found {len(results)} records. Summarized content:\n\n{combined_summaries}"
