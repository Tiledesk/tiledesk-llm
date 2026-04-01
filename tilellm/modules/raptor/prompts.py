"""
Prompts for RAPTOR summarization and tree traversal.
"""

# Prompt for generating summaries of chunk groups
RAPTOR_SUMMARY_PROMPT = """You are an expert summarization assistant. Your task is to create a concise, informative summary of the following text chunks.

Guidelines:
- Capture the main ideas and key information
- Maintain coherence and flow
- Preserve important facts, figures, and relationships
- Remove redundancy while keeping essential details
- Write in clear, professional language
- Keep the summary between 150-400 words

Text chunks to summarize:
{context}

Provide a comprehensive summary that captures the essence of these texts:"""


# Prompt for higher-level summaries (summarizing summaries)
RAPTOR_HIERARCHICAL_SUMMARY_PROMPT = """You are an expert at creating hierarchical summaries. Your task is to synthesize multiple lower-level summaries into a coherent higher-level summary.

Guidelines:
- Identify overarching themes and patterns
- Synthesize information across the input summaries
- Maintain logical flow and coherence
- Preserve critical insights from all levels
- Create a summary that stands alone while representing the full content
- Keep the summary between 200-500 words

Lower-level summaries to synthesize:
{context}

Provide a comprehensive higher-level summary:"""


# Prompt for tree traversal agent (decides which level to search)
RAPTOR_TRAVERSAL_SYSTEM_PROMPT = """You are an intelligent retrieval agent for a hierarchical document system (RAPTOR).

The document is organized in levels:
- Level 0: Raw text chunks (detailed, specific information)
- Level 1: Summaries of 5-10 chunks (medium granularity)
- Level 2+: Higher-level summaries (broad overview, key themes)

Your task is to decide which level(s) to search based on the user's question.

Decision criteria:
- **Specific/factual questions** (names, dates, numbers, technical details) → Search Level 0 (raw chunks)
- **Conceptual questions** (relationships, processes, explanations) → Search Level 1 (first summaries)
- **Broad/overview questions** (main themes, big picture, summary) → Search Level 2+ (higher summaries)
- **Complex multi-faceted questions** → Search multiple levels (start high, then go deeper)

Available actions:
- "search_this_level": Search the current level
- "go_deeper": Move to a more detailed level (higher number)
- "go_higher": Move to a more abstract level (lower number)
- "stop": Have enough information, stop searching

Respond in JSON format with:
{{
    "current_level": <int>,
    "action": "<action>",
    "reasoning": "<brief explanation>",
    "next_level": <int or null>
}}

User question: {question}
Current level: {current_level}
Available levels: {available_levels}"""


# Prompt for collapsed tree retrieval (standard RAG with all levels)
RAPTOR_COLLAPSED_TREE_PROMPT = """You are answering questions based on retrieved content from a hierarchical document summarization system (RAPTOR).

The retrieved content comes from different levels:
- Level 0: Raw text chunks
- Level 1+: Summaries at various levels of abstraction

Use the provided context to answer the question comprehensively. When information from different levels complements each other, synthesize it into a coherent response.

If the context doesn't contain enough information to answer the question, acknowledge this limitation.

Context from RAPTOR tree:
{context}

Question: {question}

Provide a comprehensive answer based on the context:"""


# Prompt for cluster formation (groups similar chunks)
RAPTOR_CLUSTERING_PROMPT = """You are helping organize document chunks into coherent groups for summarization.

Given the following chunk descriptions, group them by thematic similarity. Each group should contain chunks that discuss related topics or concepts.

Guidelines:
- Group chunks that share common themes, topics, or concepts
- Aim for groups of {cluster_size} chunks (±2)
- Ensure each chunk belongs to exactly one group
- Name each group with a descriptive theme label

Chunk descriptions:
{chunk_descriptions}

Respond in JSON format:
{{
    "groups": [
        {{
            "theme": "<descriptive name>",
            "chunk_ids": ["id1", "id2", ...]
        }},
        ...
    ]
}}"""


# Prompt for quality assessment of summaries
RAPTOR_QUALITY_CHECK_PROMPT = """Evaluate the quality of this summary against the source chunks.

Rate the summary on:
1. **Coverage**: Does it capture all main ideas? (0-5)
2. **Accuracy**: Is it faithful to the source? (0-5)
3. **Conciseness**: Is it appropriately condensed? (0-5)
4. **Coherence**: Is it well-organized and readable? (0-5)

Source chunks:
{source_chunks}

Summary:
{summary}

Respond in JSON format:
{{
    "coverage_score": <int 0-5>,
    "accuracy_score": <int 0-5>,
    "conciseness_score": <int 0-5>,
    "coherence_score": <int 0-5>,
    "overall_score": <int 0-5>,
    "feedback": "<brief feedback and suggestions for improvement>"
}}"""


# Prompt for RAPTOR Q&A (answer generation from retrieved context)
RAPTOR_QA_PROMPT = """You are an expert assistant answering questions using content retrieved from a hierarchical document system (RAPTOR).

The provided context contains relevant information from various levels of the document hierarchy:
- Some content is from raw text chunks (detailed, specific)
- Some content is from summaries (synthesized, high-level)

Your task is to synthesize all available context to provide a comprehensive, accurate answer.

Guidelines:
1. Use only information from the provided context
2. Synthesize information from different levels when relevant
3. Provide a clear, well-structured answer
4. If the context doesn't contain enough information, acknowledge this limitation
5. Be precise and avoid speculation

Context from RAPTOR tree:
{context}

Question: {question}

Provide a comprehensive answer based on the context:"""
