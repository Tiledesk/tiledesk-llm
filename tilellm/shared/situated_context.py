"""
Situated context enrichment (Anthropic Contextual Retrieval technique).

Each chunk's page_content is prepended with 1-2 sentences that situate it
within the broader document, improving retrieval accuracy for dense/sparse search.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from tilellm.models.llm import SituatedContextConfig

logger = logging.getLogger(__name__)


@dataclass
class SituatedContextResult:
    """Return value of enrich_chunks_with_situated_context."""
    documents: List[Document]
    token_usage: dict = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    })

_SITUATED_CONTEXT_PROMPT = """\
Here is the context of the broader document this chunk comes from:
<document_context>
{doc_context}
</document_context>

Here is the specific chunk to situate:
<chunk>
{chunk_text}
</chunk>

Write 1-2 sentences that situate this chunk within the broader document. \
Mention the topic, section, or argument this chunk is part of. \
Reply with only the situating sentences, with no preamble or explanation."""

_SITUATED_CONTEXT_TABLE_PROMPT = """\
Here is the context of the broader document this table was extracted from:
<document_context>
{doc_context}
</document_context>

The following is a complete data table with columns: {col_names}

<table>
{chunk_text}
</table>

Write 1-2 sentences that describe what this table represents: its topic, \
the kind of entities it lists, and what data it holds. \
Reply with only the situating sentences, with no preamble or explanation."""

_SITUATED_CONTEXT_TABLE_ROW_PROMPT = """\
The following is a single row from a data table extracted from: {doc_context}
The table has these columns: {col_names}

<row>
{chunk_text}
</row>

Write 1-2 sentences that describe the specific item or entity represented by this row, \
referencing its actual values (not generic statements about the table structure). \
For example: "Product Alpha (SKU: A001) costs €9.99 and belongs to the Electronics category." \
Reply with only the descriptive sentences, with no preamble or explanation."""

_TABLE_ELEMENT_TYPES = {"table", "table_rows"}


async def _generate_situated_context(
    doc_context: str,
    chunk_text: str,
    llm,
    chunk_metadata: Optional[dict] = None,
) -> tuple[str, dict]:
    """Call LLM to generate a situating sentence for one chunk.

    Returns:
        (context_text, token_usage) where token_usage has input_tokens/output_tokens/total_tokens.
    """
    element_type = chunk_metadata.get("element_type") if chunk_metadata else None
    col_names = (chunk_metadata.get("col_names", "") if chunk_metadata else "") or ""
    source = (chunk_metadata.get("source", "") if chunk_metadata else "") or ""

    if element_type == "table_rows":
        # Per-row chunks: generate a row-specific description so that each row gets
        # a unique embedding that captures its actual cell values (e.g. "Product Alpha,
        # SKU A001, price €9.99").  A generic table-level description would be
        # identical for all rows and collapse their embeddings → bad retrieval.
        table_doc_context = source or doc_context
        prompt = _SITUATED_CONTEXT_TABLE_ROW_PROMPT.format(
            doc_context=table_doc_context,
            col_names=col_names,
            chunk_text=chunk_text[:1200],
        )
    elif element_type == "table":
        # Atomic/adaptive table: describe what the table represents as a whole.
        table_doc_context = source or doc_context
        prompt = _SITUATED_CONTEXT_TABLE_PROMPT.format(
            doc_context=table_doc_context,
            col_names=col_names,
            chunk_text=chunk_text[:1200],
        )
    else:
        prompt = _SITUATED_CONTEXT_PROMPT.format(
            doc_context=doc_context,
            chunk_text=chunk_text[:1200],
        )
    _empty_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        usage = getattr(response, "usage_metadata", None) or _empty_usage
        return response.content.strip(), {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        logger.warning(f"Situated context LLM call failed: {e}")
        return "", _empty_usage


async def enrich_chunks_with_situated_context(
    documents: List[Document],
    llm,
    doc_context: Optional[str] = None,
    max_concurrent: int = 5
) -> "SituatedContextResult":
    """
    Enrich a list of Document chunks by prepending a situated context sentence.

    Args:
        documents: List of chunks to enrich.
        llm: Async-capable LangChain LLM instance.
        doc_context: Brief overview of the document. Defaults to first ~800 chars of content.
        max_concurrent: Max concurrent LLM calls to avoid rate limits.

    Returns:
        SituatedContextResult with enriched documents and cumulative token_usage.
    """
    if not documents:
        return SituatedContextResult(documents=documents)

    if doc_context is None:
        # For table chunks the first chars are pipe-delimited markdown — useless as
        # document context.  Fall back to the source URL when available.
        first_doc_meta = documents[0].metadata if documents else {}
        if first_doc_meta.get("element_type") in _TABLE_ELEMENT_TYPES:
            doc_context = first_doc_meta.get("source", "") or ""
        else:
            combined = " ".join(d.page_content[:150] for d in documents[:8])
            doc_context = combined[:800]

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _enrich_one(doc: Document) -> tuple[Document, dict]:
        async with semaphore:
            ctx, usage = await _generate_situated_context(
                doc_context, doc.page_content, llm, chunk_metadata=doc.metadata
            )
            if ctx:
                doc.page_content = f"{ctx}\n\n{doc.page_content}"
                doc.metadata["has_situated_context"] = True
        return doc, usage

    results = await asyncio.gather(*[_enrich_one(doc) for doc in documents])

    enriched_docs = []
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for doc, usage in results:
        enriched_docs.append(doc)
        total_usage["input_tokens"] += usage["input_tokens"]
        total_usage["output_tokens"] += usage["output_tokens"]
        total_usage["total_tokens"] += usage["total_tokens"]

    return SituatedContextResult(documents=enriched_docs, token_usage=total_usage)


async def build_llm_from_config(config: "SituatedContextConfig", fallback_api_key: Optional[str] = None) -> Optional[object]:
    """
    Build LLM instance from SituatedContextConfig.
    Returns None if disabled or api_key is missing/invalid.

    Supports: openai, anthropic, google, groq, vllm (with custom url), ollama (with custom url)
    """
    if not config or not config.enable:
        return None

    api_key_obj = config.api_key
    api_key = (
        api_key_obj.get_secret_value() if hasattr(api_key_obj, 'get_secret_value')
        else str(api_key_obj or '')
    )

    if not api_key or api_key in ('', 'sk'):
        if fallback_api_key and fallback_api_key not in ('', 'sk'):
            api_key = fallback_api_key
        else:
            logger.warning("SituatedContextConfig: api_key is missing or invalid")
            return None

    kwargs = dict(temperature=config.temperature, max_tokens=config.max_tokens)

    try:
        if config.provider in ('openai', 'vllm'):
            from langchain_openai import ChatOpenAI
            init_kwargs = dict(model=config.model or 'gpt-4o-mini', api_key=api_key, **kwargs)
            if config.url:
                init_kwargs['base_url'] = config.url
            return ChatOpenAI(**init_kwargs)

        elif config.provider == 'anthropic':
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model or 'claude-haiku-4-5-20251001',
                anthropic_api_key=api_key,
                **kwargs
            )

        elif config.provider == 'google':
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.model or 'gemini-2.5-flash',
                google_api_key=api_key,
                **kwargs
            )

        elif config.provider == 'groq':
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=config.model or 'llama3-8b-8192',
                api_key=api_key,
                **kwargs
            )

        elif config.provider == 'ollama':
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model or 'llama3',
                base_url=config.url or 'http://localhost:11434',
                **kwargs
            )

        else:
            logger.warning(f"Situated context: unsupported provider '{config.provider}'")
            return None

    except Exception as e:
        logger.warning(f"Could not build LLM from SituatedContextConfig: {e}")
        return None


async def build_llm_from_item(item) -> Optional[object]:
    """
    Build LLM instance for situated context from an ItemSingle-like object.

    Prioritizes item.situated_context if configured.
    Returns None if LLM is not properly configured.
    """
    # New schema: SituatedContextConfig
    config = getattr(item, 'situated_context', None)
    if config is not None:
        # Get global gptkey as fallback if available
        gptkey_obj = getattr(item, 'gptkey', None)
        fallback_api_key = (
            gptkey_obj.get_secret_value() if hasattr(gptkey_obj, 'get_secret_value')
            else str(gptkey_obj or '')
        )
        return await build_llm_from_config(config, fallback_api_key=fallback_api_key)

    return None
