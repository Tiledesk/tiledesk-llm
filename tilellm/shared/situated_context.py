"""
Situated context enrichment (Anthropic Contextual Retrieval technique).

Each chunk's page_content is prepended with 1-2 sentences that situate it
within the broader document, improving retrieval accuracy for dense/sparse search.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""
import asyncio
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

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


async def _generate_situated_context(doc_context: str, chunk_text: str, llm) -> str:
    """Call LLM to generate a situating sentence for one chunk."""
    prompt = _SITUATED_CONTEXT_PROMPT.format(
        doc_context=doc_context,
        chunk_text=chunk_text[:1200]
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Situated context LLM call failed: {e}")
        return ""


async def enrich_chunks_with_situated_context(
    documents: List[Document],
    llm,
    doc_context: Optional[str] = None,
    max_concurrent: int = 5
) -> List[Document]:
    """
    Enrich a list of Document chunks by prepending a situated context sentence.

    Args:
        documents: List of chunks to enrich.
        llm: Async-capable LangChain LLM instance.
        doc_context: Brief overview of the document. Defaults to first ~800 chars of content.
        max_concurrent: Max concurrent LLM calls to avoid rate limits.

    Returns:
        Same list of documents, with page_content prefixed with situated context.
    """
    if not documents:
        return documents

    if doc_context is None:
        combined = " ".join(d.page_content[:150] for d in documents[:8])
        doc_context = combined[:800]

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _enrich_one(doc: Document) -> Document:
        async with semaphore:
            ctx = await _generate_situated_context(doc_context, doc.page_content, llm)
            if ctx:
                doc.page_content = f"{ctx}\n\n{doc.page_content}"
                doc.metadata["has_situated_context"] = True
        return doc

    return list(await asyncio.gather(*[_enrich_one(doc) for doc in documents]))


async def build_llm_from_item(item) -> Optional[object]:
    """
    Build a minimal LLM instance from an ItemSingle-like object for situated context.
    Returns None if LLM fields are not configured or are placeholder values.
    """
    llm_provider = getattr(item, 'llm_provider', None) or getattr(item, 'llm', None)
    llm_model = getattr(item, 'llm_model', None) or (
        item.model if isinstance(getattr(item, 'model', None), str) else
        getattr(getattr(item, 'model', None), 'name', None)
    )
    api_key_obj = getattr(item, 'gptkey', None) or getattr(item, 'llm_key', None)

    if not llm_provider or not api_key_obj:
        return None

    try:
        api_key = api_key_obj.get_secret_value() if hasattr(api_key_obj, 'get_secret_value') else str(api_key_obj)
        if not api_key or api_key in ('sk', ''):
            return None

        if llm_provider in ('openai', 'vllm'):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=llm_model or 'gpt-4o-mini', api_key=api_key, temperature=0.0, max_tokens=256)
        elif llm_provider == 'anthropic':
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=llm_model or 'claude-haiku-4-5-20251001', anthropic_api_key=api_key, temperature=0.0, max_tokens=256)
        elif llm_provider == 'google':
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=llm_model or 'gemini-1.5-flash', google_api_key=api_key, temperature=0.0, max_tokens=256)
        else:
            logger.warning(f"Situated context: unsupported LLM provider '{llm_provider}'")
            return None
    except Exception as e:
        logger.warning(f"Could not build LLM for situated context: {e}")
        return None
