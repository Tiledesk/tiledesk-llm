"""
Situated context enrichment (Anthropic Contextual Retrieval technique).

Each chunk's page_content is prepended with 1-2 sentences that situate it
within the broader document, improving retrieval accuracy for dense/sparse search.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""
import asyncio
import json
import logging
import os
import re
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Dict, Any, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from tilellm.models.llm import SituatedContextConfig

logger = logging.getLogger(__name__)

# Cache for loaded profiles to avoid repeated disk I/O
_PROFILES_CACHE: Dict[str, Dict[str, Any]] = {}
_PROFILES_DIR = Path(__file__).parent / "profiles" / "situated_context"


def _load_profile_data(profile_name: str) -> Optional[Dict[str, Any]]:
    """Load full profile dict from a YAML profile file."""
    if profile_name in _PROFILES_CACHE:
        return _PROFILES_CACHE[profile_name]

    profile_path = _PROFILES_DIR / f"{profile_name}.yaml"
    if not profile_path.exists():
        logger.warning(f"Situated context profile '{profile_name}' not found at {profile_path}")
        return None

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if data.get("prompt"):
                _PROFILES_CACHE[profile_name] = data
                return data
    except Exception as e:
        logger.error(f"Error loading situated context profile '{profile_name}': {e}")

    return None


def _load_profile_prompt(profile_name: str) -> Optional[str]:
    """Load prompt string from a YAML profile file (backward compat)."""
    data = _load_profile_data(profile_name)
    return data.get("prompt") if data else None


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
    custom_prompt: Optional[str] = None,
    profile: Optional[str] = None,
    metadata_extraction_prompt: Optional[str] = None,
) -> Tuple[str, dict, dict]:
    """Call LLM to generate a situating sentence for one chunk.

    When *metadata_extraction_prompt* is set the LLM is asked to return a JSON
    object ``{"context": "...", "metadata": {...}}``; the metadata dict is
    returned as the third element and will be merged into doc.metadata by the
    caller.

    Returns:
        (context_text, token_usage, extracted_metadata)
        where token_usage has input_tokens/output_tokens/total_tokens.
    """
    element_type = chunk_metadata.get("element_type") if chunk_metadata else None
    if not element_type and chunk_metadata:
         # Fallback to 'type' field which is often used as a synonym for element_type
         element_type = chunk_metadata.get("type")

    # Robust col_names extraction (handles both 'col_names' and 'columns' list)
    col_names = ""
    if chunk_metadata:
        col_names = chunk_metadata.get("col_names", "")
        if not col_names and "columns" in chunk_metadata:
            cols = chunk_metadata["columns"]
            if isinstance(cols, list):
                col_names = ", ".join(cols)
            elif isinstance(cols, str):
                col_names = cols
    
    source = (chunk_metadata.get("source", "") if chunk_metadata else "") or ""

    # Priority 1: metadata_extraction_prompt (dual-output JSON mode)
    # Priority 2: custom_prompt (per-request override, plain text)
    # Priority 3: profile (loaded from YAML)
    # Priority 4: standard table/row prompts
    # Priority 5: standard generic prompt

    _fmt_kwargs = dict(
        doc_context=doc_context,
        chunk_text=chunk_text[:1200],
        col_names=col_names,
        source=source,
        element_type=element_type or "text",
    )

    use_json_mode = bool(metadata_extraction_prompt)
    base_prompt = metadata_extraction_prompt or custom_prompt
    if not base_prompt and profile:
        profile_data = _load_profile_data(profile)
        if profile_data:
            base_prompt = profile_data.get("prompt")
            if profile_data.get("json_mode", False):
                use_json_mode = True

    if base_prompt:
        # Replace only pure-identifier placeholders like {doc_context} or {chunk_text}.
        # JSON examples in the prompt (e.g. {"act_type": ..., "amount": 45000.00})
        # contain braces with quotes, spaces, or dots that str.format_map() would
        # mis-parse as format specs, raising KeyError or ValueError.
        # A regex that matches only \w+ keys is immune to those false positives.
        def _sub(m: re.Match) -> str:
            return str(_fmt_kwargs[m.group(1)]) if m.group(1) in _fmt_kwargs else m.group(0)
        prompt = re.sub(r"\{(\w+)\}", _sub, base_prompt)

    if not base_prompt:
        if element_type == "table_rows":
            table_doc_context = source or doc_context
            prompt = _SITUATED_CONTEXT_TABLE_ROW_PROMPT.format(
                doc_context=table_doc_context,
                col_names=col_names,
                chunk_text=chunk_text[:1200],
            )
        elif element_type == "table":
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
        # Handle reasoning models that return list content
        raw_content = response.content
        if isinstance(raw_content, list):
            raw_content = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in raw_content
            )
        raw_content = raw_content.strip()

        usage = getattr(response, "usage_metadata", None) or _empty_usage
        token_usage = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        if use_json_mode:
            # Strip markdown fences if model added them
            clean = raw_content
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.lower().startswith("json"):
                    clean = clean[4:]
            try:
                parsed = json.loads(clean.strip())
                context_str = str(parsed.get("context", "")).strip()
                extracted_meta = parsed.get("metadata", {})
                if not isinstance(extracted_meta, dict):
                    extracted_meta = {}
                return context_str, token_usage, extracted_meta
            except (json.JSONDecodeError, AttributeError) as parse_err:
                logger.warning(
                    f"Situated context JSON parse failed ({parse_err}); "
                    "falling back to plain-text context, no metadata extracted."
                )
                return raw_content, token_usage, {}

        return raw_content, token_usage, {}

    except Exception as e:
        logger.warning(f"Situated context LLM call failed: {e}")
        return "", _empty_usage, {}


async def enrich_chunks_with_situated_context(
    documents: List[Document],
    llm,
    doc_context: Optional[str] = None,
    max_concurrent: int = 5,
    custom_prompt: Optional[str] = None,
    profile: Optional[str] = None,
    metadata_extraction_prompt: Optional[str] = None,
) -> "SituatedContextResult":
    """Enrich a list of Document chunks by prepending a situated context sentence.

    When *metadata_extraction_prompt* is set (or a profile that embeds JSON output),
    each LLM call also extracts structured metadata that is merged into doc.metadata.
    This is useful for domain-specific enrichment (e.g. PA italiana: act_type, topics,
    amount) at zero marginal cost over the existing situated-context call.

    Args:
        documents: List of chunks to enrich.
        llm: Async-capable LangChain LLM instance.
        doc_context: Brief overview of the document. Defaults to first ~800 chars.
        max_concurrent: Max concurrent LLM calls to avoid rate limits.
        custom_prompt: Optional prompt override (plain-text situated context only).
        profile: Optional profile name loaded from YAML.
        metadata_extraction_prompt: When set, enables dual JSON output mode.
            The LLM returns {\"context\": \"...\", \"metadata\": {...}} and the
            metadata fields are merged into each chunk's doc.metadata.

    Returns:
        SituatedContextResult with enriched documents and cumulative token_usage.
    """
    if not documents:
        return SituatedContextResult(documents=documents)

    if doc_context is None:
        first_doc_meta = documents[0].metadata if documents else {}
        if first_doc_meta.get("element_type") in _TABLE_ELEMENT_TYPES or first_doc_meta.get("type") in _TABLE_ELEMENT_TYPES:
            doc_context = first_doc_meta.get("source", "") or ""
        else:
            combined = " ".join(d.page_content[:150] for d in documents[:8])
            doc_context = combined[:800]

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _enrich_one(doc: Document) -> Tuple[Document, dict]:
        async with semaphore:
            ctx, usage, extracted_meta = await _generate_situated_context(
                doc_context,
                doc.page_content,
                llm,
                chunk_metadata=doc.metadata,
                custom_prompt=custom_prompt,
                profile=profile,
                metadata_extraction_prompt=metadata_extraction_prompt,
            )
            if ctx:
                doc.page_content = f"{ctx}\n\n{doc.page_content}"
                doc.metadata["has_situated_context"] = True
            if extracted_meta:
                # Merge extracted fields; prefix sc_ to avoid collisions with
                # existing metadata keys that have different semantics.
                # Exception: well-known fields (act_type, topics, amount, …) are
                # written directly so they can be used as Qdrant payload filters.
                _DIRECT_FIELDS = {
                    "act_type", "topics", "amount", "personnel_role",
                    "temporal_scope", "doc_category", "main_topics",
                    "key_entities", "sentiment",
                }
                for k, v in extracted_meta.items():
                    target_key = k if k in _DIRECT_FIELDS else f"sc_{k}"
                    doc.metadata[target_key] = v
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
