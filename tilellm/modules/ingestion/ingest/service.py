"""
IngestMdRequest -> real vector-store write, via `repo.aadd_documents` (not
`add_item`/`add_item_hybrid` — see memory/ingestion_md_redesign.md for why:
the standard add_item "direct content" branch never applies
`additional_metadata`, so frontmatter would be silently dropped there).

Metadata is built by us per chunk. `IngestConfig.additional_metadata` (added
2026-08-06, see docs/GRAPHRAG_COST_QUALITY_PLAN.md A2) closes the gap this
docstring used to claim didn't exist here: `/api/v2/ingestion`'s canonical
path built `IngestConfig` from `ItemSingle` without ever passing
`additional_metadata` through, so callers relying on it (same field, same
semantics as the legacy pdf_ocr path) had it silently dropped. hybrid is a single
`sparse_encoder` parameter forwarded to `aadd_documents`, mirroring the
existing docx_processor.py pattern — including its known per-backend
inconsistency (Pinecone always embeds hybrid regardless of this parameter;
Qdrant honors None as dense-only; Milvus uses its own collection-level flag).
This module does not attempt to unify that — out of scope for F2.
"""
import json as json_module
import logging
import time
from typing import Awaitable, Callable, List, Optional, Union

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tilellm.analytics as analytics
from tilellm.modules.ingestion.export.models import ExtractedDocument
from tilellm.modules.ingestion.export.serializers import from_json, from_md
from tilellm.modules.ingestion.ingest.models import IngestConfig, IngestMdRequest, IngestMdResult
from tilellm.modules.ingestion.table_chunker import split_table_document
from tilellm.shared.utility import inject_embeddings_async, inject_repo_async
from tilellm.tools.document_tools import _apply_additional_metadata, _extract_file_name

logger = logging.getLogger(__name__)

MAX_SOURCE_SIZE = 10 * 1024 * 1024  # 10 MB


@inject_embeddings_async
@inject_repo_async
async def ingest_md(
    request: IngestMdRequest,
    repo=None,
    llm_embeddings=None,
    embedding_config_key=None,
    **kwargs,
) -> IngestMdResult:
    """Public DI-wired entry point (mirrors check_compliance_v2 in compliance_checker):
    both the embedding model and repo are injected here; the FastAPI route stays a thin
    wrapper. inject_embeddings_async (not inject_llm_chat_async) — IngestMdRequest has no
    .llm field, only .embedding/.gptkey, and no chat LLM is used on this path."""
    return await ingest_document(request, repo=repo, llm_embeddings=llm_embeddings)


async def ingest_document(request: IngestMdRequest, repo, llm_embeddings) -> IngestMdResult:
    return await write_extracted_document(
        lambda: _load_document(request), request, repo, llm_embeddings,
        source_url=request.md_url or request.json_url,
        source_type="md" if (request.md or request.md_url) else "json",
    )


async def write_extracted_document(
    doc: Union[ExtractedDocument, Callable[[], Awaitable[ExtractedDocument]]],
    config: IngestConfig,
    repo,
    llm_embeddings,
    *,
    source_url: Optional[str] = None,
    source_type: Optional[str] = None,
) -> IngestMdResult:
    """Load (if needed) + chunk + (optionally) situated-context-enrich + write
    to the vector store, with a kb.content_indexed analytics event either way
    (including a load failure — a zero-arg loader is accepted precisely so
    that failure stays inside this function's try/finally).

    Shared core for /api/ingest/md (ingest_document, above) and the canonical
    path of /api/v2/ingestion (api_v2/services/ingestion_v2_service.py), which
    already has an in-memory ExtractedDocument (from export_document) and
    skips the md/json round-trip entirely — pass the document directly there."""
    t0 = time.monotonic()
    error_msg: Optional[str] = None
    documents: List[Document] = []
    try:
        if callable(doc):
            doc = await doc()
        documents = _build_documents(doc, config)

        if config.situated_context and config.situated_context.enable and documents:
            documents = await _apply_situated_context(documents, config)

        sparse_encoder = config.sparse_encoder if config.hybrid else None
        chunk_ids = await repo.aadd_documents(
            engine=config.engine,
            documents=documents,
            namespace=config.namespace,
            embedding_model=llm_embeddings,
            sparse_encoder=sparse_encoder,
            metadata_id=config.id,
        )
        return IngestMdResult(
            id=config.id,
            namespace=config.namespace,
            chunks_indexed=len(documents),
            chunk_ids=list(chunk_ids or []),
        )
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        duration_ms = int((time.monotonic() - t0) * 1000)
        event_type, payload = analytics.events.content_indexed(
            kb_id=config.namespace,
            kb_name=config.namespace,
            embedding_model=analytics.events.get_embedding_model_name(config.embedding),
            engine=analytics.events.get_engine_value(config.engine),
            duration_ms=duration_ms,
            success=error_msg is None,
            source_url=source_url,
            source_type=source_type,
            chunks_indexed=len(documents),
            error_message=error_msg,
            request_id=config.request_id,
        )
        analytics.publish_nowait(event_type, config.id_project, payload)


async def _apply_situated_context(documents: List[Document], config: IngestConfig) -> List[Document]:
    from tilellm.shared.situated_context import build_llm_from_item, enrich_chunks_with_situated_context

    situated_llm = await build_llm_from_item(config)
    if not situated_llm:
        return documents
    result = await enrich_chunks_with_situated_context(
        documents,
        situated_llm,
        profile=config.situated_context.profile,
        custom_prompt=config.situated_context.custom_prompt,
        metadata_extraction_prompt=config.situated_context.metadata_extraction_prompt,
        metadata_only=config.situated_context.metadata_only,
    )
    return result.documents


async def _load_document(request: IngestMdRequest) -> ExtractedDocument:
    if request.md:
        return from_md(request.md)
    if request.json_content:
        return from_json(request.json_content)
    if request.md_url:
        return from_md((await _fetch(request.md_url)).decode("utf-8", errors="replace"))
    if request.json_url:
        return from_json((await _fetch(request.json_url)).decode("utf-8", errors="replace"))
    raise ValueError("Fornire uno tra 'md', 'md_url', 'json_content' o 'json_url'.")  # unreachable: validated by IngestMdRequest


async def _fetch(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content
    if len(content) > MAX_SOURCE_SIZE:
        raise ValueError(f"Il file è troppo grande ({len(content)} byte > {MAX_SOURCE_SIZE} byte massimi).")
    return content


def _scalar(value):
    """JSON-stringify non-scalar values — Pinecone/Qdrant/Milvus metadata must be
    scalar or list-of-str (mirrors the col_names-must-be-str precedent in table_chunker)."""
    if isinstance(value, (dict, list)) and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
        return json_module.dumps(value)
    return value


def _base_metadata(doc: ExtractedDocument, config: IngestConfig) -> dict:
    meta = {
        "id": config.id,
        "metadata_id": config.id,
        "namespace": config.namespace,
        "source": doc.resource or "",
        "doc_type": doc.type,
        # Same minimum provenance guarantee as add_item/chunk_documents: every
        # chunk must say which document it came from, citations read this key.
        "file_name": _extract_file_name(doc.resource or config.id),
    }
    if doc.title:
        meta["title"] = doc.title
    if doc.description:
        meta["description"] = doc.description
    if doc.resource:
        meta["resource"] = doc.resource
    if doc.timestamp:
        meta["timestamp"] = doc.timestamp
    for key, value in doc.extra.items():
        meta[key] = _scalar(value)

    tags = config.tags or doc.tags
    if tags:
        meta["tags"] = tags

    meta = _apply_additional_metadata(meta, getattr(config, "additional_metadata", None))
    return meta


def _build_documents(doc: ExtractedDocument, config: IngestConfig) -> List[Document]:
    base_meta = _base_metadata(doc, config)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
    )

    documents: List[Document] = []
    for block in sorted(doc.blocks, key=lambda b: b.order):
        block_meta = dict(base_meta)
        if block.page is not None:
            block_meta["page"] = block.page
        if block.position is not None:
            block_meta["position"] = block.position
        if block.heading_path:
            block_meta["heading_path"] = block.heading_path

        if block.block_type == "table":
            block_meta["element_type"] = "table"
            source_doc = Document(page_content=block.content, metadata=block_meta)
            documents.extend(split_table_document(source_doc, strategy=config.table_strategy))
        else:
            for chunk_text in splitter.split_text(block.content):
                documents.append(Document(page_content=chunk_text, metadata=dict(block_meta)))

    return documents
