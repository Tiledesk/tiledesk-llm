"""
IngestMdRequest -> real vector-store write, via `repo.aadd_documents` (not
`add_item`/`add_item_hybrid` — see memory/ingestion_md_redesign.md for why:
the standard add_item "direct content" branch never applies
`additional_metadata`, so frontmatter would be silently dropped there).

Metadata is built by us per chunk (no gap to work around). hybrid is a single
`sparse_encoder` parameter forwarded to `aadd_documents`, mirroring the
existing docx_processor.py pattern — including its known per-backend
inconsistency (Pinecone always embeds hybrid regardless of this parameter;
Qdrant honors None as dense-only; Milvus uses its own collection-level flag).
This module does not attempt to unify that — out of scope for F2.
"""
import json as json_module
import logging
from typing import List, Optional

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tilellm.modules.ingestion.export.models import ExtractedDocument
from tilellm.modules.ingestion.export.serializers import from_json, from_md
from tilellm.modules.ingestion.ingest.models import IngestMdRequest, IngestMdResult
from tilellm.modules.ingestion.table_chunker import split_table_document
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async

logger = logging.getLogger(__name__)

MAX_SOURCE_SIZE = 10 * 1024 * 1024  # 10 MB


@inject_llm_chat_async
@inject_repo_async
async def ingest_md(
    request: IngestMdRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    embedding_config_key=None,
    **kwargs,
) -> IngestMdResult:
    """Public DI-wired entry point (mirrors check_compliance_v2 in compliance_checker):
    both LLM and repo are injected here; the FastAPI route stays a thin wrapper."""
    return await ingest_document(request, repo=repo, llm_embeddings=llm_embeddings)


async def ingest_document(request: IngestMdRequest, repo, llm_embeddings) -> IngestMdResult:
    doc = await _load_document(request)
    documents = _build_documents(doc, request)

    sparse_encoder = request.sparse_encoder if request.hybrid else None
    chunk_ids = await repo.aadd_documents(
        engine=request.engine,
        documents=documents,
        namespace=request.namespace,
        embedding_model=llm_embeddings,
        sparse_encoder=sparse_encoder,
        metadata_id=request.id,
    )
    return IngestMdResult(
        id=request.id,
        namespace=request.namespace,
        chunks_indexed=len(documents),
        chunk_ids=list(chunk_ids or []),
    )


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


def _base_metadata(doc: ExtractedDocument, request: IngestMdRequest) -> dict:
    meta = {
        "id": request.id,
        "metadata_id": request.id,
        "namespace": request.namespace,
        "source": doc.resource or "",
        "doc_type": doc.type,
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

    tags = request.tags or doc.tags
    if tags:
        meta["tags"] = tags
    return meta


def _build_documents(doc: ExtractedDocument, request: IngestMdRequest) -> List[Document]:
    base_meta = _base_metadata(doc, request)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap,
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
            documents.extend(split_table_document(source_doc, strategy=request.table_strategy))
        else:
            for chunk_text in splitter.split_text(block.content):
                documents.append(Document(page_content=chunk_text, metadata=dict(block_meta)))

    return documents
