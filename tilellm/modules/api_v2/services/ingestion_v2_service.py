"""
POST /api/v2/ingestion — unified ingestion, canonical MD+frontmatter form.

Same input contract as the legacy /api/ingestion (ItemSingle), so existing
callers can adopt it by only changing the path. Routing:

  use_ocr=True (pdf/docx) -> unchanged legacy Docling/LLM-enrichment pipeline
                              (scrape_pdf / process_docx_with_images), reused
                              as-is — the opt-in for image captions/table
                              descriptions extracted by an LLM.
  type=regex_custom        -> unchanged legacy add_item/add_item_hybrid.
                              ponytail: regex-delimited chunks have no
                              ExtractedDocument shape defined yet — add a
                              canonical-form branch when a concrete use case
                              needs it, mirroring export/converters.py's
                              equivalent ceiling.
  everything else           -> canonical path: export_document() produces an
                              ExtractedDocument (same converters as
                              /api/export/md, PDF OCR decided per-document via
                              classify_pdf — see convert_pdf), written via
                              ingest.service.write_extracted_document() with
                              the same baseline provenance metadata
                              (id/source/file_name/page/position/heading_path)
                              as every other ingestion path. Dense-only or
                              hybrid via item.hybrid/sparse_encoder;
                              situated_context supported.

The legacy /api/ingestion (tilellm/modules/ingestion/controllers.py) is
production-critical and is not touched by this module — this is a new,
separate endpoint, not a replacement.
"""
import logging
from typing import Any

from tilellm.models.document_type import DocumentType
from tilellm.models.llm import ItemSingle
from tilellm.modules.ingestion.export.models import ExportMdRequest
from tilellm.modules.ingestion.export.service import export_document
from tilellm.modules.ingestion.ingest.models import IngestConfig, IngestMdResult
from tilellm.modules.ingestion.ingest.service import write_extracted_document
from tilellm.modules.ingestion.type_detector import resolve_item_type
from tilellm.shared.utility import inject_embeddings_async, inject_repo_async

logger = logging.getLogger(__name__)


def _resolve_type(item: ItemSingle):
    return resolve_item_type(
        current_type=item.type,
        source=item.source,
        content=item.content,
        file_name=None,  # ItemSingle carries no file_name/file_content (those are
        file_content=None,  # PDFScrapingRequest-only fields) — url/source-driven detection only.
    )


@inject_embeddings_async
@inject_repo_async
async def ingest_v2(
    item: ItemSingle,
    repo=None,
    llm_embeddings=None,
    embedding_config_key=None,
    **kwargs,
) -> Any:
    """Public DI-wired entry point (mirrors ingest_md/check_compliance_v2): both the
    embedding model and repo are injected here; the FastAPI route stays a thin wrapper.
    inject_embeddings_async (not inject_llm_chat_async) — ItemSingle has no .llm field,
    only .embedding/.gptkey, and no chat LLM is used on this path (situated_context, if
    enabled, builds its own LLM from config in _apply_situated_context)."""
    return await _ingest_v2_core(item, repo=repo, llm_embeddings=llm_embeddings)


async def _ingest_v2_core(item: ItemSingle, repo, llm_embeddings) -> Any:
    resolved_type = _resolve_type(item)
    if resolved_type is not None and resolved_type != item.type:
        logger.info("v2 ingestion: doc_id=%s type=%s -> %s", item.id, item.type, resolved_type.value)
        item = item.model_copy(update={"type": resolved_type})

    if resolved_type in (DocumentType.PDF, DocumentType.DOCX) and item.use_ocr:
        return await _route_legacy_ocr(item, resolved_type)

    if resolved_type == DocumentType.REGEX_CUSTOM:
        return await _route_legacy_regex_custom(item)

    return await _route_canonical(item, resolved_type, repo, llm_embeddings)


async def _route_legacy_ocr(item: ItemSingle, resolved_type: DocumentType):
    from tilellm.modules.ingestion.controllers import _build_pdf_request

    logger.info("v2 ingestion: doc_id=%s -> legacy OCR/LLM-enrichment pipeline (use_ocr=True)", item.id)
    pdf_request = _build_pdf_request(item)
    if resolved_type == DocumentType.DOCX:
        from tilellm.modules.ingestion.docx_processor import process_docx_with_images
        return await process_docx_with_images(pdf_request)
    from tilellm.modules.pdf_ocr.controllers import scrape_pdf
    return await scrape_pdf(pdf_request)


async def _route_legacy_regex_custom(item: ItemSingle):
    from tilellm.controller.controller import add_item, add_item_hybrid

    logger.info("v2 ingestion: doc_id=%s -> legacy regex_custom pipeline", item.id)
    if item.hybrid:
        return await add_item_hybrid(item)
    return await add_item(item)


async def _route_canonical(item: ItemSingle, resolved_type: DocumentType, repo, llm_embeddings) -> IngestMdResult:
    logger.info("v2 ingestion: doc_id=%s -> canonical MD+frontmatter pipeline", item.id)
    export_request = ExportMdRequest(
        type=resolved_type, source=item.source, content=item.content,
        scrape_type=int(item.scrape_type), parameters_scrape_type_4=item.parameters_scrape_type_4,
        browser_headers=item.browser_headers,
    )
    doc = await export_document(export_request)

    config = IngestConfig(
        id=item.id, namespace=item.namespace, engine=item.engine,
        embedding=item.embedding, hybrid=item.hybrid, sparse_encoder=item.sparse_encoder,
        tags=item.tags, chunk_size=item.chunk_size, chunk_overlap=item.chunk_overlap,
        situated_context=item.situated_context, gptkey=item.gptkey,
        id_project=item.id_project, request_id=item.request_id, debug=item.debug,
        additional_metadata=item.additional_metadata,
    )
    return await write_extracted_document(
        doc, config, repo, llm_embeddings,
        source_url=item.source, source_type=resolved_type.value if resolved_type else None,
    )
