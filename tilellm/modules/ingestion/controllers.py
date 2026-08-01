import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response
from tilellm.models.document_type import DocumentType
from tilellm.models.llm import ItemSingle
from tilellm.controller.controller import add_item, add_item_hybrid
from tilellm.modules.pdf_ocr.controllers import scrape_pdf
from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest
from tilellm.modules.ingestion.type_detector import resolve_item_type
from tilellm.modules.ingestion.export.models import ExportMdRequest
from tilellm.modules.ingestion.export.service import export_document
from tilellm.modules.ingestion.export.serializers import to_json, to_md
from tilellm.modules.ingestion.ingest.models import IngestMdRequest, IngestMdResult
from tilellm.modules.ingestion.ingest.service import ingest_md

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


def _build_pdf_request(item: ItemSingle) -> PDFScrapingRequest:
    """
    Build a PDFScrapingRequest from an ItemSingle.

    PDFScrapingRequest extends ItemSingle with additional required fields
    (file_name, file_content) and uses ``llm`` / ``model`` instead of
    ``llm_provider`` / ``llm_model`` for the DI decorators.

    Two things this function must get right, both learned the hard way:

    * **Pipeline.** ``ItemSingle`` carries none of the PDF-specific switches, so a
      request built from ``model_dump()`` alone leaves ``use_docling=False`` and
      ``strategy=None`` — and ``scrape_pdf`` then picks the *legacy* MinIO pipeline,
      which enqueues onto a Redis list that has no consumer in this codebase. The
      documents are uploaded and never processed. We therefore request the Docling
      (Taskiq) pipeline explicitly, while leaving ``strategy='quality'`` reachable
      for callers that really want the legacy path.
    * **Unknown keys.** ``pdf_options`` keys are validated against the real
      ``PDFScrapingRequest`` fields and rejected when unknown. Silently dropping an
      unrecognised option is exactly the failure mode above.
    """
    data = item.model_dump()

    options = data.pop("pdf_options", None) or {}
    if options:
        unknown = sorted(set(options) - set(PDFScrapingRequest.model_fields))
        if unknown:
            raise ValueError(
                f"pdf_options: chiavi sconosciute {unknown}. "
                f"Ammesse: {sorted(PDFScrapingRequest.model_fields)}"
            )
        data.update(options)

    # Docling/Taskiq pipeline unless the caller asked for something else: the legacy
    # alternative enqueues onto a queue nothing consumes.
    if not data.get("use_docling") and not data.get("strategy"):
        data["strategy"] = "auto"

    # file_content: URL or base64 blob — use source as primary
    if not data.get("file_content"):
        data["file_content"] = item.source or item.content or ""

    # file_name: derive from source URL
    if not data.get("file_name"):
        source = item.source or ""
        filename = source.rstrip("/").split("/")[-1]
        data["file_name"] = filename if filename else "document.pdf"

    # NOTE: ItemSingle now uses a nested situated_context object instead of flat fields.
    # The Pydantic model_validate will handle the nested object correctly.

    return PDFScrapingRequest.model_validate(data)


def _resolve_type(item: ItemSingle) -> DocumentType | None:
    """Detect and return the effective DocumentType for this request.

    Returns the resolved type, or None when the item carries only direct text
    content (no source / file_name / file_content) — in that case the
    repository's direct-text branch handles chunking autonomously.
    """
    file_content = getattr(item, "file_content", None)
    file_name = getattr(item, "file_name", None)

    resolved = resolve_item_type(
        current_type=item.type,
        source=item.source,
        content=item.content,
        file_name=file_name,
        file_content=file_content,
    )
    return resolved


@router.post("/ingestion")
async def unified_ingestion(item: ItemSingle):
    """
    Unified ingestion endpoint.

    When ``type`` is omitted or set to ``"auto"``, the system automatically
    detects the document format using (in priority order):

    1. **Magic bytes** — reads the first bytes of ``file_content`` (base64) to
       identify PDF (`%PDF`) and legacy Office / XLS (`OLE2`) files.
    2. **File extension** — from ``file_name`` or the path component of ``source``.
    3. **URL heuristic** — an ``http(s)`` source with no known file extension is
       treated as a web page (``type=url``).
    4. **Fallback** — plain-text / direct-content path (``type=txt``).

    Routing table after type resolution:

    ┌──────────────────────────────────────────────────────────────────────┐
    │ type = pdf   + use_ocr = True  → Advanced OCR pipeline (Docling)    │
    │ type = docx  + use_ocr = True  → DOCX image pipeline                │
    │ any type     + hybrid  = True  → add_item_hybrid                    │
    │ default                        → add_item (all other types)         │
    └──────────────────────────────────────────────────────────────────────┘

    Old endpoints ``/api/scrape/single`` and ``/api/pdf/scrape`` remain active
    for backward compatibility.
    """
    # ── Type resolution ───────────────────────────────────────────────────
    resolved_type = _resolve_type(item)

    if resolved_type is not None and resolved_type != item.type:
        logger.info(
            "Auto-detected document type for doc_id=%s: %s → %s",
            item.id,
            item.type,
            resolved_type.value,
        )
        item = item.model_copy(update={"type": resolved_type})
    elif resolved_type is None and item.type in (None, DocumentType.AUTO):
        # No signal at all (no source, no content, no file) — leave type unresolved
        logger.info(
            "doc_id=%s: no type signal detected, leaving type unresolved", item.id
        )

    logger.info(
        "Unified ingestion: doc_id=%s type=%s use_ocr=%s hybrid=%s",
        item.id, item.type, item.use_ocr, item.hybrid,
    )

    # ── PDF + OCR → Docling pipeline ──────────────────────────────────────
    if item.type == DocumentType.PDF and item.use_ocr:
        logger.info("Routing doc_id=%s → Advanced PDF/OCR pipeline", item.id)
        try:
            pdf_request = _build_pdf_request(item)
        except Exception as exc:
            logger.error("Failed to build PDFScrapingRequest for %s: %s", item.id, exc)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request for OCR pipeline: {exc}",
            )
        return await scrape_pdf(pdf_request)

    # ── DOCX + OCR → image-aware DOCX pipeline ────────────────────────────
    if item.type == DocumentType.DOCX and item.use_ocr:
        logger.info("Routing doc_id=%s → DOCX image pipeline", item.id)
        from tilellm.modules.ingestion.docx_processor import process_docx_with_images
        try:
            pdf_request = _build_pdf_request(item)
        except Exception as exc:
            logger.error("Failed to build PDFScrapingRequest for DOCX %s: %s", item.id, exc)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request for DOCX image pipeline: {exc}",
            )
        return await process_docx_with_images(pdf_request)

    # ── Hybrid ────────────────────────────────────────────────────────────
    if item.hybrid:
        logger.info("Routing doc_id=%s → hybrid ingestion pipeline", item.id)
        return await add_item_hybrid(item)

    # ── Standard ──────────────────────────────────────────────────────────
    logger.info("Routing doc_id=%s → standard ingestion pipeline", item.id)
    return await add_item(item)


@router.post("/export/md")
async def export_md_endpoint(request: ExportMdRequest):
    """
    Extract a document into the canonical form (Markdown+frontmatter or JSON) —
    **no vector store write**. The counterpart ingestion endpoint (parsing the
    frontmatter back into a real vector-store write) is a separate, later step.

    `format`: 'md' (default, frontmatter+body) or 'json' (lossless).
    """
    try:
        doc = await export_document(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if request.format == "json":
        return Response(content=to_json(doc), media_type="application/json")
    return PlainTextResponse(content=to_md(doc), media_type="text/markdown")


@router.post("/ingest/md", response_model=IngestMdResult)
async def ingest_md_endpoint(request: IngestMdRequest):
    """
    Ingest a previously-exported document (Markdown+frontmatter or JSON, from
    POST /api/export/md) into the vector store — the F2 counterpart of export/md.

    Writes via `repo.aadd_documents` (not add_item/add_item_hybrid): the caller's
    frontmatter becomes real per-chunk metadata with no gap. `hybrid=True`
    additionally generates sparse vectors (`sparse_encoder`).
    """
    try:
        return await ingest_md(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
