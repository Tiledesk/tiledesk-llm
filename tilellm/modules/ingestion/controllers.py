import logging
from fastapi import APIRouter, HTTPException
from tilellm.models.llm import ItemSingle
from tilellm.controller.controller import add_item, add_item_hybrid
from tilellm.modules.pdf_ocr.controllers import scrape_pdf
from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


def _build_pdf_request(item: ItemSingle) -> PDFScrapingRequest:
    """
    Build a PDFScrapingRequest from an ItemSingle.

    PDFScrapingRequest extends ItemSingle with additional required fields
    (file_name, file_content) and uses ``llm`` / ``model`` instead of
    ``llm_provider`` / ``llm_model`` for the DI decorators.
    """
    data = item.model_dump()

    # file_content: URL or base64 blob — use source as primary
    if not data.get("file_content"):
        data["file_content"] = item.source or item.content or ""

    # file_name: derive from source URL
    if not data.get("file_name"):
        source = item.source or ""
        filename = source.rstrip("/").split("/")[-1]
        data["file_name"] = filename if filename else "document.pdf"

    # Map ItemSingle's llm_provider / llm_model → PDFScrapingRequest's llm / model
    # (the @inject_llm_chat_async decorator reads question.llm and question.model)
    if item.llm_provider and not data.get("llm"):
        data["llm"] = item.llm_provider
    if item.llm_model and not data.get("model"):
        data["model"] = item.llm_model

    return PDFScrapingRequest.model_validate(data)


@router.post("/ingestion")
async def unified_ingestion(item: ItemSingle):
    """
    Unified ingestion endpoint.  Routes to the correct pipeline based on
    document type and configuration:

    ┌──────────────────────────────────────────────────────────────────┐
    │ type = "pdf"   + use_ocr = True  → OCR pipeline (Docling)        │
    │ type = "docx"  + use_ocr = True  → DOCX image pipeline           │
    │ any type       + hybrid  = True  → add_item_hybrid               │
    │ default                          → add_item                      │
    └──────────────────────────────────────────────────────────────────┘

    Old endpoints /api/scrape/single and /api/pdf/scrape remain active
    for backward compatibility.
    """
    logger.info(
        "Unified ingestion: doc_id=%s type=%s use_ocr=%s hybrid=%s",
        item.id, item.type, item.use_ocr, item.hybrid,
    )

    # ── PDF + OCR → Docling pipeline ──────────────────────────────────
    if item.type == "pdf" and item.use_ocr:
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

    # ── DOCX + OCR → image-aware DOCX pipeline ────────────────────────
    if item.type == "docx" and item.use_ocr:
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

    # ── Hybrid ────────────────────────────────────────────────────────
    if item.hybrid:
        logger.info("Routing doc_id=%s → hybrid ingestion pipeline", item.id)
        return await add_item_hybrid(item)

    # ── Standard ──────────────────────────────────────────────────────
    logger.info("Routing doc_id=%s → standard ingestion pipeline", item.id)
    return await add_item(item)
