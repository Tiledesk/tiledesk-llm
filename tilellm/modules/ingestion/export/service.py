"""
export_document(request) — resolve the document type, dispatch to the right
converter (models.py has none of this logic: SRP), apply frontmatter overrides.

No vector store, no chunking-for-retrieval decisions here — this is the
"just get me the structured markdown/json" half of the redesign. The other
half (frontmatter -> real ingestion) is a separate endpoint (F2).
"""
import base64
import logging
import tempfile
from typing import Optional

import httpx

from tilellm.models.document_type import DocumentType
from tilellm.modules.ingestion.export.converters import (
    convert_csv,
    convert_docx,
    convert_md,
    convert_pdf,
    convert_txt,
    convert_url,
    convert_xlsx,
)
from tilellm.modules.ingestion.export.models import ExportMdRequest, ExtractedDocument
from tilellm.modules.ingestion.type_detector import resolve_item_type

logger = logging.getLogger(__name__)

_TEXT_LIKE = {DocumentType.TEXT, DocumentType.TXT}


async def export_document(request: ExportMdRequest) -> ExtractedDocument:
    resolved = resolve_item_type(
        current_type=request.type,
        source=request.source,
        content=request.content,
        file_name=request.file_name,
        file_content=request.file_content,
    )
    if resolved is None:
        raise ValueError(
            "Impossibile determinare il tipo di documento: fornire 'source', "
            "'content' o 'file_content'."
        )

    # Document identity must never be silently lost: fall back to file_name when
    # there's no source URL (e.g. upload via file_content base64 + file_name).
    resource = request.source or request.file_name or None

    if resolved in _TEXT_LIKE:
        doc = convert_txt(await _resolve_text(request), resource=resource)
    elif resolved == DocumentType.MD:
        doc = convert_md(await _resolve_text(request), resource=resource)
    elif resolved == DocumentType.CSV:
        doc = convert_csv(await _resolve_bytes(request), resource=resource)
    elif resolved in (DocumentType.XLSX, DocumentType.XLS):
        doc = convert_xlsx(await _resolve_bytes(request), resource=resource)
    elif resolved == DocumentType.PDF:
        path = await _download_to_temp_file(request, suffix=".pdf")
        doc = await convert_pdf(path, request.file_name or "export", resource=resource)
    elif resolved == DocumentType.DOCX:
        path = await _download_to_temp_file(request, suffix=".docx")
        doc = convert_docx(path, resource=resource)
    elif resolved == DocumentType.URL:
        if not _is_http_url(request.source):
            raise ValueError("Export type=url richiede un 'source' http/https.")
        doc = await convert_url(
            request.source,
            scrape_type=request.scrape_type,
            parameters_scrape_type_4=request.parameters_scrape_type_4,
            browser_headers=request.browser_headers,
            resource=resource,
        )
    else:
        # ponytail: regex_custom/xls-legacy-quirks deferred — add a converter
        # here when a concrete use case needs it (see memory/ingestion_md_redesign).
        raise ValueError(f"Export non ancora supportato per il tipo '{resolved.value}'.")

    return _apply_overrides(doc, request)


def _apply_overrides(doc: ExtractedDocument, request: ExportMdRequest) -> ExtractedDocument:
    updates = {}
    if request.title:
        updates["title"] = request.title
    if request.description:
        updates["description"] = request.description
    if request.tags:
        updates["tags"] = request.tags
    return doc.model_copy(update=updates) if updates else doc


def _is_http_url(value: Optional[str]) -> bool:
    return bool(value) and value.startswith(("http://", "https://"))


async def _resolve_text(request: ExportMdRequest) -> str:
    if request.content:
        return request.content
    return (await _resolve_bytes(request)).decode("utf-8", errors="replace")


async def _resolve_bytes(request: ExportMdRequest) -> bytes:
    if request.content:
        return request.content.encode("utf-8")
    if request.file_content and not _is_http_url(request.file_content):
        return base64.b64decode(request.file_content)
    source = request.file_content if _is_http_url(request.file_content) else request.source
    if _is_http_url(source):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(source)
            response.raise_for_status()
            return response.content
    raise ValueError("Nessun contenuto disponibile: fornire 'content', 'file_content' o un 'source' URL.")


async def _download_to_temp_file(request: ExportMdRequest, suffix: str) -> str:
    """Resolve source/file_content to a local file path (URL download only in F1;
    base64 blobs for PDF/DOCX are a ceiling — add when a real use case needs it)."""
    url = request.source if _is_http_url(request.source) else request.file_content
    if not _is_http_url(url):
        raise ValueError(
            f"Export {suffix} richiede un 'source' URL http/https "
            "(base64 diretto non ancora supportato per questo tipo)."
        )
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response.content)
            return tmp.name
