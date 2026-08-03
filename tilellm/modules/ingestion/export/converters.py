"""
Source (text/bytes/file) -> ExtractedDocument, one function per DocumentType.

txt/md/csv/xlsx are pure and dependency-light (pandas + openpyxl only — no
`tabulate`, which is not guaranteed to be installed everywhere, see
memory/pdf_ocr_native_md_converter_seam). pdf/docx delegate to the existing
heavy engines (docling seam from pdf_ocr, StructuredDocxLoader) and accept an
injected engine for testability (dependency inversion — the real engine is
lazy-imported so importing this module never requires docling/python-docx).
"""
import asyncio
import io
import logging
from typing import Any, Awaitable, Callable, List, Optional

from tilellm.modules.ingestion.export.models import Block, ExtractedDocument
from tilellm.modules.ingestion.export.serializers import from_md

logger = logging.getLogger(__name__)


def convert_txt(content: str, *, resource: Optional[str] = None) -> ExtractedDocument:
    return ExtractedDocument(
        type="Text Document", resource=resource,
        blocks=[Block(content=content.strip())],
    )


def convert_md(content: str, *, resource: Optional[str] = None) -> ExtractedDocument:
    """Parse existing frontmatter as-is; wrap plain markdown into a single block."""
    if content.lstrip().startswith("---\n"):
        try:
            return from_md(content)
        except ValueError:
            pass
    return ExtractedDocument(
        type="Markdown Document", resource=resource,
        blocks=[Block(content=content.strip())],
    )


def _dataframe_to_markdown(df) -> str:
    """Hand-rolled markdown table — avoids the `tabulate` dependency that
    `DataFrame.to_markdown()` requires."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
    ]
    for _, row in df.iterrows():
        cells = ["" if v is None or (isinstance(v, float) and v != v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def convert_csv(content: bytes, *, resource: Optional[str] = None, delimiter: str = ",") -> ExtractedDocument:
    import pandas as pd

    df = pd.read_csv(io.BytesIO(content), delimiter=delimiter)
    return ExtractedDocument(
        type="Tabular Document", resource=resource,
        blocks=[Block(content=_dataframe_to_markdown(df), block_type="table")],
    )


def convert_xlsx(content: bytes, *, resource: Optional[str] = None) -> ExtractedDocument:
    import pandas as pd

    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, engine="openpyxl")
    blocks = [
        Block(
            content=_dataframe_to_markdown(df),
            block_type="table",
            heading_path=name,
            order=i,
        )
        for i, (name, df) in enumerate(sheets.items())
    ]
    return ExtractedDocument(type="Tabular Document", resource=resource, blocks=blocks)


PdfConverterFn = Callable[..., Awaitable[Any]]
PdfClassifierFn = Callable[[str], dict]


async def convert_pdf(
    file_path: str,
    doc_id: str,
    *,
    resource: Optional[str] = None,
    skip_ocr: Optional[bool] = None,
    converter: Optional[PdfConverterFn] = None,
    classifier: Optional[PdfClassifierFn] = None,
) -> ExtractedDocument:
    """Delegate to the docling converter seam (pdf_ocr.converter_registry).

    Structure always comes from Docling; only OCR is decided per document.
    `skip_ocr=None` (default) classifies the PDF (native/scanned/mixed, via
    pdf_ocr.pdf_classifier) and skips OCR only for a fully native-digital
    file — never a fixed default, which would silently drop scanned content
    on non-native PDFs (see docs/MIGLIORIE_DA_FARE.md, "UPGRADE" section).
    Pass skip_ocr explicitly to override the classifier.
    """
    if skip_ocr is None:
        if classifier is None:
            from tilellm.modules.pdf_ocr.services.pdf_classifier import classify_pdf
            classifier = classify_pdf
        try:
            classification = await asyncio.to_thread(classifier, file_path)
            skip_ocr = classification["doc_type"] == "native"
        except Exception as e:
            logger.warning(f"PDF classification failed for {doc_id}, defaulting to OCR on: {e}")
            skip_ocr = False

    if converter is None:
        from tilellm.modules.pdf_ocr.services.converter_registry import get_converter
        converter = get_converter("docling")

    result = await converter(file_path, doc_id, options={"skip_ocr": skip_ocr})
    blocks = [
        Block(content=md, block_type="page", page=page_no, order=i)
        for i, (page_no, md) in enumerate(result.page_bodies)
    ]
    return ExtractedDocument(type="PDF Document", resource=resource, blocks=blocks)


async def convert_url(
    source: str,
    *,
    scrape_type: int = 0,
    parameters_scrape_type_4: Optional[Any] = None,
    browser_headers: Optional[dict] = None,
    resource: Optional[str] = None,
    fetch: Optional[Callable[..., Awaitable[Any]]] = None,
) -> ExtractedDocument:
    """Delegate to the existing web-scraping engine (get_content_by_url) — same
    Trafilatura/Playwright/Chromium strategies used by add_item's url branch."""
    if fetch is None:
        from tilellm.tools.document_tools import get_content_by_url
        fetch = get_content_by_url

    docs = await fetch(
        source, scrape_type,
        parameters_scrape_type_4=parameters_scrape_type_4,
        browser_headers=browser_headers,
    )
    blocks = [
        Block(content=d.page_content, block_type="text", order=i)
        for i, d in enumerate(docs)
    ]
    return ExtractedDocument(type="Web Page", resource=resource or source, blocks=blocks)


def convert_docx(
    source: str,
    *,
    resource: Optional[str] = None,
    loader_cls: Optional[type] = None,
) -> ExtractedDocument:
    """Delegate to StructuredDocxLoader (already used by docx_processor).

    DOCX has no real page boundaries (python-docx exposes paragraphs, not
    pagination) — `Block.page` stays unset. The closest stable provenance is
    `heading_path` (section) + `position` (paragraph/table index), both already
    extracted by the loader.
    """
    if loader_cls is None:
        from tilellm.tools.structured_loaders import StructuredDocxLoader
        loader_cls = StructuredDocxLoader

    loader = loader_cls(source)
    docs, _images = loader.load_with_images()
    blocks: List[Block] = [
        Block(
            content=d.page_content,
            block_type=d.metadata.get("element_type", "text"),
            heading_path=d.metadata.get("heading_path") or None,
            position=d.metadata.get("_para_index", d.metadata.get("table_index")),
            order=i,
        )
        for i, d in enumerate(docs)
    ]
    return ExtractedDocument(type="Word Document", resource=resource, blocks=blocks)
