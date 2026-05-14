"""
Fast PDF processor for native-digital documents using PyMuPDF.

Extracts text blocks and tables without OCR.  Significantly faster than
Docling for PDFs with selectable text (no 2-4 GB model load, no GPU needed).

Output format matches ProductionDocumentProcessor._process_pdf_docling so
all downstream _index_* functions in logic.py work unchanged:

    {
        "text_elements": [{"id", "text", "page", "bbox", "type"}, ...],
        "tables":        [{"id", "data" (DataFrame), "page", "bbox",
                           "caption", "surrounding_text",
                           "parquet_path", "md_path"}, ...],
        "images":        [],   # not extracted — no PIL data, no captioning
        "formulas":      [],
        "metadata":      {"doc_id": str, "num_pages": int},
    }

Images are intentionally empty: native-digital images embedded in PDFs
are vector/raster art that cannot be captioned without a multimodal model
call on extracted pixel data.  If image indexing is needed, use
strategy='quality' (Docling pipeline).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


async def process_pdf_native(file_path: str, doc_id: str) -> Dict[str, Any]:
    """
    Async entry point — runs the blocking PyMuPDF extraction in a thread.
    """
    return await asyncio.to_thread(_extract_native, file_path, doc_id)


def _extract_native(file_path: str, doc_id: str) -> Dict[str, Any]:
    try:
        import fitz
    except ImportError as e:
        raise ImportError(
            "PyMuPDF (fitz) is required for the fast PDF path. "
            "Install with: pip install pymupdf"
        ) from e

    doc = fitz.open(file_path)
    num_pages = len(doc)

    text_elements: List[Dict] = []
    tables: List[Dict] = []

    for page_no in range(num_pages):
        page = doc.load_page(page_no)

        # ---- Text blocks -------------------------------------------------------
        try:
            blocks = page.get_text(
                "dict", flags=fitz.TEXT_PRESERVE_WHITESPACE
            ).get("blocks", [])
        except Exception as exc:
            logger.warning("[pdf_native] get_text failed on page %d: %s", page_no, exc)
            blocks = []

        for block_idx, block in enumerate(blocks):
            if block.get("type") != 0:  # 0=text, 1=image raster
                continue
            parts = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span.get("text", "").strip()
                    if t:
                        parts.append(t)
            text = " ".join(parts).strip()
            if not text:
                continue
            raw_bbox = block.get("bbox")
            text_elements.append({
                "id": f"{doc_id}_text_{page_no}_{block_idx}",
                "text": text,
                "page": page_no,
                "bbox": tuple(raw_bbox) if raw_bbox else None,
                "type": "text",
            })

        # ---- Tables (PyMuPDF ≥ 1.23) ------------------------------------------
        try:
            finder = page.find_tables()
            for tbl_idx, tbl in enumerate(finder.tables):
                data = tbl.extract()
                if not data:
                    continue
                # First row → header; remaining → body
                header = [
                    str(c) if c else f"col_{i}"
                    for i, c in enumerate(data[0])
                ]
                rows = data[1:] if len(data) > 1 else []
                df = pd.DataFrame(rows, columns=header)
                raw_bbox = tbl.bbox
                tables.append({
                    "id": f"{doc_id}_table_{page_no}_{tbl_idx}",
                    "data": df,
                    "page": page_no,
                    "bbox": tuple(raw_bbox) if raw_bbox else None,
                    "caption": None,
                    "surrounding_text": "",
                    "parquet_path": "",
                    "md_path": "",
                })
        except AttributeError:
            # find_tables() not present in PyMuPDF < 1.23 — skip silently
            logger.debug(
                "[pdf_native] find_tables() not available, skipping table extraction (page %d)",
                page_no,
            )
        except Exception as exc:
            logger.warning(
                "[pdf_native] table extraction failed on page %d: %s", page_no, exc
            )

    doc.close()

    logger.info(
        "[pdf_native] doc_id=%s: %d pages, %d text blocks, %d tables",
        doc_id, num_pages, len(text_elements), len(tables),
    )

    return {
        "text_elements": text_elements,
        "tables": tables,
        "images": [],
        "formulas": [],
        "metadata": {
            "doc_id": doc_id,
            "num_pages": num_pages,
        },
    }
