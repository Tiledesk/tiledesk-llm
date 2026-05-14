"""
PDF page-level classifier using PyMuPDF.

Determines whether a PDF is native-digital, purely scanned, or mixed by
checking how much selectable text each page contains.  The classification
drives the smart routing in logic.py: native PDFs take the fast PyMuPDF
path, scanned/mixed ones fall through to Docling for OCR.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_MIN_NATIVE_CHARS = 50  # pages with fewer selectable chars are treated as scanned


def classify_pdf(path: str) -> Dict:
    """
    Classify a PDF as 'native', 'scanned', or 'mixed'.

    Returns::

        {
            "doc_type": "native" | "scanned" | "mixed",
            "total_pages": int,
            "scan_ratio": float,   # fraction of scanned pages (0.0 = all native)
            "pages": [
                {"page": 0, "is_native": True, "char_count": 1234},
                ...
            ],
        }

    Raises:
        ImportError: if PyMuPDF is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF classification. "
            "Install with: pip install pymupdf"
        ) from e

    doc = fitz.open(path)
    total = len(doc)
    pages: List[Dict] = []

    for page_no in range(total):
        page = doc.load_page(page_no)
        text = page.get_text().strip()
        char_count = len(text)
        pages.append({
            "page": page_no,
            "is_native": char_count >= _MIN_NATIVE_CHARS,
            "char_count": char_count,
        })

    doc.close()

    scanned_count = sum(1 for p in pages if not p["is_native"])
    scan_ratio = scanned_count / total if total > 0 else 0.0

    if scan_ratio == 0.0:
        doc_type = "native"
    elif scan_ratio >= 1.0:
        doc_type = "scanned"
    else:
        doc_type = "mixed"

    logger.debug(
        "[pdf_classifier] %s: doc_type=%s, pages=%d, scan_ratio=%.2f",
        path, doc_type, total, scan_ratio,
    )

    return {
        "doc_type": doc_type,
        "total_pages": total,
        "scan_ratio": scan_ratio,
        "pages": pages,
    }
