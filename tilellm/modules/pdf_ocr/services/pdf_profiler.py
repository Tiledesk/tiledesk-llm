"""
Cheap pre-flight PDF profiling with PyMuPDF.

Produces the signals needed by the conversion strategy selector:
page count, file size, and an estimate of content density (a proxy for
how expensive Docling table-structure recognition will be).

A full profile pass on a 500-page PDF costs well under a second —
negligible compared to a Docling conversion.
"""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# A document is considered "heavy" (→ segmented conversion) above any of these.
SEGMENT_PAGE_THRESHOLD = int(os.environ.get("PDF_SEGMENT_PAGE_THRESHOLD", "30"))
SEGMENT_SIZE_MB_THRESHOLD = float(os.environ.get("PDF_SEGMENT_SIZE_MB", "20"))
# Words per page above which a page is counted as "dense" (huge tables produce
# thousands of short words per page; normal prose is ~300-600).
DENSE_PAGE_WORDS = int(os.environ.get("PDF_DENSE_PAGE_WORDS", "1500"))


@dataclass
class PdfProfile:
    """Static facts about a PDF, gathered before any conversion."""
    num_pages: int = 0
    file_size_mb: float = 0.0
    dense_pages: int = 0          # pages whose word count exceeds DENSE_PAGE_WORDS
    max_words_per_page: int = 0
    error: str = ""               # non-empty when profiling itself failed

    @property
    def needs_segmentation(self) -> bool:
        return (
            self.num_pages > SEGMENT_PAGE_THRESHOLD
            or self.file_size_mb > SEGMENT_SIZE_MB_THRESHOLD
            or self.dense_pages > 5
        )


def profile_pdf(file_path: str) -> PdfProfile:
    """Profile a PDF synchronously (call via asyncio.to_thread from async code)."""
    profile = PdfProfile()
    try:
        profile.file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    except OSError as e:
        profile.error = f"stat failed: {e}"
        return profile

    try:
        import fitz
        doc = fitz.open(file_path)
        profile.num_pages = len(doc)
        for page_no in range(profile.num_pages):
            try:
                words = len(doc.load_page(page_no).get_text("words"))
            except Exception:
                continue
            profile.max_words_per_page = max(profile.max_words_per_page, words)
            if words > DENSE_PAGE_WORDS:
                profile.dense_pages += 1
        doc.close()
    except Exception as e:
        profile.error = f"profiling failed: {e}"
        logger.warning(f"pdf_profiler: {profile.error} for {file_path}")

    logger.info(
        f"pdf_profiler: pages={profile.num_pages} size={profile.file_size_mb:.1f}MB "
        f"dense_pages={profile.dense_pages} max_words/page={profile.max_words_per_page} "
        f"needs_segmentation={profile.needs_segmentation}"
    )
    return profile
