"""
Split a PDF into page-range segments as temporary sub-PDF files.

Used by the conversion pipeline to keep Docling's peak memory bounded:
each segment is converted independently and memory is released between
segments, so peak RSS depends on the segment size — not the document size.

PyMuPDF's insert_pdf copies pages by reference (no re-rendering), so
splitting a 500-page PDF costs a few hundred ms and minimal RAM.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class PdfSegment:
    """One page-range slice of the original PDF, written to a temp file."""
    path: str
    start_page: int  # 0-based, inclusive — offset to add to in-segment page numbers
    end_page: int    # 0-based, inclusive


def split_pdf(file_path: str, pages_per_segment: int) -> List[PdfSegment]:
    """Split synchronously (call via asyncio.to_thread from async code).

    The caller owns the returned temp files: call cleanup_segments() when done.
    """
    import fitz

    src = fitz.open(file_path)
    num_pages = len(src)
    segments: List[PdfSegment] = []
    try:
        for start in range(0, num_pages, pages_per_segment):
            end = min(start + pages_per_segment - 1, num_pages - 1)
            out = fitz.open()
            out.insert_pdf(src, from_page=start, to_page=end)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_seg{start}-{end}.pdf"
            ) as tmp:
                seg_path = tmp.name
            out.save(seg_path)
            out.close()
            segments.append(PdfSegment(path=seg_path, start_page=start, end_page=end))
    except Exception:
        cleanup_segments(segments)
        raise
    finally:
        src.close()

    logger.info(
        f"pdf_segmenter: split {num_pages} pages into {len(segments)} segments "
        f"of {pages_per_segment} pages"
    )
    return segments


def cleanup_segments(segments: List[PdfSegment]) -> None:
    """Remove all segment temp files; never raises."""
    for seg in segments:
        try:
            if os.path.exists(seg.path):
                os.remove(seg.path)
        except OSError as e:
            logger.warning(f"pdf_segmenter: could not remove {seg.path}: {e}")
