"""
DocumentType — canonical enum for all supported ingestion document types.

Used by ItemSingle.type and the unified ingestion router.
Extends str so that existing string comparisons (== "pdf", in [...]) remain valid.
"""

from enum import Enum


class DocumentType(str, Enum):
    """Supported document types for the /api/ingestion endpoint.

    Pass ``auto`` (or omit ``type`` entirely) to let the system detect the
    format automatically from magic bytes, file extension, or URL heuristics.
    """

    AUTO = "auto"
    """Auto-detect from source URL, file_name, or file_content magic bytes."""

    URL = "url"
    """Remote web page — fetched with Trafilatura / Playwright."""

    PDF = "pdf"
    """PDF document.  ``use_ocr=True`` activates the Docling advanced pipeline."""

    DOCX = "docx"
    """Word document (.docx).  ``use_ocr=True`` activates the image-aware pipeline."""

    TEXT = "text"
    """Direct text content passed in the ``content`` field — embedded as-is into the
    vector store without fetching from any URL or file.  The sub-format (plain,
    Markdown, tabular) is auto-detected at chunk time by ``process_auto_detected_text``."""

    TXT = "txt"
    """Plain-text file fetched from ``source`` URL or path."""

    MD = "md"
    """Markdown document."""

    XLSX = "xlsx"
    """Excel workbook (.xlsx)."""

    XLS = "xls"
    """Legacy Excel workbook (.xls)."""

    CSV = "csv"
    """CSV file."""

    REGEX_CUSTOM = "regex_custom"
    """Custom regex-based chunking on a remote text/HTML resource."""
