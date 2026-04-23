"""
Document-type auto-detection for the unified ingestion pipeline.

Detection priority (first match wins):
  1. Magic bytes — PDF (``%PDF``), OLE2 / legacy XLS (``\\xD0\\xCF\\x11\\xE0``).
     ZIP magic (``PK\\x03\\x04``) is DOCX or XLSX: ambiguous, falls through to
     extension check so the filename disambiguates.
  2. Extension from ``file_name`` (e.g. ``report.pdf``).
  3. Extension from ``source`` URL path (strips query-string first).
  4. URL heuristic — ``http(s)`` source with no known file extension → ``url``.
  5. Fallback → ``txt`` (plain-text / direct-content path).

All detection is deterministic and zero-cost (no network, no LLM).
"""

from __future__ import annotations

import base64
import os
from typing import Optional
from urllib.parse import urlparse

from tilellm.models.document_type import DocumentType


# ---------------------------------------------------------------------------
# Binary magic-byte signatures
# ---------------------------------------------------------------------------

_MAGIC_PDF = b"%PDF"                        # PDF
_MAGIC_OLE2 = b"\xD0\xCF\x11\xE0"          # Legacy Office (XLS, old DOC …)
_MAGIC_ZIP = b"PK\x03\x04"                 # Office Open XML: DOCX / XLSX

# Minimum base64 chars needed to decode 8 bytes  (8 * 4/3 ≈ 11, rounded up)
_MIN_B64_CHARS = 12

# ---------------------------------------------------------------------------
# Extension → DocumentType map
# ---------------------------------------------------------------------------

_EXT_MAP: dict[str, DocumentType] = {
    ".pdf":      DocumentType.PDF,
    ".docx":     DocumentType.DOCX,
    ".doc":      DocumentType.DOCX,   # old Word → same loader
    ".txt":      DocumentType.TXT,
    ".md":       DocumentType.MD,
    ".markdown": DocumentType.MD,
    ".xlsx":     DocumentType.XLSX,
    ".xls":      DocumentType.XLS,
    ".csv":      DocumentType.CSV,
}

# Extensions that belong to web pages (not downloadable files)
_WEB_EXTS = {".html", ".htm", ".php", ".asp", ".aspx", ".jsp"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _magic_from_base64(file_content: str) -> Optional[DocumentType]:
    """Return a DocumentType inferred from the first bytes of a base64 blob.

    Returns ``None`` when:
    * the content is too short to decode reliably,
    * the magic bytes are ambiguous (ZIP → could be DOCX or XLSX),
    * decoding fails.
    """
    if len(file_content) < _MIN_B64_CHARS:
        return None
    try:
        raw: bytes = base64.b64decode(file_content[:_MIN_B64_CHARS])[:8]
    except Exception:
        return None

    if raw[:4] == _MAGIC_PDF:
        return DocumentType.PDF
    if raw[:4] == _MAGIC_OLE2:
        # Legacy Office container — most commonly XLS in our context.
        # Old .doc files also use OLE2 but we have no separate DOC type.
        return DocumentType.XLS
    # ZIP magic (DOCX / XLSX) → ambiguous, fall through to extension check.
    return None


def _ext_from_path(path: str) -> Optional[DocumentType]:
    """Return DocumentType from the file extension of *path* (URL or filename).

    Strips query-string and fragment before extracting the extension.
    """
    # Strip query string / fragment that may follow the path component
    clean = path.split("?")[0].split("#")[0]
    _, ext = os.path.splitext(clean.lower())
    return _EXT_MAP.get(ext)


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_document_type(
    source: Optional[str] = None,
    content: Optional[str] = None,  # noqa: ARG001 — reserved for future heuristics
    file_name: Optional[str] = None,
    file_content: Optional[str] = None,
) -> Optional[DocumentType]:
    """Infer the DocumentType from available request signals.

    Returns ``DocumentType.TEXT`` when only ``content`` is present (no
    source/filename/file_content).  Returns ``None`` only when there is truly
    no signal at all.

    Args:
        source: URL or file path provided in the request.
        content: Raw text content (direct ingestion). Currently not used for
            detection; reserved for future content-based heuristics.
        file_name: Original filename with extension (e.g. ``document.pdf``).
        file_content: Base64-encoded binary content *or* a URL string.  When
            it is a URL it is treated the same as ``source`` for extension
            detection.
    """

    # ── 1. Magic bytes (unambiguous binary formats) ─────────────────────────
    if file_content and not _is_http_url(file_content):
        magic_type = _magic_from_base64(file_content)
        if magic_type is not None:
            return magic_type
        # ZIP magic is ambiguous — fall through to extension check below.

    # ── 2. Extension from file_name ─────────────────────────────────────────
    if file_name:
        t = _ext_from_path(file_name)
        if t is not None:
            return t

    # ── 3. Extension from source URL/path ───────────────────────────────────
    if source:
        parsed_path = urlparse(source).path if _is_http_url(source) else source
        t = _ext_from_path(parsed_path)
        if t is not None:
            return t

        # ── 4. URL heuristic ─────────────────────────────────────────────────
        # HTTP(S) source whose extension is either a web extension or unknown
        # → treat as a web page to scrape.
        if _is_http_url(source):
            _, ext = os.path.splitext(urlparse(source).path.lower())
            if ext in _WEB_EXTS or ext == "":
                return DocumentType.URL

    # ── 5. Fallback: file_content URL (treat as web page) ───────────────────
    if file_content and _is_http_url(file_content):
        _, ext = os.path.splitext(urlparse(file_content).path.lower())
        t = _ext_from_path(file_content)
        if t is not None:
            return t
        if ext in _WEB_EXTS or ext == "":
            return DocumentType.URL

    # ── 6. Direct text content (content provided, no binary/URL source) ──────
    if content:
        return DocumentType.TEXT

    # ── No signal at all ─────────────────────────────────────────────────────
    return None


def resolve_item_type(
    current_type: Optional[DocumentType],
    source: Optional[str] = None,
    content: Optional[str] = None,
    file_name: Optional[str] = None,
    file_content: Optional[str] = None,
) -> Optional[DocumentType]:
    """Resolve the effective DocumentType for an ingestion request.

    * If ``current_type`` is already explicit (not None and not AUTO) → keep it.
    * Otherwise run ``detect_document_type`` and return the result (may be None
      if only direct-text content is present — callers must handle this).
    """
    if current_type is not None and current_type != DocumentType.AUTO:
        return current_type

    return detect_document_type(
        source=source,
        content=content,
        file_name=file_name,
        file_content=file_content,
    )
