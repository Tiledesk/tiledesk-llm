"""
Unit tests for tilellm/modules/ingestion/type_detector.py

All tests are pure-function: no network, no GPU, no external services.
"""

import base64
import pytest

from tilellm.models.document_type import DocumentType
from tilellm.modules.ingestion.type_detector import (
    detect_document_type,
    resolve_item_type,
    _magic_from_base64,
    _ext_from_path,
)


# ---------------------------------------------------------------------------
# Helpers to build valid base64 magic-byte prefixes
# ---------------------------------------------------------------------------

def _b64(raw_bytes: bytes) -> str:
    """Encode bytes to base64 string (padded to at least 12 chars)."""
    b64 = base64.b64encode(raw_bytes).decode()
    # Pad with trailing 'A' characters so length >= 12 for the decoder
    while len(b64) < 12:
        b64 += "A"
    return b64


_B64_PDF = _b64(b"%PDF-1.7 extra bytes here")
_B64_OLE2 = _b64(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1")
_B64_ZIP = _b64(b"PK\x03\x04\x14\x00\x00\x00")
_B64_RAND = _b64(b"\x00\x01\x02\x03\x04\x05\x06\x07")


# ---------------------------------------------------------------------------
# _magic_from_base64
# ---------------------------------------------------------------------------

class TestMagicFromBase64:
    def test_pdf_magic(self):
        assert _magic_from_base64(_B64_PDF) == DocumentType.PDF

    def test_ole2_magic_returns_xls(self):
        assert _magic_from_base64(_B64_OLE2) == DocumentType.XLS

    def test_zip_magic_returns_none(self):
        """ZIP magic is ambiguous (DOCX vs XLSX) — must fall through to extension."""
        assert _magic_from_base64(_B64_ZIP) is None

    def test_unknown_magic_returns_none(self):
        assert _magic_from_base64(_B64_RAND) is None

    def test_too_short_returns_none(self):
        assert _magic_from_base64("abc") is None

    def test_invalid_base64_returns_none(self):
        assert _magic_from_base64("!!!not-base64!!!") is None

    def test_url_string_not_called_as_b64(self):
        """Callers must guard against URL strings; we confirm decode raises gracefully."""
        result = _magic_from_base64("https://example.com/file.pdf")
        # May return None (invalid b64) or PDF if the string accidentally decodes
        # to %PDF — either is acceptable; must not raise.
        assert result in (None, DocumentType.PDF)


# ---------------------------------------------------------------------------
# _ext_from_path
# ---------------------------------------------------------------------------

class TestExtFromPath:
    @pytest.mark.parametrize("path,expected", [
        ("document.pdf",                   DocumentType.PDF),
        ("/path/to/report.PDF",            DocumentType.PDF),   # case-insensitive
        ("https://example.com/file.docx",  DocumentType.DOCX),
        ("data.doc",                       DocumentType.DOCX),  # old Word → DOCX
        ("notes.txt",                      DocumentType.TXT),
        ("README.md",                      DocumentType.MD),
        ("guide.markdown",                 DocumentType.MD),
        ("workbook.xlsx",                  DocumentType.XLSX),
        ("legacy.xls",                     DocumentType.XLS),
        ("data.csv",                       DocumentType.CSV),
        ("file.pdf?v=2&token=abc",         DocumentType.PDF),   # query string stripped
        ("file.pdf#section",               DocumentType.PDF),   # fragment stripped
    ])
    def test_known_extensions(self, path, expected):
        assert _ext_from_path(path) == expected

    @pytest.mark.parametrize("path", [
        "https://example.com/page",
        "https://example.com/page.html",
        "https://example.com/",
        "unknown.xyz",
        "",
    ])
    def test_unknown_or_web_extensions_return_none(self, path):
        assert _ext_from_path(path) is None


# ---------------------------------------------------------------------------
# detect_document_type
# ---------------------------------------------------------------------------

class TestDetectDocumentType:

    # ── Magic bytes ─────────────────────────────────────────────────────────

    def test_pdf_magic_bytes_beats_extension(self):
        """Magic bytes take priority over extension (catches mislabelled files)."""
        result = detect_document_type(
            file_name="report.txt",   # wrong extension
            file_content=_B64_PDF,
        )
        assert result == DocumentType.PDF

    def test_ole2_magic_bytes(self):
        result = detect_document_type(file_content=_B64_OLE2)
        assert result == DocumentType.XLS

    def test_zip_magic_falls_through_to_extension_docx(self):
        """ZIP magic → use file_name extension to pick DOCX vs XLSX."""
        result = detect_document_type(
            file_name="contract.docx",
            file_content=_B64_ZIP,
        )
        assert result == DocumentType.DOCX

    def test_zip_magic_falls_through_to_extension_xlsx(self):
        result = detect_document_type(
            file_name="data.xlsx",
            file_content=_B64_ZIP,
        )
        assert result == DocumentType.XLSX

    # ── file_name extension ──────────────────────────────────────────────────

    def test_file_name_pdf(self):
        result = detect_document_type(file_name="invoice.pdf")
        assert result == DocumentType.PDF

    def test_file_name_csv(self):
        result = detect_document_type(file_name="export.csv")
        assert result == DocumentType.CSV

    # ── source URL extension ─────────────────────────────────────────────────

    def test_source_url_with_pdf_extension(self):
        result = detect_document_type(source="https://example.com/docs/report.pdf")
        assert result == DocumentType.PDF

    def test_source_url_with_xlsx_extension(self):
        result = detect_document_type(source="https://files.example.com/data.xlsx")
        assert result == DocumentType.XLSX

    def test_source_url_with_md_extension(self):
        result = detect_document_type(source="https://raw.githubusercontent.com/org/repo/main/README.md")
        assert result == DocumentType.MD

    def test_source_url_with_docx_and_query_string(self):
        result = detect_document_type(
            source="https://example.com/download/contract.docx?token=xyz"
        )
        assert result == DocumentType.DOCX

    # ── URL heuristic ────────────────────────────────────────────────────────

    def test_http_url_no_extension_returns_url_type(self):
        result = detect_document_type(source="https://example.com/blog/post")
        assert result == DocumentType.URL

    def test_http_url_html_extension_returns_url_type(self):
        result = detect_document_type(source="https://example.com/page.html")
        assert result == DocumentType.URL

    def test_http_url_root_path_returns_url_type(self):
        result = detect_document_type(source="https://example.com/")
        assert result == DocumentType.URL

    def test_http_url_asp_extension_returns_url_type(self):
        result = detect_document_type(source="http://legacy.example.com/report.aspx")
        assert result == DocumentType.URL

    # ── Direct text content (type=text) ─────────────────────────────────────

    def test_content_only_returns_text(self):
        """No source, no file — content passed directly → TEXT."""
        result = detect_document_type(content="Some raw text here")
        assert result == DocumentType.TEXT

    def test_content_markdown_returns_text(self):
        """Even markdown content passed directly → TEXT (sub-format detected later)."""
        result = detect_document_type(content="# Heading\nSome text")
        assert result == DocumentType.TEXT

    def test_all_none_returns_none(self):
        """Truly no signal → None."""
        result = detect_document_type()
        assert result is None

    # ── file_content as URL ──────────────────────────────────────────────────

    def test_file_content_as_url_with_pdf_extension(self):
        result = detect_document_type(
            file_content="https://example.com/manual.pdf"
        )
        assert result == DocumentType.PDF

    def test_file_content_as_url_no_extension(self):
        result = detect_document_type(
            file_content="https://example.com/landing"
        )
        assert result == DocumentType.URL


# ---------------------------------------------------------------------------
# resolve_item_type
# ---------------------------------------------------------------------------

class TestResolveItemType:

    def test_explicit_type_kept(self):
        """If caller provides an explicit non-auto type, respect it."""
        result = resolve_item_type(
            current_type=DocumentType.PDF,
            source="https://example.com/file.xlsx",  # would detect XLSX if auto
        )
        assert result == DocumentType.PDF

    def test_auto_triggers_detection(self):
        result = resolve_item_type(
            current_type=DocumentType.AUTO,
            source="https://example.com/report.pdf",
        )
        assert result == DocumentType.PDF

    def test_none_triggers_detection(self):
        result = resolve_item_type(
            current_type=None,
            source="https://example.com/data.csv",
        )
        assert result == DocumentType.CSV

    def test_none_with_content_returns_text(self):
        """type=None + content only → resolve to TEXT (direct-content path)."""
        result = resolve_item_type(
            current_type=None,
            content="Hello world",
        )
        assert result == DocumentType.TEXT

    def test_auto_with_content_returns_text(self):
        """type=auto + content only → resolve to TEXT."""
        result = resolve_item_type(
            current_type=DocumentType.AUTO,
            content="Hello world",
        )
        assert result == DocumentType.TEXT

    def test_none_no_signal_returns_none(self):
        """type=None, no content, no source → truly no signal → None."""
        result = resolve_item_type(current_type=None)
        assert result is None

    def test_explicit_text_kept(self):
        """type=text explicit → keep (no detection needed)."""
        result = resolve_item_type(
            current_type=DocumentType.TEXT,
            source="https://example.com/page",  # source is ignored when type is explicit
        )
        assert result == DocumentType.TEXT

    def test_explicit_regex_custom_kept(self):
        result = resolve_item_type(
            current_type=DocumentType.REGEX_CUSTOM,
            source="https://example.com/page",
        )
        assert result == DocumentType.REGEX_CUSTOM
