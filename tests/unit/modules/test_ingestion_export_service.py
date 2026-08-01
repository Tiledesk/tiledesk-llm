#!/usr/bin/env python3
"""
export_document(request) — dispatch by resolved DocumentType to the right
converter, then apply frontmatter overrides (title/description/tags).
No vector store involved — this is the F1 "just export" endpoint's logic.
"""
import base64
from unittest.mock import AsyncMock, patch

import pytest

from tilellm.models.document_type import DocumentType
from tilellm.modules.ingestion.export.models import ExportMdRequest
from tilellm.modules.ingestion.export.service import export_document


class TestTextLikeDispatch:
    @pytest.mark.asyncio
    async def test_txt_inline_content(self):
        req = ExportMdRequest(type=DocumentType.TXT, content="hello world")
        doc = await export_document(req)
        assert doc.type == "Text Document"
        assert doc.blocks[0].content == "hello world"

    @pytest.mark.asyncio
    async def test_md_inline_content(self):
        req = ExportMdRequest(type=DocumentType.MD, content="# Title\n\nBody")
        doc = await export_document(req)
        assert doc.type == "Markdown Document"

    @pytest.mark.asyncio
    async def test_auto_detects_type_from_file_name(self):
        req = ExportMdRequest(content="hello", file_name="note.txt")
        doc = await export_document(req)
        assert doc.type == "Text Document"

    @pytest.mark.asyncio
    async def test_base64_file_content_decoded_for_txt(self):
        b64 = base64.b64encode(b"decoded content").decode()
        req = ExportMdRequest(type=DocumentType.TXT, file_content=b64)
        doc = await export_document(req)
        assert doc.blocks[0].content == "decoded content"

    @pytest.mark.asyncio
    async def test_no_signal_raises(self):
        req = ExportMdRequest()
        with pytest.raises(ValueError):
            await export_document(req)


class TestCsvXlsxDispatch:
    @pytest.mark.asyncio
    async def test_csv_from_base64_file_content(self):
        b64 = base64.b64encode(b"a,b\n1,2\n").decode()
        req = ExportMdRequest(type=DocumentType.CSV, file_content=b64)
        doc = await export_document(req)
        assert doc.type == "Tabular Document"
        assert doc.blocks[0].block_type == "table"


class TestFrontmatterOverrides:
    @pytest.mark.asyncio
    async def test_title_description_tags_applied(self):
        req = ExportMdRequest(
            type=DocumentType.TXT, content="hi",
            title="My title", description="My desc", tags=["a", "b"],
        )
        doc = await export_document(req)
        assert doc.title == "My title"
        assert doc.description == "My desc"
        assert doc.tags == ["a", "b"]

    @pytest.mark.asyncio
    async def test_resource_defaults_to_source(self):
        req = ExportMdRequest(type=DocumentType.TXT, content="hi", source="https://x/y.txt")
        doc = await export_document(req)
        assert doc.resource == "https://x/y.txt"

    @pytest.mark.asyncio
    async def test_resource_falls_back_to_file_name_without_source(self):
        """Document identity must never be silently lost: file uploaded via
        file_content (base64) + file_name, with no source URL, must still carry
        an identity in `resource` — every downstream chunk needs to know 'which
        document' it came from."""
        b64 = base64.b64encode(b"hello").decode()
        req = ExportMdRequest(type=DocumentType.TXT, file_content=b64, file_name="capitolato.txt")
        doc = await export_document(req)
        assert doc.resource == "capitolato.txt"

    @pytest.mark.asyncio
    async def test_resource_none_when_neither_source_nor_file_name(self):
        req = ExportMdRequest(type=DocumentType.TXT, content="hi")
        doc = await export_document(req)
        assert doc.resource is None


class TestPdfDocxDispatch:
    @pytest.mark.asyncio
    async def test_pdf_downloads_url_then_converts(self):
        fake_doc = AsyncMock()
        with patch(
            "tilellm.modules.ingestion.export.service.convert_pdf",
            new=AsyncMock(return_value="CONVERTED"),
        ) as mock_convert, patch(
            "tilellm.modules.ingestion.export.service._download_to_temp_file",
            new=AsyncMock(return_value="/tmp/fake.pdf"),
        ) as mock_download:
            req = ExportMdRequest(type=DocumentType.PDF, source="https://x/doc.pdf")
            result = await export_document(req)

        mock_download.assert_awaited_once()
        mock_convert.assert_awaited_once()
        assert result == "CONVERTED"

    @pytest.mark.asyncio
    async def test_pdf_without_url_source_raises(self):
        req = ExportMdRequest(type=DocumentType.PDF, content="not a url")
        with pytest.raises(ValueError):
            await export_document(req)

    @pytest.mark.asyncio
    async def test_docx_downloads_url_then_converts(self):
        with patch(
            "tilellm.modules.ingestion.export.service.convert_docx",
            return_value="CONVERTED_DOCX",
        ) as mock_convert, patch(
            "tilellm.modules.ingestion.export.service._download_to_temp_file",
            new=AsyncMock(return_value="/tmp/fake.docx"),
        ):
            req = ExportMdRequest(type=DocumentType.DOCX, source="https://x/doc.docx")
            result = await export_document(req)

        mock_convert.assert_called_once()
        assert result == "CONVERTED_DOCX"


class TestUnsupportedType:
    @pytest.mark.asyncio
    async def test_url_type_not_yet_supported(self):
        # ponytail ceiling: web-page scraping into ExtractedDocument deferred to F1.1
        req = ExportMdRequest(type=DocumentType.URL, source="https://example.com")
        with pytest.raises(ValueError):
            await export_document(req)
