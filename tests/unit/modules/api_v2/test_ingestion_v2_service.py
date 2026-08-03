#!/usr/bin/env python3
"""
POST /api/v2/ingestion routing: use_ocr=True (pdf/docx) and regex_custom stay
on the unchanged legacy pipelines (scrape_pdf/process_docx_with_images,
add_item/add_item_hybrid); everything else goes through the canonical
export_document -> write_extracted_document path (same baseline provenance
metadata as every other ingestion path, dense-only or hybrid).
"""
from unittest.mock import AsyncMock, patch

import pytest

from tilellm.models.document_type import DocumentType
from tilellm.models.llm import ItemSingle
from tilellm.modules.api_v2.services.ingestion_v2_service import _ingest_v2_core
from tilellm.modules.ingestion.ingest.models import IngestMdResult


def _engine():
    from tilellm.models.vector_store import Engine
    return Engine(name="pinecone", type="serverless", apikey="k", vector_size=1536, index_name="idx")


def _item(**over):
    kw = dict(id="doc1", namespace="ns", engine=_engine(), source="https://x/doc.txt")
    kw.update(over)
    return ItemSingle(**kw)


class TestRoutingDecision:
    @pytest.mark.asyncio
    async def test_pdf_with_use_ocr_routes_to_legacy_ocr(self):
        with patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service._route_legacy_ocr",
            new=AsyncMock(return_value="LEGACY_OCR"),
        ) as mock_route:
            result = await _ingest_v2_core(_item(type=DocumentType.PDF, use_ocr=True), repo=None, llm_embeddings=None)

        mock_route.assert_called_once()
        assert result == "LEGACY_OCR"

    @pytest.mark.asyncio
    async def test_docx_with_use_ocr_routes_to_legacy_ocr(self):
        with patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service._route_legacy_ocr",
            new=AsyncMock(return_value="LEGACY_OCR"),
        ) as mock_route:
            await _ingest_v2_core(
                _item(source="https://x/doc.docx", type=DocumentType.DOCX, use_ocr=True), repo=None, llm_embeddings=None,
            )

        mock_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_pdf_without_use_ocr_routes_to_canonical(self):
        with patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service._route_canonical",
            new=AsyncMock(return_value="CANONICAL"),
        ) as mock_route:
            result = await _ingest_v2_core(_item(type=DocumentType.PDF, use_ocr=False), repo=None, llm_embeddings=None)

        mock_route.assert_called_once()
        assert result == "CANONICAL"

    @pytest.mark.asyncio
    async def test_regex_custom_routes_to_legacy_regex(self):
        with patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service._route_legacy_regex_custom",
            new=AsyncMock(return_value="LEGACY_REGEX"),
        ) as mock_route:
            result = await _ingest_v2_core(_item(type=DocumentType.REGEX_CUSTOM), repo=None, llm_embeddings=None)

        mock_route.assert_called_once()
        assert result == "LEGACY_REGEX"

    @pytest.mark.asyncio
    async def test_txt_routes_to_canonical(self):
        with patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service._route_canonical",
            new=AsyncMock(return_value="CANONICAL"),
        ) as mock_route:
            await _ingest_v2_core(_item(type=DocumentType.TXT), repo=None, llm_embeddings=None)

        mock_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_url_routes_to_canonical(self):
        with patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service._route_canonical",
            new=AsyncMock(return_value="CANONICAL"),
        ) as mock_route:
            await _ingest_v2_core(_item(type=DocumentType.URL, source="https://example.com"), repo=None, llm_embeddings=None)

        mock_route.assert_called_once()


class TestRouteLegacyOcr:
    @pytest.mark.asyncio
    async def test_pdf_calls_scrape_pdf(self):
        from tilellm.modules.api_v2.services.ingestion_v2_service import _route_legacy_ocr

        with patch(
            "tilellm.modules.ingestion.controllers._build_pdf_request", return_value="PDF_REQUEST",
        ), patch(
            "tilellm.modules.pdf_ocr.controllers.scrape_pdf", new=AsyncMock(return_value="SCRAPE_RESULT"),
        ) as mock_scrape:
            result = await _route_legacy_ocr(_item(type=DocumentType.PDF, use_ocr=True), DocumentType.PDF)

        mock_scrape.assert_called_once_with("PDF_REQUEST")
        assert result == "SCRAPE_RESULT"

    @pytest.mark.asyncio
    async def test_docx_calls_process_docx_with_images(self):
        from tilellm.modules.api_v2.services.ingestion_v2_service import _route_legacy_ocr

        with patch(
            "tilellm.modules.ingestion.controllers._build_pdf_request", return_value="DOCX_REQUEST",
        ), patch(
            "tilellm.modules.ingestion.docx_processor.process_docx_with_images",
            new=AsyncMock(return_value="DOCX_RESULT"),
        ) as mock_docx:
            result = await _route_legacy_ocr(
                _item(source="https://x/doc.docx", type=DocumentType.DOCX, use_ocr=True), DocumentType.DOCX,
            )

        mock_docx.assert_called_once_with("DOCX_REQUEST")
        assert result == "DOCX_RESULT"


class TestRouteLegacyRegexCustom:
    @pytest.mark.asyncio
    async def test_hybrid_false_calls_add_item(self):
        from tilellm.modules.api_v2.services.ingestion_v2_service import _route_legacy_regex_custom

        with patch(
            "tilellm.controller.controller.add_item", new=AsyncMock(return_value="ADD_ITEM_RESULT"),
        ) as mock_add, patch(
            "tilellm.controller.controller.add_item_hybrid", new=AsyncMock(),
        ) as mock_add_hybrid:
            result = await _route_legacy_regex_custom(_item(type=DocumentType.REGEX_CUSTOM, hybrid=False))

        mock_add.assert_called_once()
        mock_add_hybrid.assert_not_called()
        assert result == "ADD_ITEM_RESULT"

    @pytest.mark.asyncio
    async def test_hybrid_true_calls_add_item_hybrid(self):
        from tilellm.modules.api_v2.services.ingestion_v2_service import _route_legacy_regex_custom

        with patch(
            "tilellm.controller.controller.add_item", new=AsyncMock(),
        ) as mock_add, patch(
            "tilellm.controller.controller.add_item_hybrid", new=AsyncMock(return_value="HYBRID_RESULT"),
        ) as mock_add_hybrid:
            result = await _route_legacy_regex_custom(_item(type=DocumentType.REGEX_CUSTOM, hybrid=True))

        mock_add_hybrid.assert_called_once()
        mock_add.assert_not_called()
        assert result == "HYBRID_RESULT"


class TestRouteCanonical:
    @pytest.mark.asyncio
    async def test_exports_then_writes_dense_only(self):
        from tilellm.modules.api_v2.services.ingestion_v2_service import _route_canonical
        from tilellm.modules.ingestion.export.models import Block, ExtractedDocument

        doc = ExtractedDocument(type="Text Document", resource="https://x/doc.txt", blocks=[Block(content="x")])
        expected = IngestMdResult(id="doc1", namespace="ns", chunks_indexed=1, chunk_ids=["c1"])
        with patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service.export_document",
            new=AsyncMock(return_value=doc),
        ), patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service.write_extracted_document",
            new=AsyncMock(return_value=expected),
        ) as mock_write:
            result = await _route_canonical(_item(hybrid=False), DocumentType.TXT, repo="REPO", llm_embeddings="EMB")

        assert result == expected
        args, kwargs = mock_write.call_args
        assert args[0] == doc
        assert args[1].hybrid is False
        assert kwargs["source_type"] == "txt"

    @pytest.mark.asyncio
    async def test_hybrid_flag_forwarded_to_ingest_config(self):
        from tilellm.modules.api_v2.services.ingestion_v2_service import _route_canonical
        from tilellm.modules.ingestion.export.models import Block, ExtractedDocument

        doc = ExtractedDocument(type="document", blocks=[Block(content="x")])
        with patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service.export_document",
            new=AsyncMock(return_value=doc),
        ), patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service.write_extracted_document",
            new=AsyncMock(return_value=IngestMdResult(id="doc1", namespace="ns", chunks_indexed=0)),
        ) as mock_write:
            await _route_canonical(
                _item(hybrid=True, sparse_encoder="bge-m3"), DocumentType.TXT, repo="REPO", llm_embeddings="EMB",
            )

        args, _ = mock_write.call_args
        config = args[1]
        assert config.hybrid is True
        assert config.sparse_encoder == "bge-m3"

    @pytest.mark.asyncio
    async def test_situated_context_forwarded_to_ingest_config(self):
        from tilellm.models.llm import SituatedContextConfig
        from tilellm.modules.api_v2.services.ingestion_v2_service import _route_canonical
        from tilellm.modules.ingestion.export.models import Block, ExtractedDocument

        doc = ExtractedDocument(type="document", blocks=[Block(content="x")])
        sc = SituatedContextConfig(enable=True)
        with patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service.export_document",
            new=AsyncMock(return_value=doc),
        ), patch(
            "tilellm.modules.api_v2.services.ingestion_v2_service.write_extracted_document",
            new=AsyncMock(return_value=IngestMdResult(id="doc1", namespace="ns", chunks_indexed=0)),
        ) as mock_write:
            await _route_canonical(_item(situated_context=sc), DocumentType.TXT, repo="REPO", llm_embeddings="EMB")

        args, _ = mock_write.call_args
        assert args[1].situated_context is sc


class TestIngestionV2EndpointRegistered:
    def test_route_exists(self):
        from tilellm.modules.api_v2.controllers import router
        paths = {route.path for route in router.routes}
        assert "/api/v2/ingestion" in paths
