#!/usr/bin/env python3
"""
POST /api/export/md — thin routing: export_document() -> serialize per `format`.
No vector store call in this path (regression guard).
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from tilellm.models.document_type import DocumentType
from tilellm.modules.ingestion.controllers import export_md_endpoint
from tilellm.modules.ingestion.export.models import Block, ExportMdRequest, ExtractedDocument


def _doc():
    return ExtractedDocument(type="Text Document", title="T", blocks=[Block(content="hi")])


class TestExportMdEndpoint:
    @pytest.mark.asyncio
    async def test_default_format_returns_markdown(self):
        req = ExportMdRequest(type=DocumentType.TXT, content="hi")
        with patch(
            "tilellm.modules.ingestion.controllers.export_document",
            new=AsyncMock(return_value=_doc()),
        ):
            response = await export_md_endpoint(req)
        body = response.body.decode() if hasattr(response, "body") else response
        assert "type: Text Document" in body
        assert "hi" in body

    @pytest.mark.asyncio
    async def test_json_format_returns_json(self):
        req = ExportMdRequest(type=DocumentType.TXT, content="hi", format="json")
        with patch(
            "tilellm.modules.ingestion.controllers.export_document",
            new=AsyncMock(return_value=_doc()),
        ):
            response = await export_md_endpoint(req)
        assert response.media_type == "application/json"
        assert b'"type":"Text Document"' in response.body

    @pytest.mark.asyncio
    async def test_value_error_becomes_400(self):
        req = ExportMdRequest()
        with patch(
            "tilellm.modules.ingestion.controllers.export_document",
            new=AsyncMock(side_effect=ValueError("no signal")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await export_md_endpoint(req)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_does_not_touch_vector_store(self):
        """Regression guard: export/md must never call add_item / repo."""
        with patch(
            "tilellm.modules.ingestion.controllers.export_document",
            new=AsyncMock(return_value=_doc()),
        ), patch("tilellm.controller.controller.add_item") as mock_add_item:
            req = ExportMdRequest(type=DocumentType.TXT, content="hi")
            await export_md_endpoint(req)
            mock_add_item.assert_not_called()


class TestRouteRegistered:
    def test_export_md_route_exists(self):
        from tilellm.modules.ingestion.controllers import router
        paths = {route.path for route in router.routes}
        assert "/api/export/md" in paths
