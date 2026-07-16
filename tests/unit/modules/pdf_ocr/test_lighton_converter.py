"""
LightOnOCR converter — VLM-endpoint engine for the md_simple pipeline.

The runtime endpoint is not available yet, so these tests mock the
rasterization and the per-page OCR HTTP call. They pin the contract:
- __call__ builds page_bodies in reading order, images/tables stay empty
- a missing endpoint_url raises a clear ValueError before any network call
- the engine is registered under "lighton"
"""

import pytest

from tilellm.modules.pdf_ocr.services.converter_registry import (
    ConverterResult,
    get_converter,
)
from tilellm.modules.pdf_ocr.services.lighton_converter import LightOnOCRConverter


CFG = {"endpoint_url": "https://ocr.example/v1/chat/completions", "model": "lighton-ocr"}


@pytest.mark.asyncio
async def test_call_builds_page_bodies_in_order(mocker):
    conv = LightOnOCRConverter()
    mocker.patch.object(conv, "_rasterize_pages", return_value=[b"p1", b"p2", b"p3"])
    mocker.patch.object(
        conv, "_ocr_page",
        side_effect=lambda png, cfg: f"# Page for {png.decode()}",
    )

    result = await conv("/tmp/doc.pdf", "doc1", options=CFG)

    assert isinstance(result, ConverterResult)
    assert result.page_bodies == [
        (1, "# Page for p1"),
        (2, "# Page for p2"),
        (3, "# Page for p3"),
    ]
    assert result.images == []
    assert result.tables == []
    assert result.num_pages == 3
    assert result.extraction_quality == "full"


@pytest.mark.asyncio
async def test_missing_endpoint_raises_before_network(mocker):
    conv = LightOnOCRConverter()
    # If rasterize were reached the test would still pass, but we assert it isn't.
    spy = mocker.patch.object(conv, "_rasterize_pages")

    with pytest.raises(ValueError, match="endpoint_url"):
        await conv("/tmp/doc.pdf", "doc1", options={"model": "lighton-ocr"})

    spy.assert_not_called()


@pytest.mark.asyncio
async def test_none_options_raises(mocker):
    conv = LightOnOCRConverter()
    with pytest.raises(ValueError, match="endpoint_url"):
        await conv("/tmp/doc.pdf", "doc1", options=None)


def test_registered_under_lighton():
    assert isinstance(get_converter("lighton"), LightOnOCRConverter)
