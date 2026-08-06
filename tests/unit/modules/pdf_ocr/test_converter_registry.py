"""Part B — pluggable converter registry contract."""

import sys

import pytest

import tilellm.modules.pdf_ocr.services.converter_registry as converter_registry
from tilellm.modules.pdf_ocr.services.converter_registry import (
    ConverterResult,
    available_converters,
    get_converter,
    register_converter,
)


def test_docling_registered_on_agent_import():
    # Importing the agent module must self-register the default "docling" engine.
    import tilellm.modules.pdf_ocr.services.markdown_extraction_agent  # noqa: F401

    assert "docling" in available_converters()


def test_get_converter_docling_self_heals_when_pdf_ocr_never_imported():
    """Reproduces the production bug (2026-08-06): /api/v2/ingestion's canonical
    PDF path calls get_converter("docling") without ever importing pdf_ocr —
    it's a separate feature, independently gated by TILELLM_PROFILE
    (shared/utility.py), so its router (and the markdown_extraction_agent import
    that registers "docling" as a side effect) may never load in that process.
    Real failure: KeyError("Unknown PDF converter 'docling'. Registered: []").

    get_converter() must self-heal for the well-known "docling" name instead of
    depending on caller-side import order."""
    agent_module = "tilellm.modules.pdf_ocr.services.markdown_extraction_agent"
    saved_registry = dict(converter_registry._REGISTRY)
    saved_agent_module = sys.modules.pop(agent_module, None)
    converter_registry._REGISTRY.clear()
    try:
        assert "docling" not in available_converters()  # precondition: genuinely unregistered
        converter = get_converter("docling")  # must not raise KeyError
        assert callable(converter)
        assert "docling" in available_converters()
    finally:
        converter_registry._REGISTRY.clear()
        converter_registry._REGISTRY.update(saved_registry)
        if saved_agent_module is not None:
            sys.modules[agent_module] = saved_agent_module


@pytest.mark.asyncio
async def test_register_and_get_roundtrip():
    async def fake(file_path, doc_id, attempt=1, options=None):
        return ConverterResult(num_pages=7, extraction_quality="full")

    register_converter("fake-engine", fake)
    got = get_converter("fake-engine")
    assert got is fake
    result = await got("/x.pdf", "d")
    assert result.num_pages == 7


def test_unknown_converter_raises_with_listing():
    with pytest.raises(KeyError) as ei:
        get_converter("does-not-exist")
    assert "does-not-exist" in str(ei.value)
