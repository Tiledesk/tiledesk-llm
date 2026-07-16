"""Part B — pluggable converter registry contract."""

import pytest

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
