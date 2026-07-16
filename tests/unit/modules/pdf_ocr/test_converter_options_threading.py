"""
converter_options must be merged (alongside skip_ocr) into the options dict the
converter receives. _extract_structure_node performs that merge and does not use
`self`, so we invoke it unbound with a dummy self — this avoids the docling-gated
agent constructor and keeps the test runnable in the poetry venv.
"""

from types import SimpleNamespace

import pytest

from tilellm.modules.pdf_ocr.services.converter_registry import (
    ConverterResult,
    register_converter,
)
from tilellm.modules.pdf_ocr.services.markdown_extraction_agent import (
    MarkdownExtractionAgent,
)


def _base_state(**overrides):
    state = {
        "file_path": "/tmp/doc.pdf",
        "doc_id": "doc1",
        "attempt": 1,
        "converter": "spy",
        "skip_ocr": False,
        "converter_options": None,
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_converter_options_merged_into_options():
    captured = {}

    async def spy_converter(file_path, doc_id, attempt=1, options=None):
        captured["options"] = options
        return ConverterResult(page_bodies=[(1, "body")], num_pages=1)

    register_converter("spy", spy_converter)

    state = _base_state(
        skip_ocr=True,
        converter_options={"endpoint_url": "https://x", "model": "m"},
    )
    await MarkdownExtractionAgent._extract_structure_node(SimpleNamespace(), state)

    assert state.get("error_message") is None
    assert captured["options"] == {
        "skip_ocr": True,
        "endpoint_url": "https://x",
        "model": "m",
    }


@pytest.mark.asyncio
async def test_none_converter_options_is_safe():
    captured = {}

    async def spy_converter(file_path, doc_id, attempt=1, options=None):
        captured["options"] = options
        return ConverterResult(page_bodies=[(1, "body")], num_pages=1)

    register_converter("spy2", spy_converter)

    state = _base_state(converter="spy2", converter_options=None)
    await MarkdownExtractionAgent._extract_structure_node(SimpleNamespace(), state)

    assert captured["options"] == {"skip_ocr": False}
