"""
MinerU converter — local-library engine for the md_simple pipeline.

MinerU is a heavy optional dependency and no runtime is installed here, so we
unit-test the pure parser (`_content_list_to_result`) against a realistic
`content_list.json` fixture, plus the lazy-import guard and registration.
"""

import os

import pytest
from PIL import Image

from tilellm.modules.pdf_ocr.services.converter_registry import (
    ConverterResult,
    get_converter,
)
from tilellm.modules.pdf_ocr.services.mineru_converter import MinerUConverter


def _write_png(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (1, 1), (255, 0, 0)).save(path, format="PNG")


@pytest.fixture
def mineru_output(tmp_path):
    """A tiny MinerU output dir: content_list spanning 2 pages + one image."""
    images_dir = tmp_path / "images"
    _write_png(str(images_dir / "fig1.png"))
    content_list = [
        {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Intro paragraph.", "page_idx": 0},
        {"type": "image", "img_path": "images/fig1.png",
         "image_caption": ["Figure 1"], "page_idx": 0},
        {"type": "table", "table_body": "<table><tr><td>a</td></tr></table>",
         "table_caption": ["Table 1"], "page_idx": 1},
        {"type": "equation", "text": "E=mc^2", "text_format": "latex", "page_idx": 1},
    ]
    return str(tmp_path), content_list


def test_content_list_to_result_groups_pages(mineru_output):
    out_dir, content_list = mineru_output
    result = MinerUConverter._content_list_to_result(content_list, out_dir, "doc1")

    assert isinstance(result, ConverterResult)
    # Two pages, 1-based numbering, reading order preserved.
    assert [p for p, _ in result.page_bodies] == [1, 2]
    page1 = dict(result.page_bodies)[1]
    assert page1.startswith("# Title")
    assert "Intro paragraph." in page1
    page2 = dict(result.page_bodies)[2]
    assert "<table>" in page2          # table embedded inline
    assert "E=mc^2" in page2           # equation embedded inline
    assert result.num_pages == 2


def test_content_list_to_result_extracts_image(mineru_output):
    out_dir, content_list = mineru_output
    result = MinerUConverter._content_list_to_result(content_list, out_dir, "doc1")

    assert len(result.images) == 1
    img = result.images[0]
    assert img["id"] == "doc1_img_0"
    assert img["page"] == 0
    assert isinstance(img["image_data"], Image.Image)  # loaded PIL image


def test_content_list_to_result_extracts_table(mineru_output):
    out_dir, content_list = mineru_output
    result = MinerUConverter._content_list_to_result(content_list, out_dir, "doc1")

    assert len(result.tables) == 1
    tbl = result.tables[0]
    assert tbl["id"] == "doc1_tbl_0"
    assert tbl["page"] == 1
    assert "<table>" in tbl["markdown_table"]
    assert tbl["caption"] == "Table 1"


def test_missing_image_file_is_skipped(tmp_path):
    # img_path points to a file that does not exist → no image emitted, no raise.
    content_list = [
        {"type": "text", "text": "Body", "page_idx": 0},
        {"type": "image", "img_path": "images/missing.png", "page_idx": 0},
    ]
    result = MinerUConverter._content_list_to_result(content_list, str(tmp_path), "doc1")
    assert result.images == []
    assert dict(result.page_bodies)[1] == "Body"


@pytest.mark.asyncio
async def test_call_without_mineru_raises_actionable_error():
    # mineru is not installed in the test venv → clear install hint, not a crash.
    conv = MinerUConverter()
    with pytest.raises(RuntimeError, match=r"mineru"):
        await conv("/tmp/doc.pdf", "doc1")


def test_registered_under_mineru():
    assert isinstance(get_converter("mineru"), MinerUConverter)
