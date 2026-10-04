"""
Docling OCR engine selectable per request via converter_options.

Found on a real tender: a one-page scanned declaration ("privi di lattice/ftalati")
was indexed as "sonoprivi dilattice/ftalati" by Docling's default OCR (auto engine
-> RapidOCR, raster areas only). Glued words defeat both sparse and dense retrieval,
so the declaration was never found. Tesseract with Italian, full page, reads it
cleanly. converter_options was "ignored by docling"; now it carries
ocr_engine / ocr_lang / force_full_page_ocr down to the Docling child process.
"""
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("docling")

from tilellm.modules.pdf_ocr.services import conversion_pipeline, docling_subprocess  # noqa: E402
from tilellm.modules.pdf_ocr.services.markdown_extraction_agent import docling_convert  # noqa: E402


def test_tesseract_full_page_with_language():
    from docling.datamodel.pipeline_options import TesseractCliOcrOptions

    opts = docling_subprocess._ocr_options(
        {"ocr_engine": "tesseract", "ocr_lang": ["ita", "eng"], "force_full_page_ocr": True})

    assert isinstance(opts, TesseractCliOcrOptions)
    assert opts.lang == ["ita", "eng"] and opts.force_full_page_ocr is True


def test_no_ocr_config_keeps_docling_default():
    assert docling_subprocess._ocr_options(None) is None
    assert docling_subprocess._ocr_options({}) is None


def test_unknown_engine_is_rejected_not_ignored():
    with pytest.raises(ValueError, match="ocr_engine"):
        docling_subprocess._ocr_config({"ocr_engine": "tesseractt"})


@pytest.mark.asyncio
async def test_converter_options_reach_the_conversion_subprocess():
    captured = {}

    async def fake_convert(file_path, do_table_structure=True, do_ocr=True, ocr=None):
        captured["ocr"] = ocr
        raise RuntimeError("stop after capture")

    profile = type("P", (), {"needs_segmentation": False, "num_pages": 1})()
    with patch.object(conversion_pipeline, "profile_pdf", return_value=profile), \
         patch.object(conversion_pipeline, "convert_in_subprocess", new=fake_convert):
        with pytest.raises(RuntimeError, match="stop after capture"):
            await docling_convert("/tmp/x.pdf", "doc1", options={
                "skip_ocr": False, "ocr_engine": "tesseract", "ocr_lang": ["ita"],
                "force_full_page_ocr": True, "unrelated": "ignored",
            })

    assert captured["ocr"] == {"ocr_engine": "tesseract", "ocr_lang": ["ita"], "force_full_page_ocr": True}


@pytest.mark.asyncio
async def test_bad_engine_fails_before_the_conversion_starts():
    run = AsyncMock()
    with patch.object(conversion_pipeline, "run_conversion", new=run):
        with pytest.raises(ValueError, match="ocr_engine"):
            await docling_convert("/tmp/x.pdf", "doc1", options={"ocr_engine": "nope"})
    run.assert_not_awaited()
