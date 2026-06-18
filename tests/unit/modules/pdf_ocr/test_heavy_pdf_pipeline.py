"""
Tests for the memory-bounded PDF conversion pipeline:
profiler, segmenter, strategy ladder, attempt tracker, native adapter.

The Docling subprocess itself is exercised by integration tests (it loads
models); here we cover the orchestration logic that must never regress:
the degradation ladder and the crash-proof attempt accounting.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tilellm.modules.pdf_ocr.services.pdf_profiler import PdfProfile, profile_pdf
from tilellm.modules.pdf_ocr.services.pdf_segmenter import cleanup_segments, split_pdf
from tilellm.modules.pdf_ocr.services.conversion_pipeline import (
    ConversionPlan,
    select_plan,
)


@pytest.fixture
def small_pdf(tmp_path):
    import fitz
    path = str(tmp_path / "small.pdf")
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text((50, 72), f"Page {i + 1} content. " * 10)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def heavy_pdf(tmp_path):
    import fitz
    path = str(tmp_path / "heavy.pdf")
    doc = fitz.open()
    for i in range(45):
        page = doc.new_page()
        y = 40
        for row in range(50):
            page.insert_text((30, y), " | ".join(f"SKU-{i}-{row}-{c}" for c in range(10)))
            y += 14
    doc.save(path)
    doc.close()
    return path


class TestPdfProfiler:
    def test_small_pdf_no_segmentation(self, small_pdf):
        profile = profile_pdf(small_pdf)
        assert profile.num_pages == 5
        assert not profile.needs_segmentation
        assert profile.error == ""

    def test_heavy_pdf_needs_segmentation(self, heavy_pdf):
        profile = profile_pdf(heavy_pdf)
        assert profile.num_pages == 45
        assert profile.needs_segmentation

    def test_missing_file_sets_error(self):
        profile = profile_pdf("/nonexistent/file.pdf")
        assert profile.error != ""
        assert not profile.needs_segmentation


class TestPdfSegmenter:
    def test_split_covers_all_pages_without_overlap(self, heavy_pdf):
        import fitz
        segments = split_pdf(heavy_pdf, pages_per_segment=20)
        try:
            assert [(s.start_page, s.end_page) for s in segments] == [
                (0, 19), (20, 39), (40, 44),
            ]
            for seg in segments:
                d = fitz.open(seg.path)
                assert len(d) == seg.end_page - seg.start_page + 1
                d.close()
        finally:
            cleanup_segments(segments)

    def test_cleanup_removes_temp_files(self, small_pdf):
        import os
        segments = split_pdf(small_pdf, pages_per_segment=2)
        cleanup_segments(segments)
        assert not any(os.path.exists(s.path) for s in segments)


class TestStrategyLadder:
    def test_attempt_1_small_doc_full_unsegmented(self):
        plan = select_plan(PdfProfile(num_pages=10, file_size_mb=1), attempt=1)
        assert plan.level == "full"
        assert not plan.segmented
        assert plan.do_table_structure
        assert plan.extraction_quality == "full"

    def test_attempt_1_heavy_doc_full_segmented(self):
        plan = select_plan(PdfProfile(num_pages=200, file_size_mb=30), attempt=1)
        assert plan.level == "full"
        assert plan.segmented

    def test_attempt_2_light_no_table_structure(self):
        plan = select_plan(PdfProfile(num_pages=200, file_size_mb=30), attempt=2)
        assert plan.level == "light"
        assert plan.segmented
        assert not plan.do_table_structure
        assert plan.extraction_quality == "no_table_structure"

    def test_attempt_3_and_beyond_native(self):
        for attempt in (3, 4, 10):
            plan = select_plan(PdfProfile(num_pages=10, file_size_mb=1), attempt=attempt)
            assert plan.level == "native"
            assert plan.extraction_quality == "degraded_native"

    def test_dense_pages_trigger_segmentation(self):
        profile = PdfProfile(num_pages=10, file_size_mb=1, dense_pages=6)
        assert profile.needs_segmentation


class TestAttemptTracker:
    @pytest.mark.asyncio
    async def test_incr_returns_sequential_attempts(self):
        from tilellm.modules.pdf_ocr.services import attempt_tracker

        counters = {}

        class FakeRedis:
            async def incr(self, key):
                counters[key] = counters.get(key, 0) + 1
                return counters[key]
            async def expire(self, key, ttl):
                pass
            async def delete(self, key):
                counters.pop(key, None)

        with patch.object(attempt_tracker, "_get_client", return_value=FakeRedis()):
            assert await attempt_tracker.incr_attempt("ns", "doc1") == 1
            assert await attempt_tracker.incr_attempt("ns", "doc1") == 2
            assert await attempt_tracker.incr_attempt("ns", "doc2") == 1
            await attempt_tracker.clear_attempts("ns", "doc1")
            assert await attempt_tracker.incr_attempt("ns", "doc1") == 1

    @pytest.mark.asyncio
    async def test_fail_open_without_redis(self):
        from tilellm.modules.pdf_ocr.services import attempt_tracker
        with patch.object(attempt_tracker, "_get_client", return_value=None):
            assert await attempt_tracker.incr_attempt("ns", "doc") == 1
            await attempt_tracker.clear_attempts("ns", "doc")  # must not raise


class TestNativeAdapter:
    def _agent(self):
        from tilellm.modules.pdf_ocr.services.markdown_extraction_agent import (
            MarkdownExtractionAgent,
        )
        return MarkdownExtractionAgent.__new__(MarkdownExtractionAgent)

    def test_native_result_adapts_to_agent_shapes(self):
        import pandas as pd
        native = {
            "text_elements": [
                {"id": "d_text_0_0", "text": "Intro", "page": 0, "type": "text"},
                {"id": "d_text_1_0", "text": "Body", "page": 1, "type": "text"},
            ],
            "tables": [
                {"id": "d_table_1_0", "data": pd.DataFrame({"sku": ["A"], "ok": [1]}),
                 "page": 1, "caption": None},
            ],
            "images": [],
            "formulas": [],
            "metadata": {"doc_id": "d", "num_pages": 2},
        }
        texts, images, tables, formulas = self._agent()._native_result_to_elements(native, "d")
        assert len(texts) == 2 and len(tables) == 1 and images == [] and formulas == []
        assert texts[0]["page"] == 0 and texts[1]["page"] == 1
        assert tables[0]["dataframe"] is not None
        assert tables[0]["markdown_table"]
        assert tables[0]["columns"] == ["sku", "ok"]
        # order must be globally unique and monotonic
        orders = [e["order"] for e in texts + tables]
        assert orders == sorted(orders) and len(set(orders)) == len(orders)

    def test_empty_tables_skipped(self):
        import pandas as pd
        native = {
            "text_elements": [],
            "tables": [{"id": "t0", "data": pd.DataFrame(), "page": 0}],
            "images": [], "formulas": [], "metadata": {},
        }
        _, _, tables, _ = self._agent()._native_result_to_elements(native, "d")
        assert tables == []


class TestRunConversionDispatch:
    @pytest.mark.asyncio
    async def test_native_level_skips_docling(self, small_pdf):
        from tilellm.modules.pdf_ocr.services import conversion_pipeline as cp
        with patch(
            "tilellm.modules.pdf_ocr.services.docling_subprocess.convert_in_subprocess",
            new=AsyncMock(side_effect=AssertionError("docling must not be called")),
        ):
            outcome = await cp.run_conversion(small_pdf, "doc", attempt=3)
        assert outcome.native_result is not None
        assert outcome.segments == []
        assert outcome.extraction_quality == "degraded_native"
        assert outcome.native_result["metadata"]["num_pages"] == 5

    @pytest.mark.asyncio
    async def test_segmented_conversion_offsets(self, heavy_pdf):
        from tilellm.modules.pdf_ocr.services import conversion_pipeline as cp

        seen_paths = []

        async def fake_convert(path, do_table_structure=True, do_ocr=True):
            seen_paths.append(path)
            return f"doc_{len(seen_paths)}"  # placeholder document

        with patch.object(cp, "convert_in_subprocess", new=fake_convert):
            outcome = await cp.run_conversion(heavy_pdf, "doc", attempt=1)

        assert outcome.plan.segmented
        assert [s.page_offset for s in outcome.segments] == [0, 20, 40]
        assert len(seen_paths) == 3
        # temp segment files must be cleaned up afterwards
        import os
        assert not any(os.path.exists(p) for p in seen_paths)
