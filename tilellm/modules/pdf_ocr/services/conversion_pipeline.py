"""
Memory-bounded PDF conversion pipeline with attempt-driven degradation.

Orchestrates: profile → plan → convert (subprocess, optionally segmented).

The strategy ladder, driven by the crash-proof attempt counter:

  attempt 1 → 'full'   : Docling with table structure; segmented when the
                         profile says the document is heavy.
  attempt 2 → 'light'  : Docling without table-structure recognition
                         (tables come out as plain text), forced segmentation
                         with smaller segments. Survives almost anything.
  attempt 3+ → 'native': PyMuPDF raw extraction (process_pdf_native).
                         Cheap and deterministic — always succeeds on a
                         readable PDF. Content is indexed with
                         extraction_quality='degraded_native'.

Content is never silently truncated: every level processes ALL pages.
Degradation reduces structural fidelity, not coverage, and the resulting
quality level is recorded in the outcome for downstream metadata/webhooks.

SOLID notes: this module owns only orchestration (S). New levels are added
by extending select_plan and run_conversion's dispatch (O). Callers depend
on ConversionOutcome, never on Docling types directly (D).
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tilellm.modules.pdf_ocr.services.pdf_profiler import PdfProfile, profile_pdf
from tilellm.modules.pdf_ocr.services.pdf_segmenter import (
    PdfSegment,
    cleanup_segments,
    split_pdf,
)
from tilellm.modules.pdf_ocr.services.docling_subprocess import convert_in_subprocess

logger = logging.getLogger(__name__)

PAGES_PER_SEGMENT_FULL = int(os.environ.get("PDF_SEGMENT_PAGES", "20"))
PAGES_PER_SEGMENT_LIGHT = int(os.environ.get("PDF_SEGMENT_PAGES_LIGHT", "10"))


@dataclass
class ConversionPlan:
    level: str                      # 'full' | 'light' | 'native'
    segmented: bool = False
    pages_per_segment: int = PAGES_PER_SEGMENT_FULL
    do_table_structure: bool = True
    do_ocr: bool = True

    @property
    def extraction_quality(self) -> str:
        return {
            "full": "full",
            "light": "no_table_structure",
            "native": "degraded_native",
        }[self.level]


@dataclass
class ConvertedSegment:
    """One converted slice: a DoclingDocument plus its page offset in the original."""
    document: Any                   # DoclingDocument
    page_offset: int = 0            # 0-based page number of the segment's first page


@dataclass
class ConversionOutcome:
    plan: ConversionPlan
    profile: PdfProfile
    segments: List[ConvertedSegment] = field(default_factory=list)   # docling levels
    native_result: Optional[Dict[str, Any]] = None                   # native level

    @property
    def extraction_quality(self) -> str:
        return self.plan.extraction_quality


def select_plan(profile: PdfProfile, attempt: int) -> ConversionPlan:
    """Map (document profile, attempt number) to a conversion plan."""
    if attempt >= 3:
        return ConversionPlan(level="native")
    if attempt == 2:
        return ConversionPlan(
            level="light",
            segmented=True,
            pages_per_segment=PAGES_PER_SEGMENT_LIGHT,
            do_table_structure=False,
            do_ocr=True,
        )
    return ConversionPlan(
        level="full",
        segmented=profile.needs_segmentation,
        pages_per_segment=PAGES_PER_SEGMENT_FULL,
        do_table_structure=True,
        do_ocr=True,
    )


async def run_conversion(file_path: str, doc_id: str, attempt: int = 1) -> ConversionOutcome:
    """Profile the PDF, select a plan for this attempt, and execute it.

    Raises ConversionProcessDied (from docling_subprocess) when the child
    process is killed — the caller's retry logic will re-enter with a higher
    attempt number and a more conservative plan.
    """
    profile = await asyncio.to_thread(profile_pdf, file_path)
    plan = select_plan(profile, attempt)
    logger.info(
        f"conversion_pipeline: doc={doc_id} attempt={attempt} → level={plan.level} "
        f"segmented={plan.segmented} quality={plan.extraction_quality}"
    )

    outcome = ConversionOutcome(plan=plan, profile=profile)

    if plan.level == "native":
        from tilellm.modules.pdf_ocr.services.pdf_native_processor import process_pdf_native
        outcome.native_result = await process_pdf_native(file_path, doc_id)
        return outcome

    if not plan.segmented:
        document = await convert_in_subprocess(
            file_path,
            do_table_structure=plan.do_table_structure,
            do_ocr=plan.do_ocr,
        )
        outcome.segments.append(ConvertedSegment(document=document, page_offset=0))
        return outcome

    segments: List[PdfSegment] = await asyncio.to_thread(
        split_pdf, file_path, plan.pages_per_segment
    )
    try:
        for i, seg in enumerate(segments):
            logger.info(
                f"conversion_pipeline: doc={doc_id} converting segment "
                f"{i + 1}/{len(segments)} (pages {seg.start_page + 1}-{seg.end_page + 1})"
            )
            document = await convert_in_subprocess(
                seg.path,
                do_table_structure=plan.do_table_structure,
                do_ocr=plan.do_ocr,
            )
            outcome.segments.append(
                ConvertedSegment(document=document, page_offset=seg.start_page)
            )
    finally:
        cleanup_segments(segments)

    return outcome
