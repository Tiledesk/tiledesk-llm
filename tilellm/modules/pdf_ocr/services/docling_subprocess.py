"""
Run Docling conversions in an isolated child process.

Why: Docling conversion of heavy PDFs (huge multi-page tables) can exhaust
memory. When it runs in-process, the OOM killer takes down the whole TaskIQ
worker — no exception is raised, the message is never XACKed, and the task
is re-delivered forever. By isolating the conversion in a child process:

  - an OOM kill terminates only the child; the parent receives a clean
    ConversionProcessDied exception and normal retry/degradation logic applies;
  - the worker process never loads Docling models for this path, reducing
    its own baseline memory.

The pool is a single-worker ProcessPoolExecutor with the 'spawn' context
(fork is unsafe with CUDA and active event loops). The child caches its
DocumentConverter in a module global, so consecutive segments reuse warm
models for the lifetime of the pool process.

The child returns the DoclingDocument serialized via export_to_dict()
(plain picklable dict); the parent rebuilds it with model_validate.
"""

import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Per-segment conversion timeout. A 20-page segment converts in minutes;
# anything beyond this is considered stuck and the child is killed.
SEGMENT_TIMEOUT_S = int(os.environ.get("PDF_OCR_SEGMENT_TIMEOUT_S", "1800"))

# Optional hard memory cap (MB) for the child via RLIMIT_AS. Disabled by
# default. RLIMIT_AS limits virtual address space, not real RSS, so it only
# makes sense on CPU-only deployments — _child_init() below auto-skips it
# (with a warning) whenever a CUDA device is detected, since capping virtual
# memory kills the child the instant it initializes a CUDA context.
CHILD_MEM_LIMIT_MB = int(os.environ.get("PDF_OCR_CHILD_MEM_LIMIT_MB", "0"))


class ConversionProcessDied(Exception):
    """The child process performing the Docling conversion was killed
    (most likely OOM) or exceeded the segment timeout."""


# --------------------------------------------------------------------------
# Child-process side — everything below runs in the spawned process.
# --------------------------------------------------------------------------

_child_converters: Dict[tuple, Any] = {}


def _child_init() -> None:
    """Pool initializer: optional memory cap (CPU-only deployments), quiet logging.

    RLIMIT_AS caps virtual address space, not real RSS — a CUDA context alone
    reserves tens of GB of VA (measured: ~60 GB virtual vs ~5 GB real RSS on a
    TEI process), so applying this on a GPU deployment kills the child the
    instant it initializes CUDA. That failure looks identical to a real OOM
    (ConversionProcessDied), so a misconfigured env var here silently breaks
    every conversion — see docs/MIGLIORIE_DA_FARE.md P1#10. Guarded on
    CUDA availability instead of trusting the "CPU-only" comment above to be
    followed: the cap is skipped (with a warning) whenever CUDA is present.
    """
    if CHILD_MEM_LIMIT_MB <= 0:
        return

    cuda_available = _cuda_available()
    if cuda_available:
        logger.warning(
            f"PDF_OCR_CHILD_MEM_LIMIT_MB={CHILD_MEM_LIMIT_MB} ignored: RLIMIT_AS caps "
            f"virtual memory, which breaks CUDA context initialization. Only takes "
            f"effect on CPU-only deployments (no CUDA device detected)."
        )
        return

    try:
        import resource
        limit = CHILD_MEM_LIMIT_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


_OCR_KEYS = ("ocr_engine", "ocr_lang", "force_full_page_ocr")
_OCR_ENGINES = ("auto", "rapidocr", "tesseract")


def _ocr_config(options: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The OCR keys of a converter's options, validated; None = Docling's default
    (auto engine, raster areas only). An unknown engine is an error, not a silent
    fallback to the default that produced the bad text in the first place."""
    ocr = {k: options[k] for k in _OCR_KEYS if options and options.get(k) is not None}
    if ocr.get("ocr_engine", "auto") not in _OCR_ENGINES:
        raise ValueError(f"ocr_engine '{ocr['ocr_engine']}' non valido: usa uno tra {_OCR_ENGINES}")
    return ocr or None


def _ocr_options(ocr: Optional[Dict[str, Any]]):
    """Docling OcrOptions for a validated OCR config (see _ocr_config)."""
    ocr = _ocr_config(ocr)
    if not ocr:
        return None
    from docling.datamodel.pipeline_options import (
        OcrAutoOptions,
        RapidOcrOptions,
        TesseractCliOcrOptions,
    )
    cls = {"auto": OcrAutoOptions, "rapidocr": RapidOcrOptions,
           "tesseract": TesseractCliOcrOptions}[ocr.get("ocr_engine", "auto")]
    kwargs = {"force_full_page_ocr": bool(ocr.get("force_full_page_ocr", False))}
    if ocr.get("ocr_lang"):
        kwargs["lang"] = list(ocr["ocr_lang"])
    return cls(**kwargs)


def _child_convert(file_path: str, do_table_structure: bool, do_ocr: bool,
                   ocr: Optional[Dict[str, Any]] = None) -> dict:
    """Convert one PDF (segment) and return the serialized DoclingDocument."""
    key = (do_table_structure, do_ocr, repr(sorted((ocr or {}).items())))
    converter = _child_converters.get(key)
    if converter is None:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = do_ocr
        ocr_options = _ocr_options(ocr)
        if ocr_options is not None:
            opts.ocr_options = ocr_options
        opts.do_table_structure = do_table_structure
        if do_table_structure:
            opts.table_structure_options.do_cell_matching = True
        opts.accelerator_options = AcceleratorOptions(device="auto")
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        _child_converters[key] = converter

    result = converter.convert(file_path)
    return result.document.export_to_dict()


# --------------------------------------------------------------------------
# Parent-process side
# --------------------------------------------------------------------------

_pool: Optional[ProcessPoolExecutor] = None
_pool_lock: Optional[asyncio.Lock] = None


def _get_pool_lock() -> asyncio.Lock:
    global _pool_lock
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()
    return _pool_lock


def _create_pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_child_init,
    )


def _kill_pool() -> None:
    """Hard-kill the pool and its child (used on timeout/broken pool)."""
    global _pool
    if _pool is None:
        return
    try:
        for proc in list(getattr(_pool, "_processes", {}).values()):
            try:
                proc.kill()
            except Exception:
                pass
        _pool.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        logger.warning(f"docling_subprocess: pool shutdown error: {e}")
    finally:
        _pool = None


async def convert_in_subprocess(
    file_path: str,
    do_table_structure: bool = True,
    do_ocr: bool = True,
    ocr: Optional[Dict[str, Any]] = None,
) -> Any:
    """Convert a PDF in the isolated child process.

    `ocr`: optional OCR engine config (see _ocr_config); None keeps Docling's default.

    Returns a rebuilt DoclingDocument.
    Raises ConversionProcessDied when the child is killed (OOM) or times out.
    """
    global _pool
    async with _get_pool_lock():
        if _pool is None:
            _pool = _create_pool()
        pool = _pool

    loop = asyncio.get_event_loop()
    try:
        doc_dict = await asyncio.wait_for(
            loop.run_in_executor(
                pool, _child_convert, file_path, do_table_structure, do_ocr, ocr
            ),
            timeout=SEGMENT_TIMEOUT_S,
        )
    except BrokenProcessPool as e:
        async with _get_pool_lock():
            _kill_pool()
        raise ConversionProcessDied(
            f"Docling child process died during conversion of {file_path} "
            f"(likely OOM): {e}"
        ) from e
    except asyncio.TimeoutError as e:
        async with _get_pool_lock():
            _kill_pool()
        raise ConversionProcessDied(
            f"Docling conversion of {file_path} exceeded {SEGMENT_TIMEOUT_S}s, "
            f"child process killed"
        ) from e

    from docling_core.types.doc import DoclingDocument
    return DoclingDocument.model_validate(doc_dict)
