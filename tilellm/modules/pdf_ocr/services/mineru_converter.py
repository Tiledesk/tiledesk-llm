"""MinerU converter — local-library engine for the md_simple pipeline.

MinerU (https://github.com/opendatalab/MinerU) is a layout-aware document
parser. We run it as a subprocess (memory isolation, mirroring the docling
pipeline), then parse its ``*_content_list.json`` into a ``ConverterResult``:
per-page markdown bodies plus extracted images and tables.

MinerU is a heavy optional dependency (detection/OCR/formula/table models, GPU
recommended). It is imported lazily so this module is importable — and the
engine stays registered — even when the package is absent. Install it with::

    poetry install -E mineru      # or -E all

Config via ``options`` (the request's ``converter_options``):
    {"skip_ocr": bool, "timeout": int}   # all optional

SOLID: depends only on ``ConverterResult`` (D); registered by name (O); owns
only the MinerU engine (S).
"""

import asyncio
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tilellm.modules.pdf_ocr.services.converter_registry import (
    ConverterResult,
    register_converter,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600


class MinerUConverter:
    """PdfConverter that delegates conversion to the local MinerU library."""

    async def __call__(
        self,
        file_path: str,
        doc_id: str,
        attempt: int = 1,
        options: Optional[Dict[str, Any]] = None,
    ) -> ConverterResult:
        """Convert *file_path* via MinerU's Python API (no CLI subprocess needed).

        Supported ``options`` keys (all optional):
            skip_ocr   bool   Use txt method (native-digital PDFs); default False.
            backend    str    MinerU backend; default "pipeline" (lighter, no GPU).
            api_url    str    Remote mineru-api base URL; omit to start a local server.
            lang       str    OCR language hint, e.g. "it"; default "" (auto).
        """
        cfg = options or {}
        self._ensure_mineru()

        import tempfile
        from pathlib import Path

        out_dir = tempfile.mkdtemp(prefix=f"mineru_{doc_id}_")
        try:
            await self._run_via_python_api(file_path, out_dir, cfg)
            content_list, content_dir = self._load_content_list(out_dir)
            return self._content_list_to_result(content_list, content_dir, doc_id)
        finally:
            self._cleanup(out_dir)

    @staticmethod
    def _ensure_mineru() -> None:
        """Fail fast with an actionable message if MinerU is not installed."""
        try:
            import mineru  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "MinerU is not installed. Install it with: "
                "poetry install -E mineru  (or -E all)."
            ) from e

    @staticmethod
    async def _run_via_python_api(file_path: str, out_dir: str, cfg: Dict[str, Any]) -> None:
        """Drive MinerU via its public Python API (run_orchestrated_cli).

        Uses backend='pipeline' by default (lighter, CPU-friendly).
        Pass ``backend='hybrid-engine'`` or ``backend='vlm-engine'`` in
        converter_options for GPU/accuracy.  Pass ``api_url`` to use a
        pre-running mineru-api server instead of starting a local one.
        """
        from pathlib import Path
        from mineru.cli.client import run_orchestrated_cli

        method = "txt" if cfg.get("skip_ocr") else "auto"
        # ponytail: pipeline avoids GPU requirement; change to hybrid-engine when GPU available
        backend = cfg.get("backend", "pipeline")
        api_url = cfg.get("api_url")  # None = start local server automatically

        logger.info(f"[mineru] convert {file_path!r} method={method} backend={backend}")
        await run_orchestrated_cli(
            input_path=Path(file_path),
            output_dir=Path(out_dir),
            method=method,
            backend=backend,
            lang=str(cfg.get("lang", "")),
            server_url=None,
            api_url=api_url,
            start_page_id=0,
            end_page_id=None,
            formula_enable=True,
            table_enable=True,
            image_analysis=True,
            # Regenerate content_list.json locally from middle.json so we can parse it.
            client_side_output_generation=True,
        )

    @staticmethod
    def _load_content_list(out_dir: str):
        """Locate *_content_list.json (v3 nested layout) and return (content_list, its_dir).

        v3 layout: {out_dir}/{stem}/{method}/{stem}_content_list.json
        We prefer _content_list.json over _content_list_v2.json for stability.
        """
        import glob
        import json

        # Prefer v1 content_list (stable, widest compat); fall back to v2 if absent.
        for pattern in ("*_content_list.json", "*_content_list_v2.json"):
            matches = glob.glob(
                os.path.join(out_dir, "**", pattern), recursive=True
            )
            # Exclude v2 when looking for v1
            if pattern == "*_content_list.json":
                matches = [m for m in matches if "_v2" not in m]
            if matches:
                path = matches[0]
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f), os.path.dirname(path)

        raise RuntimeError(
            f"MinerU produced no *_content_list.json under {out_dir}. "
            "Make sure --client-side-output-generation true is supported by "
            "your MinerU version."
        )

    @staticmethod
    def _content_list_to_result(
        content_list: List[Dict[str, Any]],
        content_dir: str,
        doc_id: str,
    ) -> ConverterResult:
        """Pure parser: MinerU content_list → ConverterResult.

        - text blocks become the page body (headings prefixed by '#'*text_level)
        - table/equation bodies are embedded inline in the page body (MinerU
          emits table_body as HTML and equations as LaTeX; kept verbatim — no
          extra dependency needed to render them in the markdown)
        - image blocks are loaded as PIL images for downstream captioning/upload
        - table blocks also surface as separate objects (markdown_table=HTML)
        """
        from PIL import Image

        by_page: Dict[int, List[str]] = defaultdict(list)
        images: List[Dict[str, Any]] = []
        tables: List[Dict[str, Any]] = []
        text_elements: List[Dict[str, Any]] = []
        formulas: List[Dict[str, Any]] = []
        order = img_idx = tbl_idx = 0
        max_page = -1

        for block in content_list:
            page_idx = int(block.get("page_idx", 0))
            max_page = max(max_page, page_idx)
            btype = block.get("type")

            if btype == "text":
                text = (block.get("text") or "").strip()
                if not text:
                    continue
                level = block.get("text_level")
                if level:
                    text = f"{'#' * min(int(level), 6)} {text}"
                by_page[page_idx].append(text)
                text_elements.append({
                    "id": f"{doc_id}_text_{order}", "type": "text",
                    "text": text, "page": page_idx, "order": order,
                })
                order += 1

            elif btype == "equation":
                latex = (block.get("text") or "").strip()
                if not latex:
                    continue
                by_page[page_idx].append(f"$$\n{latex}\n$$")
                formulas.append({
                    "id": f"{doc_id}_eq_{order}", "latex": latex,
                    "page": page_idx, "order": order,
                })
                order += 1

            elif btype == "image":
                image_data = MinerUConverter._load_image(
                    content_dir, block.get("img_path")
                )
                if image_data is None:
                    continue
                caption = " ".join(block.get("image_caption") or []).strip()
                images.append({
                    "id": f"{doc_id}_img_{img_idx}", "page": page_idx,
                    "image_data": image_data, "order": order,
                    "caption": caption,
                })
                img_idx += 1
                order += 1

            elif btype == "table":
                table_html = (block.get("table_body") or "").strip()
                if not table_html:
                    continue
                caption = " ".join(block.get("table_caption") or []).strip()
                by_page[page_idx].append(table_html)
                tables.append({
                    "id": f"{doc_id}_tbl_{tbl_idx}", "page": page_idx,
                    "markdown_table": table_html, "caption": caption,
                    "order": order,
                })
                tbl_idx += 1
                order += 1

        page_bodies = [
            (p + 1, "\n\n".join(by_page[p])) for p in sorted(by_page)
        ]
        num_pages = max_page + 1 if max_page >= 0 else 0
        return ConverterResult(
            page_bodies=page_bodies,
            images=images,
            tables=tables,
            text_elements=text_elements,
            formulas=formulas,
            num_pages=num_pages,
            extraction_quality="full",
        )

    @staticmethod
    def _load_image(content_dir: str, img_path: Optional[str]):
        """Load an extracted image fully into memory; None if missing/unreadable."""
        if not img_path:
            return None
        from PIL import Image

        candidate = os.path.join(content_dir, img_path)
        if not os.path.exists(candidate):
            candidate = os.path.join(content_dir, os.path.basename(img_path))
        if not os.path.exists(candidate):
            logger.warning(f"[mineru] image not found: {img_path}")
            return None
        try:
            with Image.open(candidate) as im:
                return im.convert("RGB")  # in-memory copy, safe after file close
        except Exception as e:
            logger.warning(f"[mineru] failed to load image {img_path}: {e}")
            return None

    @staticmethod
    def _cleanup(out_dir: str) -> None:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


register_converter("mineru", MinerUConverter())
