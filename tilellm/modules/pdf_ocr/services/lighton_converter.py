"""LightOnOCR converter — VLM-endpoint engine for the md_simple pipeline.

LightOnOCR is a vision-language OCR model served behind an HTTP endpoint
(OpenAI-compatible chat/vision API). This converter rasterizes each PDF page to
a PNG, sends it to the endpoint, and uses the returned markdown as that page's
body. Tables and images are transcribed *inline* by the model, so this engine
emits no separate image/table objects (and therefore needs no vision captioning
downstream).

Config is passed per-request via ``options`` (the request's ``converter_options``):

    {
        "endpoint_url": "https://host/v1/chat/completions",   # required
        "api_key": "...",                                       # optional (Bearer)
        "model": "lighton-ocr",                                 # optional
        "prompt": "Transcribe this page to GitHub-flavored Markdown.",  # optional
        "timeout": 120,                                         # optional seconds
        "dpi": 200                                              # optional raster DPI
    }

SOLID: depends only on ``ConverterResult`` (D); registered by name (O); this
file owns only the LightOnOCR engine (S). The runtime endpoint is not part of
this codebase — a missing ``endpoint_url`` fails fast with a clear error.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

from tilellm.modules.pdf_ocr.services.converter_registry import (
    ConverterResult,
    register_converter,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Transcribe this document page to GitHub-flavored Markdown. Preserve headings, lists, tables and reading order. Output only the Markdown."
DEFAULT_TIMEOUT = 120
DEFAULT_DPI = 200


class LightOnOCRConverter:
    """PdfConverter that delegates page→markdown to a LightOnOCR HTTP endpoint."""

    async def __call__(
        self,
        file_path: str,
        doc_id: str,
        attempt: int = 1,
        options: Optional[Dict[str, Any]] = None,
    ) -> ConverterResult:
        cfg = options or {}
        endpoint_url = cfg.get("endpoint_url")
        if not endpoint_url:
            raise ValueError(
                "LightOnOCR converter requires 'endpoint_url' in converter_options "
                "(the OpenAI-compatible vision endpoint of the served model)."
            )

        pages = self._rasterize_pages(file_path, dpi=int(cfg.get("dpi", DEFAULT_DPI)))
        page_bodies: List[Any] = []
        for idx, png in enumerate(pages):
            markdown = await self._ocr_page(png, cfg)
            page_bodies.append((idx + 1, (markdown or "").strip()))

        logger.info(
            f"[lighton] OCR'd {len(page_bodies)} pages for {doc_id} via {endpoint_url}"
        )
        return ConverterResult(
            page_bodies=page_bodies,
            images=[],
            tables=[],
            text_elements=[],
            formulas=[],
            num_pages=len(page_bodies),
            extraction_quality="full",
        )

    @staticmethod
    def _rasterize_pages(file_path: str, dpi: int = DEFAULT_DPI) -> List[bytes]:
        """Render each PDF page to PNG bytes via PyMuPDF (fitz)."""
        try:
            import fitz
        except ImportError as e:
            raise ImportError(
                "PyMuPDF (fitz) is required for the LightOnOCR converter. "
                "Install with: pip install pymupdf"
            ) from e

        pages: List[bytes] = []
        doc = fitz.open(file_path)
        try:
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=matrix)
                pages.append(pix.tobytes("png"))
        finally:
            doc.close()
        return pages

    async def _ocr_page(self, png_bytes: bytes, cfg: Dict[str, Any]) -> str:
        """POST one page image to the endpoint and return its markdown."""
        import httpx

        b64 = base64.b64encode(png_bytes).decode("ascii")
        payload = {
            "model": cfg.get("model", "lighton-ocr"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cfg.get("prompt", DEFAULT_PROMPT)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        headers = {}
        api_key = cfg.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = float(cfg.get("timeout", DEFAULT_TIMEOUT))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(cfg["endpoint_url"], json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


register_converter("lighton", LightOnOCRConverter())
