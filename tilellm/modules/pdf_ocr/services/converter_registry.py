"""Pluggable PDF-converter registry.

The md_simple pipeline turns a PDF into per-page markdown plus images/tables
for LLM enrichment. Docling is the default converter; other engines (MinerU,
LightOnOCR, PaddleOCR, ...) are added by implementing the PdfConverter
contract and registering a name — the agent needs no change.

SOLID: this module owns only the registry (S); new engines extend it by
registration (O); the agent depends on ConverterResult, never on a concrete
engine (D). This module imports no engine, so engines can import it freely.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class ConverterResult:
    """Engine-agnostic output the markdown agent consumes.

    page_bodies: (page_no, markdown) in reading order — the document body.
    images / tables: dicts carrying the raw data the LLM enrichment needs.
    """
    page_bodies: List[Tuple[int, str]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    text_elements: List[Dict[str, Any]] = field(default_factory=list)
    formulas: List[Dict[str, Any]] = field(default_factory=list)
    num_pages: int = 0
    extraction_quality: str = "full"


class PdfConverter(Protocol):
    async def __call__(
        self,
        file_path: str,
        doc_id: str,
        attempt: int = 1,
        options: Optional[Dict[str, Any]] = None,
    ) -> ConverterResult: ...


_REGISTRY: Dict[str, PdfConverter] = {}


def register_converter(name: str, converter: PdfConverter) -> None:
    _REGISTRY[name] = converter


def get_converter(name: str) -> PdfConverter:
    if name == "docling" and name not in _REGISTRY:
        # "docling" registers itself as an import side effect of
        # markdown_extraction_agent.py. Callers outside pdf_ocr's own module
        # tree (e.g. /api/v2/ingestion's canonical PDF path) have no reason
        # to import that module explicitly, and pdf_ocr itself is a separate
        # feature independently gated by TILELLM_PROFILE (shared/utility.py)
        # — it may never load in this process. Self-heal for the one engine
        # every caller assumes is always there, instead of a KeyError whose
        # real cause is "an unrelated feature flag was off".
        import tilellm.modules.pdf_ocr.services.markdown_extraction_agent  # noqa: F401
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown PDF converter '{name}'. Registered: {sorted(_REGISTRY)}"
        )


def available_converters() -> List[str]:
    return sorted(_REGISTRY)
