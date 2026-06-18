"""
XlsxExtractionService — extracts TenderLotRequirements from an xlsx file via LLM.

Workflow per sheet:
  1. Download xlsx (max 10 MB).
  2. Extract cell text with openpyxl (one text block per sheet).
  3. Call LLM with_structured_output(_LLMLotExtractionResult) per sheet.
  4. Convert _LLMLotExtractionResult → TenderLotRequirements (sanitize bad mode/max_points).
  5. Return List[TenderLotRequirements] — one per sheet.
"""
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List

import httpx
import openpyxl

from tilellm.modules.compliance_checker.models_v2 import (
    DiscretionaryCriterion,
    DiscretionaryMode,
    TenderInfo,
    TenderLotRequirements,
    TabularRequirementV2,
    _RequirementsBlock,
)
from tilellm.modules.compliance_checker.prompts.xlsx_extraction import (
    XLSX_EXTRACTION_SYSTEM_PROMPT,
    XLSX_EXTRACTION_USER_TEMPLATE,
    _LLMLotExtractionResult,
)

from tilellm.shared.utility import inject_llm_chat_async

logger = logging.getLogger(__name__)

_VALID_MODES = {m.value for m in DiscretionaryMode}
_DEFAULT_MODE = DiscretionaryMode.VARIABILE

MAX_XLSX_SIZE: int = 10 * 1024 * 1024  # 10 MB

_MISSING_MAX_POINTS_FLAG = (
    "⚠️ PUNTEGGIO MASSIMO NON TROVATO NEL DOCUMENTO — verificare e correggere manualmente."
)
_MISSING_MAX_POINTS_PLACEHOLDER = 1.0


@dataclass
class ExtractedLot:
    """Un lotto estratto da uno sheet, con i warning da mostrare in revisione umana."""
    lot: TenderLotRequirements
    warnings: List[str] = field(default_factory=list)


class XlsxExtractionService:
    def __init__(self, llm):
        self._llm = llm

    # ------------------------------------------------------------------
    # Public: full pipeline
    # ------------------------------------------------------------------

    async def extract_requirements(self, url: str) -> List[ExtractedLot]:
        """Download xlsx from *url* and return one ExtractedLot per sheet.

        Ogni riga individuata nel foglio viene sempre mantenuta nell'output —
        anche quando dati come il punteggio massimo non sono ricavabili dal
        documento — così che l'operatore umano possa intervenire sulla
        sezione (tabellare o discrezionale) prima di lanciare il check.
        """
        xlsx_bytes = await self._download(url)
        sheets = self.extract_cells(xlsx_bytes)
        extracted: List[ExtractedLot] = []
        structured_llm = self._llm.with_structured_output(_LLMLotExtractionResult)
        for sheet_name, sheet_content in sheets.items():
            user_msg = XLSX_EXTRACTION_USER_TEMPLATE.format(
                sheet_name=sheet_name,
                sheet_content=sheet_content,
            )
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=XLSX_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
            result: _LLMLotExtractionResult = await structured_llm.ainvoke(messages)
            extracted.append(self._convert(result))
        return extracted

    # ------------------------------------------------------------------
    # Public: cell extraction (testable independently)
    # ------------------------------------------------------------------

    def extract_cells(self, xlsx_bytes: bytes) -> Dict[str, str]:
        """
        Parse xlsx bytes and return {sheet_name: text_content} for each sheet.

        Only non-empty rows are included; cells are joined with tabs.
        """
        wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), read_only=True, data_only=True)
        sheets: Dict[str, str] = {}
        for ws in wb.worksheets:
            lines = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    lines.append("\t".join(cells))
            sheets[ws.title] = "\n".join(lines)
        wb.close()
        return sheets

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _download(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
            if len(content) > MAX_XLSX_SIZE:
                raise ValueError(
                    f"Il file xlsx è troppo grande "
                    f"({len(content)} byte > {MAX_XLSX_SIZE} byte massimi)."
                )
            return content

    def _convert(self, result: _LLMLotExtractionResult) -> ExtractedLot:
        """Convert LLM extraction result to TenderLotRequirements, sanitizing bad values.

        Nessun criterio viene mai scartato: i dati non ricavabili dal documento
        (es. punteggio massimo assente) vengono sostituiti con un placeholder
        e segnalati nei `warnings`, in modo che la riga resti visibile e
        modificabile dall'operatore umano.
        """
        tabular = [
            TabularRequirementV2(id=t.id, text=t.text, mandatory=t.mandatory)
            for t in result.tabular
        ]

        warnings: List[str] = list(result.warnings)
        discretionary: List[DiscretionaryCriterion] = []
        for c in result.discretionary:
            max_points = c.max_points
            notes = c.notes
            if max_points <= 0:
                logger.warning(
                    "XlsxExtractionService: criterio '%s' ha max_points=%s <= 0 — "
                    "impostato placeholder=%s, richiede revisione manuale.",
                    c.id, c.max_points, _MISSING_MAX_POINTS_PLACEHOLDER,
                )
                warnings.append(
                    f"Criterio '{c.id}': punteggio massimo non trovato nel documento "
                    f"— impostato a {_MISSING_MAX_POINTS_PLACEHOLDER} come placeholder, "
                    f"verificare manualmente."
                )
                max_points = _MISSING_MAX_POINTS_PLACEHOLDER
                notes = f"{_MISSING_MAX_POINTS_FLAG} {notes}" if notes else _MISSING_MAX_POINTS_FLAG

            raw_mode = (c.mode or "").strip().lower().replace("/", "_")
            if raw_mode not in _VALID_MODES:
                logger.warning(
                    "XlsxExtractionService: modalità '%s' non riconosciuta per '%s' — "
                    "impostata a 'variabile'.",
                    c.mode, c.id,
                )
                warnings.append(
                    f"Criterio '{c.id}': modalità '{c.mode}' non riconosciuta — "
                    f"impostata a 'variabile'."
                )
                raw_mode = _DEFAULT_MODE.value

            discretionary.append(
                DiscretionaryCriterion(
                    id=c.id,
                    text=c.text,
                    mode=DiscretionaryMode(raw_mode),
                    max_points=max_points,
                    human_only=c.human_only,
                    notes=notes,
                )
            )

        lot = TenderLotRequirements(
            tender=TenderInfo(
                title=result.lot_name,
                lot_id=result.lot_id,
                lot_name=result.lot_name,
            ),
            requirements=_RequirementsBlock(
                tabular=tabular,
                discretionary=discretionary,
            ),
        )
        return ExtractedLot(lot=lot, warnings=warnings)


# ---------------------------------------------------------------------------
# Public entry point — decorated for DI (no repo needed, only LLM)
# ---------------------------------------------------------------------------

@inject_llm_chat_async
async def extract_requirements_di(
    request,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    embedding_config_key=None,
    **kwargs,
):
    """
    DI-wrapped entry point for xlsx extraction.

    *request* must have: source (URL), plus standard LLM fields (gptkey, model, …).
    Decorated with @inject_llm_chat_async which resolves the LLM from request credentials.
    """
    svc = XlsxExtractionService(llm=llm)
    return await svc.extract_requirements(request.source)
