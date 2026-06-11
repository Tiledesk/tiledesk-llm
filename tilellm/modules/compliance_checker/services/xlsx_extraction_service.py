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


class XlsxExtractionService:
    def __init__(self, llm):
        self._llm = llm

    # ------------------------------------------------------------------
    # Public: full pipeline
    # ------------------------------------------------------------------

    async def extract_requirements(self, url: str) -> List[TenderLotRequirements]:
        """Download xlsx from *url* and return one TenderLotRequirements per sheet."""
        xlsx_bytes = await self._download(url)
        sheets = self.extract_cells(xlsx_bytes)
        lots: List[TenderLotRequirements] = []
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
            lot = self._convert(result)
            if lot is not None:
                lots.append(lot)
        return lots

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

    def _convert(self, result: _LLMLotExtractionResult) -> TenderLotRequirements | None:
        """Convert LLM extraction result to TenderLotRequirements, sanitizing bad values."""
        tabular = [
            TabularRequirementV2(id=t.id, text=t.text, mandatory=t.mandatory)
            for t in result.tabular
        ]

        discretionary: List[DiscretionaryCriterion] = []
        for c in result.discretionary:
            if c.max_points <= 0:
                logger.warning(
                    "XlsxExtractionService: criterio '%s' ha max_points=%s <= 0 — saltato.",
                    c.id, c.max_points,
                )
                continue

            raw_mode = (c.mode or "").strip().lower().replace("/", "_")
            if raw_mode not in _VALID_MODES:
                logger.warning(
                    "XlsxExtractionService: modalità '%s' non riconosciuta per '%s' — "
                    "impostata a 'variabile'.",
                    c.mode, c.id,
                )
                raw_mode = _DEFAULT_MODE.value

            discretionary.append(
                DiscretionaryCriterion(
                    id=c.id,
                    text=c.text,
                    mode=DiscretionaryMode(raw_mode),
                    max_points=c.max_points,
                    human_only=c.human_only,
                    notes=c.notes,
                )
            )

        return TenderLotRequirements(
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
