"""
ComplianceChecker v2 — FastAPI controller.

New endpoints (never touches v1):
  POST /api/compliance/v2/check               — tabular + discretionary, returns JSON
  POST /api/compliance/v2/check/markdown      — tabular + discretionary, returns Markdown
  POST /api/compliance/v2/requirements/extract — xlsx → YAML (LLM-assisted)

Included via router.include_router(router_v2) in controllers.py.
"""
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from tilellm.modules.compliance_checker.models_v2 import (
    ComplianceReportV2,
    ComplianceRequestV2,
    ExtractRequirementsRequest,
    ExtractRequirementsResponse,
    LotYamlDocument,
)
from tilellm.modules.compliance_checker.services.discretionary_check_service import check_compliance_v2
from tilellm.modules.compliance_checker.services.xlsx_extraction_service import extract_requirements_di

logger = logging.getLogger(__name__)

router_v2 = APIRouter(prefix="/v2", tags=["Compliance Checker v2"])


@router_v2.post("/check", response_model=ComplianceReportV2)
async def run_compliance_check_v2(request: ComplianceRequestV2):
    """
    Run a full v2 compliance check: tabular requirements + scored discretionary criteria.

    Supply requirements as YAML (inline via `requirements_yaml` OR via URL in
    `requirements_yaml_url` — mutually exclusive).  See the YAML schema in the docs.
    """
    try:
        report = await check_compliance_v2(request)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("ComplianceChecker v2 error: %s", e)
        raise HTTPException(status_code=500, detail=f"Compliance check v2 failed: {str(e)}")


@router_v2.post("/check/markdown", response_class=PlainTextResponse)
async def run_compliance_check_v2_markdown(request: ComplianceRequestV2):
    """
    Run a v2 compliance check and return the result as a **Markdown table**.

    Format: tabular requirements (SI/NO/PARZIALE/N/V) + discretionary scoring table.
    Criteria requiring human review are marked `⚠ REVISIONE UMANA`.
    """
    try:
        report = await check_compliance_v2(request)
        return PlainTextResponse(content=report.to_markdown(), media_type="text/markdown")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("ComplianceChecker v2 Markdown error: %s", e)
        raise HTTPException(status_code=500, detail=f"Compliance check v2 failed: {str(e)}")


@router_v2.post("/requirements/extract", response_model=ExtractRequirementsResponse)
async def extract_requirements_from_xlsx(request: ExtractRequirementsRequest):
    """
    Extract tender requirements from an **xlsx file** and return one YAML document per lot.

    The LLM classifies each row as:
    - **tabular** (binary conformity requirements, excluded if absent)
    - **discretionary** (scored criteria: variabile / proporzionale / on_off)

    The returned YAML documents are ready for human review before being used in `/v2/check`.
    """
    try:
        lots = await extract_requirements_di(request)
        lot_docs = [
            LotYamlDocument(
                lot_id=lot.tender.lot_id,
                lot_name=lot.tender.lot_name,
                yaml=lot.to_yaml(),
            )
            for lot in lots
        ]
        return ExtractRequirementsResponse(lots=lot_docs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("xlsx extraction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
