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
from fastapi.responses import PlainTextResponse, Response

from tilellm.modules.compliance_checker.models_v2 import (
    AggregateReport,
    AggregateReportRequest,
    BulkComplianceReport,
    BulkComplianceRequestV2,
    ComplianceReportV2,
    ComplianceRequestV2,
    ExtractRequirementsRequest,
    ExtractRequirementsResponse,
    LotYamlDocument,
    RequirementsToXlsxRequest,
)
from tilellm.modules.compliance_checker.services.discretionary_check_service import check_compliance_v2
from tilellm.modules.compliance_checker.services.bulk_check_service import check_compliance_v2_bulk
from tilellm.modules.compliance_checker.services.xlsx_extraction_service import extract_requirements_di
from tilellm.modules.compliance_checker.services.yaml_requirements_loader import export_requirements_xlsx
from tilellm.modules.compliance_checker.services.requirements_xlsx_service import RequirementsXlsxService
from tilellm.modules.compliance_checker.services.restituzione_xlsx_service import RestituzioneXlsxService
from tilellm.modules.compliance_checker.services.aggregate_report_service import (
    build_aggregate_report,
    render_report,
)

_XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

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


@router_v2.post("/check/xlsx")
async def run_compliance_check_v2_xlsx(request: ComplianceRequestV2):
    """
    Run a v2 compliance check and return the result as the **standardized restituzione
    xlsx** (market template), with an "Operatore Economico" column.

    Single-operator export (one namespace). The same template is reused by the massive
    multi-operator evaluation (rows ordered by criterion, then operator).
    """
    try:
        report = await check_compliance_v2(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("ComplianceChecker v2 xlsx error: %s", e)
        raise HTTPException(status_code=500, detail=f"Compliance check v2 failed: {str(e)}")

    content = RestituzioneXlsxService().build_workbook([report])
    return Response(
        content=content,
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": 'attachment; filename="restituzione.xlsx"'},
    )


@router_v2.post("/check/bulk", response_model=BulkComplianceReport)
async def run_compliance_check_v2_bulk(request: BulkComplianceRequestV2):
    """
    **Massive multi-operator evaluation** of a single lot.

    Supply the lot requirements once (`requirements_yaml` / `requirements_yaml_url` /
    `requirements_xlsx_url`) and the list of `operators` (one namespace per economic
    operator). Each operator is checked, then `proporzionale` criteria ("ampiezza di
    gamma") are resolved across operators — the proposal is filled in but stays flagged
    for human confirmation (semi-automatic).

    `output_format`:
    - `json` → structured `BulkComplianceReport`.
    - `xlsx` → restituzione workbook (one sheet for the lot, rows by criterion then
      operator, with the "Operatore Economico" column).
    """
    try:
        bulk = await check_compliance_v2_bulk(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("ComplianceChecker v2 bulk error: %s", e)
        raise HTTPException(status_code=500, detail=f"Bulk compliance check v2 failed: {str(e)}")

    if request.output_format == "xlsx":
        labels = {o.namespace: o.operator_label for o in bulk.operators}
        reports = [o.report for o in bulk.operators]
        content = RestituzioneXlsxService().build_workbook(reports, operator_labels=labels)
        return Response(
            content=content,
            media_type=_XLSX_MEDIA_TYPE,
            headers={"Content-Disposition": 'attachment; filename="restituzione_massiva.xlsx"'},
        )

    return bulk


@router_v2.post("/requirements/extract", response_model=ExtractRequirementsResponse)
async def extract_requirements_from_xlsx(request: ExtractRequirementsRequest):
    """
    Extract tender requirements from an **xlsx file** and return them for human review.

    The LLM classifies each row as:
    - **tabular** (binary conformity requirements, excluded if absent)
    - **discretionary** (scored criteria: variabile / proporzionale / on_off)

    `output_format`:
    - `yaml` (default) → JSON response with one YAML document per lot.
    - `xlsx` → a downloadable **standardized workbook** (one sheet per lot) that a
      business user can review/correct in Excel and feed back into `/v2/check` via
      `requirements_xlsx_url` (no YAML editing required).
    """
    try:
        extracted_lots = await extract_requirements_di(request)

        if request.output_format == "xlsx":
            lots = [item.lot for item in extracted_lots]
            warnings_by_lot = {item.lot.tender.lot_id: item.warnings for item in extracted_lots}
            content = RequirementsXlsxService().build_workbook(lots, warnings_by_lot=warnings_by_lot)
            return Response(
                content=content,
                media_type=_XLSX_MEDIA_TYPE,
                headers={
                    "Content-Disposition": 'attachment; filename="requisiti_gara.xlsx"',
                },
            )

        lot_docs = [
            LotYamlDocument(
                lot_id=item.lot.tender.lot_id,
                lot_name=item.lot.tender.lot_name,
                yaml=item.lot.to_yaml(),
                warnings=item.warnings,
            )
            for item in extracted_lots
        ]
        return ExtractRequirementsResponse(lots=lot_docs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("xlsx extraction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router_v2.post("/requirements/to-xlsx")
async def requirements_to_xlsx(request: RequirementsToXlsxRequest):
    """
    Export an existing **requirements YAML** as the standardized **review xlsx**
    (market template), so a business operator can review it in Excel without reading YAML.

    Deterministic transform (no LLM). Provide the requirements inline via
    `requirements_yaml` OR via URL in `requirements_yaml_url` (mutually exclusive).
    The reviewed file can be fed back into `/v2/check` via `requirements_xlsx_url`.
    """
    try:
        content = await export_requirements_xlsx(
            yaml_inline=request.requirements_yaml,
            yaml_url=request.requirements_yaml_url,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("requirements to-xlsx failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    return Response(
        content=content,
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": 'attachment; filename="requisiti_gara.xlsx"'},
    )


_AGGREGATE_MEDIA_TYPES = {
    "md": "text/markdown; charset=utf-8",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
}


@router_v2.post("/reports/aggregate")
async def aggregate_reports(request: AggregateReportRequest):
    """
    Combine N completed v2 check reports (JSON) into a **cross-operator comparison**.

    Provide a list of `sources` (each a URL to a `ComplianceReportV2` JSON, with an
    optional `operator_label`). Reports are grouped per lot; for each operator the
    aggregate reports AI score, tabular compliance, points pending human review and
    a short synthesis (deterministic, or LLM-generated when `synthesis_llm=true`).

    `output_format`:
    - `json` → structured `AggregateReport`
    - `md` / `docx` / `pdf` → downloadable file. If the docx/pdf library is missing,
      the export falls back to Markdown and the reason is reported in `X-Report-Warning`.
    """
    try:
        aggregate = await build_aggregate_report(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("aggregate report failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Aggregate report failed: {str(e)}")

    if request.output_format == "json":
        return aggregate

    content, actual_fmt, warning = render_report(aggregate, request.output_format)
    filename = f"report_comparativo.{actual_fmt}"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Report-Format": actual_fmt,
    }
    if warning:
        headers["X-Report-Warning"] = warning
    return Response(
        content=content,
        media_type=_AGGREGATE_MEDIA_TYPES.get(actual_fmt, "application/octet-stream"),
        headers=headers,
    )
