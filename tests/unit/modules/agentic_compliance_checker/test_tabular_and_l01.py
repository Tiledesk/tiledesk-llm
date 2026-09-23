"""
compliance_check_tabular, compliance_check_l01 (P4).

Both wrap DiscretionaryCheckService._check_tabular / ._check_l01 — the same
paths /v2/check uses — rather than reimplementing tabular judgment or L01
reconciliation. Parity tests here mirror test_evaluate_and_report.py's
acceptance test: same mocks, agentic path vs a direct service call, same
result.
"""
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import openpyxl
import pytest
from pydantic import SecretStr

from tilellm.models import Engine
from tilellm.models.schemas.retrieval_schemas import RetrievalChunksResult
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.agentic_compliance_checker.services.tools_core import (
    check_l01_core,
    check_tabular_core,
    build_report_core,
    evaluate_criteria_core,
)
from tilellm.modules.compliance_checker.models import ComplianceReport, ComplianceResult, ComplianceSummary
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    DiscretionaryCriterion,
    OperatorRef,
    TabularRequirementV2,
    TenderInfo,
    TenderLotRequirements,
    _RequirementsBlock,
)
from tilellm.modules.compliance_checker.services.discretionary_check_service import (
    DiscretionaryCheckService,
)


@pytest.fixture
def fake_redis():
    SessionStore._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield SessionStore._client
    SessionStore._client = None


def _request(**overrides) -> BulkComplianceRequestV2:
    kwargs = dict(
        requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\nrequirements: {}\n",
        operators=[OperatorRef(namespace="ns-oe1", operator_label="OE 1")],
        engine=Engine(name="qdrant"),
        llm="openai",
        gptkey=SecretStr("sk-test"),
        model="gpt-4o-mini",
    )
    kwargs.update(overrides)
    return BulkComplianceRequestV2(**kwargs)


def _lot(tabular=None, discretionary=None) -> TenderLotRequirements:
    return TenderLotRequirements(
        tender=TenderInfo(title="Gara test", lot_id="L1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(
            tabular=tabular or [], discretionary=discretionary or [],
        ),
    )


def _tabular_result(req_id, judgment="compliant"):
    return ComplianceResult(
        requirement_id=req_id, requirement_text="testo", category=None, mandatory=True,
        judgment=judgment, confidence=0.9, evidence_text="evidenza", justification="ok",
        evidence_document="offerta.pdf", evidence_page=1, evidence_section="",
        evidence_chunk_index=1, evidence_chunk_ids=[],
    )


def _l01_bytes(rows, headers=("Codice", "Descrizione")):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(list(headers))
    for r in rows:
        ws.append(list(r))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# compliance_check_tabular
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_tabular_stores_and_returns_results(fake_redis):
    lot = _lot(tabular=[
        TabularRequirementV2(id="T1", text="ISO 9001", mandatory=True),
        TabularRequirementV2(id="T2", text="Marcatura CE", mandatory=True),
    ])
    session_id = await SessionStore.create(_request(), lot)

    with patch(
        "tilellm.modules.compliance_checker.services.discretionary_check_service.check_compliance",
    ) as mock_cc:
        mock_cc.return_value = ComplianceReport(
            domain="e_procurement", namespace="ns-oe1", summary=ComplianceSummary(total=2),
            results=[_tabular_result("T1", "compliant"), _tabular_result("T2", "non_compliant")],
        )
        raw = await check_tabular_core(session_id=session_id)

    body = json.loads(raw)
    assert {r["requirement_id"] for r in body["results"]} == {"T1", "T2"}
    stored = await SessionStore.get_tabular_results(session_id, namespace="ns-oe1")
    assert {r.requirement_id for r in stored} == {"T1", "T2"}


@pytest.mark.asyncio
async def test_check_tabular_filters_by_requirement_ids(fake_redis):
    lot = _lot(tabular=[
        TabularRequirementV2(id="T1", text="ISO 9001", mandatory=True),
        TabularRequirementV2(id="T2", text="Marcatura CE", mandatory=True),
    ])
    session_id = await SessionStore.create(_request(), lot)

    with patch(
        "tilellm.modules.compliance_checker.services.discretionary_check_service.check_compliance",
    ) as mock_cc:
        mock_cc.return_value = ComplianceReport(
            domain="e_procurement", namespace="ns-oe1", summary=ComplianceSummary(total=1),
            results=[_tabular_result("T1", "compliant")],
        )
        raw = await check_tabular_core(session_id=session_id, requirement_ids=["T1"])

    # the filtered lot passed to _check_tabular must contain only T1 — verify
    # via the v1 ComplianceRequest built from it (mock_cc's call args).
    v1_request = mock_cc.call_args.args[0]
    assert [r.id for r in v1_request.requirements] == ["T1"]
    body = json.loads(raw)
    assert [r["requirement_id"] for r in body["results"]] == ["T1"]


@pytest.mark.asyncio
async def test_check_tabular_unknown_requirement_id_raises(fake_redis):
    session_id = await SessionStore.create(_request(), _lot(tabular=[
        TabularRequirementV2(id="T1", text="ISO 9001", mandatory=True),
    ]))
    with pytest.raises(ValueError):
        await check_tabular_core(session_id=session_id, requirement_ids=["NOPE"])


@pytest.mark.asyncio
async def test_check_tabular_matches_direct_service_call(fake_redis):
    """Acceptance test, same shape as P3's: agentic path and a direct
    DiscretionaryCheckService._check_tabular call, same mock, same result."""
    lot = _lot(tabular=[TabularRequirementV2(id="T1", text="ISO 9001", mandatory=True)])
    bulk_request = _request()
    request = bulk_request.to_operator_request(bulk_request.operators[0])

    with patch(
        "tilellm.modules.compliance_checker.services.discretionary_check_service.check_compliance",
    ) as mock_cc:
        mock_cc.return_value = ComplianceReport(
            domain="e_procurement", namespace="ns-oe1", summary=ComplianceSummary(total=1),
            results=[_tabular_result("T1", "compliant")],
        )
        direct_service = DiscretionaryCheckService(repo=AsyncMock(), llm=AsyncMock(), request=request)
        direct_results = await direct_service._check_tabular(lot)

        session_id = await SessionStore.create(bulk_request, lot)
        await check_tabular_core(session_id=session_id)

    agentic_results = await SessionStore.get_tabular_results(session_id, namespace="ns-oe1")
    assert len(agentic_results) == 1
    assert agentic_results[0].judgment == direct_results[0].judgment
    assert agentic_results[0].confidence == direct_results[0].confidence


# ---------------------------------------------------------------------------
# compliance_check_l01
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_l01_without_url_reports_unused_and_no_llm_call(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    repo = AsyncMock()
    llm = AsyncMock()

    with patch(
        "tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
        AsyncMock(return_value=(repo, llm)),
    ):
        raw = await check_l01_core(session_id=session_id)

    body = json.loads(raw)
    assert body["results"][0]["used"] is False
    llm.ainvoke.assert_not_awaited()
    stored = await SessionStore.get_l01_result(session_id, "ns-oe1")
    assert stored.used is False


@pytest.mark.asyncio
async def test_check_l01_with_url_runs_reconciliation(fake_redis):
    request = _request(operators=[
        OperatorRef(namespace="ns-oe1", operator_label="OE 1", l01_xlsx_url="http://x/l01.xlsx")
    ])
    session_id = await SessionStore.create(request, _lot())
    repo = AsyncMock()
    repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
        namespace="ns-oe1", chunks=["Scheda tecnica ABC-123"],
        metadata=[{"file_name": "scheda.pdf", "page": 1}],
    ))
    llm = AsyncMock()

    with patch(
        "tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
        AsyncMock(return_value=(repo, llm)),
    ), patch(
        "tilellm.modules.compliance_checker.services.l01_service.fetch_l01",
        new=AsyncMock(return_value=_l01_bytes([("ABC-123", "Guanti")])),
    ):
        raw = await check_l01_core(session_id=session_id)

    body = json.loads(raw)
    assert body["results"][0]["used"] is True
    assert body["results"][0]["matched"] == 1
    llm.ainvoke.assert_not_awaited()  # zero-LLM check, by design


# ---------------------------------------------------------------------------
# compliance_build_report now folds in tabular + L01 (P4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_report_includes_tabular_and_l01(fake_redis):
    lot = _lot(
        tabular=[TabularRequirementV2(id="T1", text="ISO 9001", mandatory=True)],
        discretionary=[DiscretionaryCriterion(id="P1", text="plasticità", mode="variabile", max_points=8)],
    )
    session_id = await SessionStore.create(_request(), lot)
    repo = AsyncMock()
    repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
        namespace="ns-oe1", chunks=["c"], metadata=[{"file_name": "x.pdf", "page": 1}],
    ))
    llm = AsyncMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "coefficient": 0.8, "measured_value": None, "measured_quantity": None,
        "motivation": "ok", "confidence": 0.9, "source_chunk_index": 1,
        "evidence_text": "chunk uno", "capitolato_discrepancy": None,
    })
    llm.ainvoke = AsyncMock(return_value=resp)

    with patch(
        "tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
        AsyncMock(return_value=(repo, llm)),
    ):
        await evaluate_criteria_core(session_id=session_id, criterion_ids=["P1"])
        await check_l01_core(session_id=session_id)
        with patch(
            "tilellm.modules.compliance_checker.services.discretionary_check_service.check_compliance",
        ) as mock_cc:
            mock_cc.return_value = ComplianceReport(
                domain="e_procurement", namespace="ns-oe1", summary=ComplianceSummary(total=1),
                results=[_tabular_result("T1", "compliant")],
            )
            await check_tabular_core(session_id=session_id)
        raw = await build_report_core(session_id=session_id)

    body = json.loads(raw)
    assert body["summary"]["tabular"]["total"] == 1
    assert body["summary"]["tabular"]["compliant"] == 1
    assert body["unevaluated_tabular"] == []
    assert body["l01_check"]["used"] is False
