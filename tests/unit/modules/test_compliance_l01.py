#!/usr/bin/env python3
"""
L01 products-consistency check (Compliance v2).

Direction L01 -> PDF: for every product listed in the structured L01 workbook,
verify a PDF technical sheet in the namespace supports it. Match on product code,
fallback to normalized name. Explicit `l01_check.used` flag in the report.

Decisions: memory/compliance_l01_check_decisions.md
"""
import io
from unittest.mock import AsyncMock, patch

import openpyxl
import pytest

from tilellm.models import Engine
from tilellm.modules.compliance_checker.models_v2 import (
    ComplianceRequestV2,
    BulkComplianceRequestV2,
    L01CheckResult,
    L01Product,
    OperatorRef,
    TenderInfo,
    TenderLotRequirements,
)
from tilellm.modules.compliance_checker.services.l01_service import (
    parse_l01,
    reconcile_l01_to_pdf,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _l01_bytes(rows, headers=("Codice", "Descrizione")):
    """Build an L01 xlsx: a header row followed by product rows."""
    wb = openpyxl.Workbook()
    ws = wb.active
    if headers is not None:
        ws.append(list(headers))
    for r in rows:
        ws.append(list(r))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _retrieve_returning(text):
    """Fake retrieve() that always returns a single chunk with *text*."""
    async def _r(query):
        return ([text], [{"file_name": "scheda.pdf", "page": 1}])
    return _r


def _base_request(**over):
    kw = dict(
        requirements_yaml="tender:\n  title: T\n  lot_id: L1\n  lot_name: Lotto 1\n",
        namespace="op1",
        engine=Engine(name="pinecone", type="serverless", apikey="k", vector_size=1536),
    )
    kw.update(over)
    return ComplianceRequestV2(**kw)


# ---------------------------------------------------------------------------
# parse_l01
# ---------------------------------------------------------------------------

class TestParseL01:
    def test_code_and_name_columns(self):
        products = parse_l01(_l01_bytes([("ABC-123", "Guanti nitrile"), ("XYZ-9", "Camici")]))
        assert [(p.code, p.name) for p in products] == [
            ("ABC-123", "Guanti nitrile"),
            ("XYZ-9", "Camici"),
        ]

    def test_tolerant_headers(self):
        products = parse_l01(
            _l01_bytes([("A1", "Prodotto uno")], headers=("Codice Articolo", "Denominazione"))
        )
        assert products[0].code == "A1"
        assert products[0].name == "Prodotto uno"

    def test_fallback_no_recognized_headers(self):
        # first column becomes code, the rest is joined into name
        products = parse_l01(
            _l01_bytes([("SKU1", "foo", "bar")], headers=("colA", "colB", "colC"))
        )
        assert products[0].code == "SKU1"
        assert "foo" in products[0].name and "bar" in products[0].name

    def test_skips_empty_rows(self):
        products = parse_l01(_l01_bytes([("A1", "uno"), ("", ""), ("B2", "due")]))
        assert [p.code for p in products] == ["A1", "B2"]

    def test_empty_sheet_returns_empty(self):
        assert parse_l01(_l01_bytes([])) == []


# ---------------------------------------------------------------------------
# reconcile_l01_to_pdf
# ---------------------------------------------------------------------------

class TestReconcile:
    @pytest.mark.asyncio
    async def test_present_by_code(self):
        products = [L01Product(code="ABC-123", name="Guanti")]
        res = await reconcile_l01_to_pdf(
            products, _retrieve_returning("Scheda tecnica ABC-123 lattice"), concurrency=2
        )
        assert res.used is True
        assert res.l01_products_total == 1
        assert res.matched == 1
        assert res.missing_count == 0
        assert res.items[0].present_in_pdf is True
        assert res.items[0].matched_by == "code"

    @pytest.mark.asyncio
    async def test_fallback_by_name_when_code_absent(self):
        products = [L01Product(code="ZZZ-000", name="Guanti nitrile")]
        res = await reconcile_l01_to_pdf(
            products, _retrieve_returning("Offriamo Guanti nitrile monouso"), concurrency=2
        )
        assert res.matched == 1
        assert res.items[0].matched_by == "name"

    @pytest.mark.asyncio
    async def test_missing_when_neither_matches(self):
        products = [L01Product(code="ZZZ-000", name="Camici sterili")]
        res = await reconcile_l01_to_pdf(
            products, _retrieve_returning("Tutt'altro contenuto"), concurrency=2
        )
        assert res.matched == 0
        assert res.missing_count == 1
        assert res.items[0].present_in_pdf is False
        assert res.items[0].matched_by is None
        assert "ZZZ-000" in res.missing

    @pytest.mark.asyncio
    async def test_counts_mixed(self):
        products = [
            L01Product(code="ABC-123", name="Guanti"),
            L01Product(code="ZZZ-000", name="Camici sterili"),
        ]
        res = await reconcile_l01_to_pdf(
            products, _retrieve_returning("Scheda ABC-123"), concurrency=2
        )
        assert (res.matched, res.missing_count, res.l01_products_total) == (1, 1, 2)

    @pytest.mark.asyncio
    async def test_empty_products(self):
        res = await reconcile_l01_to_pdf([], _retrieve_returning("x"), concurrency=2)
        assert res.used is True
        assert res.l01_products_total == 0
        assert res.matched == 0


# ---------------------------------------------------------------------------
# models / request wiring
# ---------------------------------------------------------------------------

class TestModels:
    def test_l01_check_result_default_off(self):
        assert L01CheckResult().used is False

    def test_request_accepts_l01_xlsx_url(self):
        req = _base_request(l01_xlsx_url="http://x/l01.xlsx")
        assert req.l01_xlsx_url == "http://x/l01.xlsx"

    def test_request_l01_default_none(self):
        assert _base_request().l01_xlsx_url is None

    def test_bulk_propagates_l01_to_operator_request(self):
        bulk = BulkComplianceRequestV2(
            requirements_yaml="tender:\n  title: T\n  lot_id: L1\n  lot_name: Lotto 1\n",
            operators=[OperatorRef(namespace="op1")],
            engine=Engine(name="pinecone", type="serverless", apikey="k", vector_size=1536),
            l01_xlsx_url="http://x/l01.xlsx",
        )
        sub = bulk.to_operator_request("op1")
        assert sub.l01_xlsx_url == "http://x/l01.xlsx"


# ---------------------------------------------------------------------------
# integration into evaluate_lot
# ---------------------------------------------------------------------------

def _empty_lot():
    return TenderLotRequirements(tender=TenderInfo(title="T", lot_id="L1", lot_name="Lotto 1"))


class TestEvaluateLotL01:
    @pytest.mark.asyncio
    async def test_no_url_marks_l01_unused(self):
        from tilellm.modules.compliance_checker.services.discretionary_check_service import (
            DiscretionaryCheckService,
        )
        svc = DiscretionaryCheckService(repo=AsyncMock(), llm=AsyncMock(), request=_base_request())
        report = await svc.evaluate_lot(_empty_lot())
        assert report.l01_check is not None
        assert report.l01_check.used is False

    @pytest.mark.asyncio
    async def test_with_url_runs_reconciliation(self):
        from tilellm.modules.compliance_checker.services.discretionary_check_service import (
            DiscretionaryCheckService,
        )
        repo = AsyncMock()
        retrieval = AsyncMock()
        retrieval.chunks = ["Scheda tecnica ABC-123"]
        retrieval.metadata = [{"file_name": "scheda.pdf", "page": 1}]
        repo.get_chunks_from_repo = AsyncMock(return_value=retrieval)

        req = _base_request(l01_xlsx_url="http://x/l01.xlsx")
        svc = DiscretionaryCheckService(repo=repo, llm=AsyncMock(), request=req)

        with patch(
            "tilellm.modules.compliance_checker.services.l01_service.fetch_l01",
            new=AsyncMock(return_value=_l01_bytes([("ABC-123", "Guanti")])),
        ):
            report = await svc.evaluate_lot(_empty_lot())

        assert report.l01_check.used is True
        assert report.l01_check.l01_products_total == 1
        assert report.l01_check.matched == 1


class TestMarkdownL01Section:
    def _report(self, l01):
        from tilellm.modules.compliance_checker.models_v2 import (
            ComplianceReportV2,
            ComplianceSummaryV2,
        )
        return ComplianceReportV2(
            tender=TenderInfo(title="T", lot_id="L1", lot_name="Lotto 1"),
            namespace="op1",
            summary=ComplianceSummaryV2(),
            tabular_results=[],
            discretionary_results=[],
            l01_check=l01,
        )

    def test_section_present_when_used(self):
        from tilellm.modules.compliance_checker.models_v2 import L01ReconItem
        l01 = L01CheckResult(
            used=True, l01_products_total=2, matched=1, missing_count=1,
            missing=["ZZZ-000"],
            items=[
                L01ReconItem(code="ABC-123", name="Guanti", present_in_pdf=True, matched_by="code"),
                L01ReconItem(code="ZZZ-000", name="Camici", present_in_pdf=False),
            ],
        )
        md = self._report(l01).to_markdown()
        assert "Coerenza L01" in md
        assert "ZZZ-000" in md

    def test_no_section_when_unused(self):
        md = self._report(L01CheckResult(used=False)).to_markdown()
        assert "Coerenza L01" not in md
