#!/usr/bin/env python3
"""
Per-operator L01 in the massive multi-operator evaluation, plus the L01-sourced
"ampiezza di gamma" quantity.

Two gaps this covers:

1. `BulkComplianceRequestV2.l01_xlsx_url` is a SINGLE url shared by every operator,
   but in a real tender each economic operator files its own L01 ("L. 01 - <OE>").
   Without a per-operator url every operator gets checked against the same workbook.

2. "Maggior ampiezza gamma" is a `proporzionale` criterion whose comparable quantity
   is the NUMBER OF PRODUCTS in that operator's L01 — and the L01 is deliberately NOT
   ingested into the vector store, so the LLM judge can never measure it from the
   retrieved chunks (it returns `measured_quantity=None` and the criterion stays
   unscored for everyone). The count has to come from the parsed L01.
"""
import io
from unittest.mock import AsyncMock, patch

import openpyxl
import pytest

from tilellm.models import Engine
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    ComplianceRequestV2,
    DiscretionaryCriterion,
    DiscretionaryMode,
    OperatorRef,
    TenderInfo,
    TenderLotRequirements,
    _RequirementsBlock,
)
from tilellm.modules.compliance_checker.services.l01_service import parse_l01


def _l01_bytes(rows, headers):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(list(headers))
    for r in rows:
        ws.append(list(r))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _engine():
    return Engine(name="pinecone", type="serverless", apikey="k", vector_size=1536)


_YAML = "tender:\n  title: T\n  lot_id: L1\n  lot_name: Lotto 1\n"


# ---------------------------------------------------------------------------
# 1. parse_l01 against the real PA "tracciato" headers
# ---------------------------------------------------------------------------

# Exactly the header row of the standard L01 workbook filed by the operators.
_REAL_HEADERS = (
    "Lotto", "Voce", "CIG",
    "CODIFICA ARTICOLO OPERATORE ECONOMICO",
    "DENOMINAZIONE ARTICOLO OPERATORE ECONOMICO",
    "CODICE PARAFARMACO", "CODICE EAN",
    "CONFEZIONAMENTO PRIMARIO", "CONFEZIONAMENTO SECONDARIO",
    "NOME COMMERCIALE MODELLO",
)


class TestParseL01RealHeaders:
    def test_recognizes_codifica_and_denominazione(self):
        products = parse_l01(
            _l01_bytes(
                [("1", 1, "B910B42093", "1200/A", "CEMEX RX 40 GR", None, None, 1, 1, "CEMEX RX")],
                headers=_REAL_HEADERS,
            )
        )
        assert len(products) == 1
        # NOT the "Lotto" column ("1"), which is what the untyped fallback would pick.
        assert products[0].code == "1200/A"
        assert products[0].name == "CEMEX RX 40 GR"

    def test_uses_only_the_first_name_column(self):
        """Joining every name-ish column produces a string that matches nothing."""
        products = parse_l01(
            _l01_bytes(
                [("1", 1, "CIG", "1200/A", "CEMEX RX 40 GR", None, None, 1, 1, "CEMEX RX")],
                headers=_REAL_HEADERS,
            )
        )
        assert products[0].name == "CEMEX RX 40 GR"

    def test_counts_every_product_row(self):
        rows = [
            ("1", 1, "CIG", "1200/A", "CEMEX RX 40 GR", None, None, 1, 1, "a"),
            ("1", 1, "CIG", "1200/I", "CEMEX ISOPLASTIC 40 GR", None, None, 1, 1, "b"),
            ("1", 1, "CIG", "1220/I", "CEMEX ISOPLASTIC 1/2 PACK", None, None, 1, 1, "c"),
            ("1", 1, "CIG", "12A3000", "CEMEX FAST 20GR + 20GR", None, None, 1, 1, "d"),
        ]
        products = parse_l01(_l01_bytes(rows, headers=_REAL_HEADERS))
        assert len(products) == 4
        assert [p.code for p in products] == ["1200/A", "1200/I", "1220/I", "12A3000"]


# ---------------------------------------------------------------------------
# 2. per-operator L01 url
# ---------------------------------------------------------------------------

class TestPerOperatorL01:
    def _bulk(self, **over):
        kw = dict(
            requirements_yaml=_YAML,
            operators=[
                OperatorRef(namespace="op1", l01_xlsx_url="http://x/L01-op1.xlsx"),
                OperatorRef(namespace="op2", l01_xlsx_url="http://x/L01-op2.xlsx"),
            ],
            engine=_engine(),
        )
        kw.update(over)
        return BulkComplianceRequestV2(**kw)

    def test_operator_ref_carries_its_own_l01(self):
        op = OperatorRef(namespace="op1", l01_xlsx_url="http://x/L01-op1.xlsx")
        assert op.l01_xlsx_url == "http://x/L01-op1.xlsx"

    def test_operator_l01_defaults_to_none(self):
        assert OperatorRef(namespace="op1").l01_xlsx_url is None

    def test_each_operator_gets_its_own_l01(self):
        bulk = self._bulk()
        assert bulk.to_operator_request(bulk.operators[0]).l01_xlsx_url == "http://x/L01-op1.xlsx"
        assert bulk.to_operator_request(bulk.operators[1]).l01_xlsx_url == "http://x/L01-op2.xlsx"

    def test_request_level_l01_is_the_fallback(self):
        """An operator without its own L01 falls back to the shared request-level one."""
        bulk = self._bulk(
            operators=[OperatorRef(namespace="op1")],
            l01_xlsx_url="http://x/shared.xlsx",
        )
        assert bulk.to_operator_request(bulk.operators[0]).l01_xlsx_url == "http://x/shared.xlsx"

    def test_operator_l01_wins_over_request_level(self):
        bulk = self._bulk(l01_xlsx_url="http://x/shared.xlsx")
        sub = bulk.to_operator_request(bulk.operators[0])
        assert sub.l01_xlsx_url == "http://x/L01-op1.xlsx"

    def test_accepts_a_plain_namespace_string(self):
        """Back-compat: to_operator_request(str) still works for existing callers."""
        bulk = self._bulk(l01_xlsx_url="http://x/shared.xlsx")
        sub = bulk.to_operator_request("op9")
        assert sub.namespace == "op9"
        assert sub.l01_xlsx_url == "http://x/shared.xlsx"


# ---------------------------------------------------------------------------
# 3. "ampiezza di gamma" measured from the L01 product count
# ---------------------------------------------------------------------------

def _lot_with_gamma_criterion(**over):
    kw = dict(
        id="T7",
        text="Maggior ampiezza gamma misure a listino.",
        mode=DiscretionaryMode.PROPORZIONALE,
        max_points=11.0,
        quantity_from_l01=True,
    )
    kw.update(over)
    return TenderLotRequirements(
        tender=TenderInfo(title="T", lot_id="L1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(discretionary=[DiscretionaryCriterion(**kw)]),
    )


def _service(l01_url):
    from tilellm.modules.compliance_checker.services.discretionary_check_service import (
        DiscretionaryCheckService,
    )
    repo = AsyncMock()
    retrieval = AsyncMock()
    retrieval.chunks = ["scheda tecnica generica"]
    retrieval.metadata = [{"file_name": "scheda.pdf", "page": 1}]
    repo.get_chunks_from_repo = AsyncMock(return_value=retrieval)
    request = ComplianceRequestV2(
        requirements_yaml=_YAML, namespace="op1", engine=_engine(), l01_xlsx_url=l01_url,
    )
    return DiscretionaryCheckService(repo=repo, llm=AsyncMock(), request=request)


_JUDGE_UNMEASURED = {
    "verdict": "PARZIALE", "score": 0.0, "confidence": 0.3,
    "motivation": "non quantificabile dai documenti",
    "measured_value": None, "measured_quantity": None,
    "sources": [],
}


class TestAmpiezzaGammaFromL01:
    def test_criterion_declares_the_l01_source(self):
        c = DiscretionaryCriterion(
            id="T7", text="Maggior ampiezza gamma", mode=DiscretionaryMode.PROPORZIONALE,
            max_points=11.0, quantity_from_l01=True,
        )
        assert c.quantity_from_l01 is True

    def test_quantity_from_l01_defaults_false(self):
        c = DiscretionaryCriterion(
            id="D1", text="x", mode=DiscretionaryMode.VARIABILE, max_points=5.0,
        )
        assert c.quantity_from_l01 is False

    @pytest.mark.asyncio
    async def test_product_count_becomes_measured_quantity(self):
        svc = _service("http://x/l01.xlsx")
        l01 = _l01_bytes(
            [
                ("1", 1, "CIG", "1200/A", "CEMEX RX", None, None, 1, 1, "a"),
                ("1", 1, "CIG", "1200/I", "CEMEX ISO", None, None, 1, 1, "b"),
                ("1", 1, "CIG", "1220/I", "CEMEX 1/2", None, None, 1, 1, "c"),
            ],
            headers=_REAL_HEADERS,
        )
        with patch(
            "tilellm.modules.compliance_checker.services.l01_service.fetch_l01",
            new=AsyncMock(return_value=l01),
        ), patch.object(
            svc, "_invoke_judge", new=AsyncMock(return_value=dict(_JUDGE_UNMEASURED))
        ):
            report = await svc.evaluate_lot(_lot_with_gamma_criterion())

        result = report.discretionary_results[0]
        assert result.measured_quantity == 3.0
        assert report.l01_check.l01_products_total == 3

    @pytest.mark.asyncio
    async def test_untouched_when_l01_not_used(self):
        svc = _service(None)
        with patch.object(
            svc, "_invoke_judge", new=AsyncMock(return_value=dict(_JUDGE_UNMEASURED))
        ):
            report = await svc.evaluate_lot(_lot_with_gamma_criterion())
        assert report.discretionary_results[0].measured_quantity is None

    @pytest.mark.asyncio
    async def test_does_not_override_a_judge_measurement(self):
        """The L01 count is a fallback: an explicit judge quantity wins."""
        svc = _service("http://x/l01.xlsx")
        judged = dict(_JUDGE_UNMEASURED, measured_quantity=42.0)
        with patch(
            "tilellm.modules.compliance_checker.services.l01_service.fetch_l01",
            new=AsyncMock(return_value=_l01_bytes(
                [("1", 1, "CIG", "A", "n", None, None, 1, 1, "a")], headers=_REAL_HEADERS)),
        ), patch.object(svc, "_invoke_judge", new=AsyncMock(return_value=judged)):
            report = await svc.evaluate_lot(_lot_with_gamma_criterion())
        assert report.discretionary_results[0].measured_quantity == 42.0

    @pytest.mark.asyncio
    async def test_other_criteria_are_not_touched(self):
        svc = _service("http://x/l01.xlsx")
        lot = _lot_with_gamma_criterion(quantity_from_l01=False)
        with patch(
            "tilellm.modules.compliance_checker.services.l01_service.fetch_l01",
            new=AsyncMock(return_value=_l01_bytes(
                [("1", 1, "CIG", "A", "n", None, None, 1, 1, "a")], headers=_REAL_HEADERS)),
        ), patch.object(
            svc, "_invoke_judge", new=AsyncMock(return_value=dict(_JUDGE_UNMEASURED))
        ):
            report = await svc.evaluate_lot(lot)
        assert report.discretionary_results[0].measured_quantity is None
