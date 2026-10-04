"""
Proportional scoring rigor: zero-point threshold and units.

Found on a real tender:
- "Maggior resistenza alla compressione MPA: se uguale a 70=0 se > 70 proporzionale"
  — an operator declaring exactly 70 MPa got 9.08/15 instead of 0: the threshold
  lived only in the criterion text, the formula was q/qmax.
- "Tempo di miscelazione ≤ 5 min": one operator measured in seconds (45), the others
  in minutes (0.33, 0.5, ...) — compared as-is, the 45 "minutes" got 0.07 points.

Both are now explicit criterion fields (like `direction`), never inferred from the
text, and a quantity whose unit can't be verified is never scored.
"""
import io

import openpyxl
import pytest

from tilellm.modules.compliance_checker.models_v2 import (
    ComplianceReportV2,
    ComplianceSummaryV2,
    DiscretionaryCriterion,
    DiscretionaryDirection,
    DiscretionaryMode,
    DiscretionaryResult,
    TenderInfo,
    TenderLotRequirements,
    _RequirementsBlock,
)
from tilellm.modules.compliance_checker.prompts.discretionary_judge import (
    DISCRETIONARY_JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
)
from tilellm.modules.compliance_checker.services.bulk_check_service import resolve_proportional
from tilellm.modules.compliance_checker.services.requirements_xlsx_service import (
    RequirementsXlsxService,
)

INV = DiscretionaryDirection.INVERSO


def _report(ns, q, unit=None, measured_unit=None, baseline=None,
            direction=DiscretionaryDirection.DIRETTO, max_points=15.0):
    d = DiscretionaryResult(
        criterion_id="P1", criterion_text="criterio", mode=DiscretionaryMode.PROPORZIONALE,
        max_points=max_points, measured_quantity=q, measured_unit=measured_unit,
        direction=direction, baseline=baseline, unit=unit,
        motivation="m", confidence=0.9, human_review_required=True,
    )
    return ComplianceReportV2(
        tender=TenderInfo(title="t", lot_id="1", lot_name="Lotto 1"), namespace=ns,
        summary=ComplianceSummaryV2.from_results([], [d]),
        tabular_results=[], discretionary_results=[d],
    )


def _resolve(*reports):
    resolve_proportional(list(reports))
    return {r.namespace: r.discretionary_results[0] for r in reports}


# ---------------------------------------------------------------------------
# Zero-point threshold
# ---------------------------------------------------------------------------

def test_value_equal_to_threshold_scores_zero_and_excess_is_proportional():
    by = _resolve(_report("a", 70.0, baseline=70.0), _report("b", 110.0, baseline=70.0),
                  _report("c", 115.7, baseline=70.0))

    assert by["a"].score == 0
    assert by["b"].score == round((110 - 70) / (115.7 - 70) * 15, 2)
    assert by["c"].score == 15
    assert "soglia" in by["b"].human_review_reason


def test_value_below_threshold_scores_zero_not_negative():
    by = _resolve(_report("a", 60.0, baseline=70.0), _report("b", 80.0, baseline=70.0))

    assert by["a"].score == 0
    assert by["b"].score == 15


def test_nobody_above_threshold_everybody_scores_zero():
    by = _resolve(_report("a", 70.0, baseline=70.0), _report("b", 65.0, baseline=70.0))

    assert by["a"].score == 0 and by["b"].score == 0


def test_inverse_threshold_rewards_distance_below_it():
    by = _resolve(_report("a", 0.5, baseline=5.0, direction=INV),
                  _report("b", 5.0, baseline=5.0, direction=INV),
                  _report("c", 2.75, baseline=5.0, direction=INV))

    assert by["a"].score == 15
    assert by["b"].score == 0
    assert by["c"].score == 7.5


def test_without_threshold_the_formula_is_unchanged():
    by = _resolve(_report("a", 70.0), _report("b", 140.0))

    assert by["a"].score == 7.5


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def test_quantity_in_a_unit_other_than_the_declared_one_is_not_scored():
    by = _resolve(_report("a", 0.5, unit="min", measured_unit="minuti", direction=INV),
                  _report("b", 45.0, unit="min", measured_unit="s", direction=INV),
                  _report("c", 1.0, unit="min", measured_unit="min", direction=INV))

    assert by["b"].score is None
    assert "unità" in by["b"].human_review_reason
    # the comparison is made among the verifiable quantities only
    assert by["a"].score == 15
    assert by["c"].score == 7.5


def test_declared_unit_but_judge_reported_none_is_not_scored():
    by = _resolve(_report("a", 0.5, unit="min", measured_unit=None, direction=INV),
                  _report("b", 1.0, unit="min", measured_unit="min", direction=INV))

    assert by["a"].score is None
    assert by["b"].score == 15


def test_mixed_units_without_a_declared_unit_score_nobody():
    by = _resolve(_report("a", 0.5, measured_unit="min", direction=INV),
                  _report("b", 45.0, measured_unit="secondi", direction=INV))

    assert by["a"].score is None and by["b"].score is None
    assert "unità" in by["a"].human_review_reason


# ---------------------------------------------------------------------------
# Plumbing: criterion → judge prompt → result, and the criteria workbook
# ---------------------------------------------------------------------------

def test_judge_is_told_the_unit_and_must_report_it():
    prompt = build_judge_user_prompt(
        criterion_id="P1", criterion_text="Tempo di miscelazione", mode="proporzionale",
        max_points=10, evidence_block="[1] x", unit="min",
    )

    assert "min" in prompt and "measured_unit" in prompt
    assert '"measured_unit"' in DISCRETIONARY_JUDGE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_criterion_threshold_and_unit_reach_the_result():
    from unittest.mock import AsyncMock, patch

    from tilellm.modules.compliance_checker.models_v2 import ComplianceRequestV2
    from tilellm.modules.compliance_checker.services.discretionary_check_service import (
        DiscretionaryCheckService,
    )

    criterion = DiscretionaryCriterion(
        id="P1", text="Resistenza", mode=DiscretionaryMode.PROPORZIONALE, max_points=15,
        baseline=70, unit="MPa",
    )
    req = ComplianceRequestV2(
        requirements_yaml="tender:\n  title: T\n  lot_id: L1\n  lot_name: Lotto 1\n",
        namespace="ns", engine={"name": "pinecone", "type": "serverless"},
    )
    svc = DiscretionaryCheckService(repo=AsyncMock(), llm=AsyncMock(), request=req)
    judged = {"coefficient": None, "measured_value": "110 MPa", "measured_quantity": 110,
              "measured_unit": "MPa", "motivation": "m", "confidence": 0.9,
              "source_chunk_index": 1, "evidence_text": "110 MPa"}
    with patch("tilellm.modules.compliance_checker.services.discretionary_check_service._retrieve_evidence",
               new=AsyncMock(return_value=(["110 MPa"], [{"file_name": "st.pdf", "page": 1}]))), \
         patch.object(svc, "_invoke_judge", new=AsyncMock(return_value=judged)):
        result = await svc._evaluate_criterion(criterion)

    assert (result.baseline, result.unit, result.measured_unit) == (70, "MPa", "MPa")


def test_criteria_workbook_round_trips_threshold_and_unit():
    lot = TenderLotRequirements(
        tender=TenderInfo(title="t", lot_id="1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(discretionary=[DiscretionaryCriterion(
            id="P1", text="Resistenza alla compressione", mode=DiscretionaryMode.PROPORZIONALE,
            max_points=15, baseline=70, unit="MPa",
        )]),
    )
    svc = RequirementsXlsxService()

    parsed = svc.parse_workbook(svc.build_workbook([lot]))[0].requirements.discretionary[0]

    assert (parsed.baseline, parsed.unit) == (70, "MPa")
