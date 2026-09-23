"""
compliance_resolve_proportional (P5).

Delegates entirely to bulk_check_service.resolve_proportional — this module's
only job is assembling the per-operator result set (from stored
compliance_evaluate_criteria results) and re-storing what comes back mutated.
The acceptance test mirrors P3/P4's: same stored results, agentic tool vs a
direct resolve_proportional(reports) call, same scores.
"""
import json

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from tilellm.models import Engine
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.agentic_compliance_checker.services.tools_core import resolve_proportional_core
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    ComplianceReportV2,
    ComplianceSummaryV2,
    DiscretionaryCriterion,
    DiscretionaryDirection,
    DiscretionaryMode,
    DiscretionaryResult,
    OperatorRef,
    TenderInfo,
    TenderLotRequirements,
    _RequirementsBlock,
)
from tilellm.modules.compliance_checker.services.bulk_check_service import resolve_proportional


@pytest.fixture
def fake_redis():
    SessionStore._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield SessionStore._client
    SessionStore._client = None


def _request(operators=None, **overrides) -> BulkComplianceRequestV2:
    kwargs = dict(
        requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\nrequirements: {}\n",
        operators=operators or [
            OperatorRef(namespace="ns-a", operator_label="Alpha"),
            OperatorRef(namespace="ns-b", operator_label="Beta"),
        ],
        engine=Engine(name="qdrant"),
        llm="openai",
        gptkey=SecretStr("sk-test"),
        model="gpt-4o-mini",
    )
    kwargs.update(overrides)
    return BulkComplianceRequestV2(**kwargs)


def _lot(direction=DiscretionaryDirection.DIRETTO) -> TenderLotRequirements:
    return TenderLotRequirements(
        tender=TenderInfo(title="Gara test", lot_id="L1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(discretionary=[
            DiscretionaryCriterion(
                id="P2", text="ampiezza gamma", mode="proporzionale", max_points=10.0, direction=direction,
            ),
        ]),
    )


def _prop_result(q, cid="P2", max_points=10.0, direction=DiscretionaryDirection.DIRETTO) -> DiscretionaryResult:
    return DiscretionaryResult(
        criterion_id=cid, criterion_text="ampiezza gamma",
        mode=DiscretionaryMode.PROPORZIONALE, max_points=max_points, score=None,
        measured_value=(f"{q} misure" if q is not None else None), measured_quantity=q,
        direction=direction, motivation="m", confidence=0.7, human_review_required=True,
        human_review_reason="Confronto tra operatori richiesto.",
    )


async def _seed_result(session_id, namespace, result):
    await SessionStore.store_result(session_id, namespace, result.criterion_id, result)


# ---------------------------------------------------------------------------
# core behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_proportional_diretto_scores_correctly(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    await _seed_result(session_id, "ns-a", _prop_result(10.0))
    await _seed_result(session_id, "ns-b", _prop_result(5.0))

    raw = await resolve_proportional_core(session_id=session_id)

    body = json.loads(raw)
    assert body["partial"] is False
    by_ns = {r["operator"]: r for r in body["results"]}
    assert by_ns["Alpha"]["score"] == 10.0  # max quantity wins full points
    assert by_ns["Beta"]["score"] == 5.0  # 5/10 * 10
    assert by_ns["Alpha"]["proportional_auto"] is True

    stored_a = await SessionStore.get_result(session_id, "ns-a", "P2")
    assert stored_a.score == 10.0
    assert stored_a.human_review_required is True  # proposal, never auto-final


@pytest.mark.asyncio
async def test_resolve_proportional_inverso_scores_correctly(fake_redis):
    session_id = await SessionStore.create(_request(), _lot(direction=DiscretionaryDirection.INVERSO))
    await _seed_result(session_id, "ns-a", _prop_result(3.0, direction=DiscretionaryDirection.INVERSO))
    await _seed_result(session_id, "ns-b", _prop_result(6.0, direction=DiscretionaryDirection.INVERSO))

    raw = await resolve_proportional_core(session_id=session_id)

    body = json.loads(raw)
    by_ns = {r["operator"]: r for r in body["results"]}
    assert by_ns["Alpha"]["score"] == 10.0  # lowest quantity wins for inverso
    assert by_ns["Beta"]["score"] == 5.0  # 3/6 * 10


@pytest.mark.asyncio
async def test_resolve_proportional_missing_operator_raises(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    await _seed_result(session_id, "ns-a", _prop_result(10.0))
    # ns-b never evaluated

    with pytest.raises(ValueError, match="Beta"):
        await resolve_proportional_core(session_id=session_id)

    # nothing was mutated/stored for ns-a either — all-or-nothing by default
    stored_a = await SessionStore.get_result(session_id, "ns-a", "P2")
    assert stored_a.score is None


@pytest.mark.asyncio
async def test_resolve_proportional_allow_partial_proceeds_and_flags(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    await _seed_result(session_id, "ns-a", _prop_result(10.0))
    # ns-b never evaluated

    raw = await resolve_proportional_core(session_id=session_id, allow_partial=True)

    body = json.loads(raw)
    assert body["partial"] is True
    assert body["missing"] == {"Beta": ["P2"]}
    assert len(body["results"]) == 1  # only ns-a had something to resolve
    stored_a = await SessionStore.get_result(session_id, "ns-a", "P2")
    assert stored_a.score == 10.0
    assert "allow_partial" in stored_a.human_review_reason


@pytest.mark.asyncio
async def test_resolve_proportional_unknown_criterion_id_raises(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    with pytest.raises(ValueError):
        await resolve_proportional_core(session_id=session_id, criterion_ids=["NOPE"])


# ---------------------------------------------------------------------------
# Acceptance test (P5's stated criterion)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_proportional_matches_direct_bulk_service_call(fake_redis):
    """Same stored per-operator results, resolved via the agentic tool vs a
    direct bulk_check_service.resolve_proportional(reports) call — same
    scores, or this module reimplemented the cross-operator formula."""
    session_id = await SessionStore.create(_request(), _lot())
    result_a, result_b = _prop_result(10.0), _prop_result(5.0)
    await _seed_result(session_id, "ns-a", result_a)
    await _seed_result(session_id, "ns-b", result_b)

    # Path A: direct service call on independent copies of the same data
    direct_a, direct_b = _prop_result(10.0), _prop_result(5.0)
    direct_reports = [
        ComplianceReportV2(
            tender=TenderInfo(title="t", lot_id="L1", lot_name="n"), namespace="ns-a",
            summary=ComplianceSummaryV2(), tabular_results=[], discretionary_results=[direct_a],
        ),
        ComplianceReportV2(
            tender=TenderInfo(title="t", lot_id="L1", lot_name="n"), namespace="ns-b",
            summary=ComplianceSummaryV2(), tabular_results=[], discretionary_results=[direct_b],
        ),
    ]
    resolve_proportional(direct_reports)

    # Path B: through the agentic tool
    await resolve_proportional_core(session_id=session_id)
    agentic_a = await SessionStore.get_result(session_id, "ns-a", "P2")
    agentic_b = await SessionStore.get_result(session_id, "ns-b", "P2")

    assert agentic_a.score == direct_a.score
    assert agentic_b.score == direct_b.score
    assert agentic_a.proportional_auto == direct_a.proportional_auto
