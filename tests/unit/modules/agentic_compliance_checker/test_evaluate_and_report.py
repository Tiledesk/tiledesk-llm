"""
compliance_retrieve_evidence, compliance_evaluate_criteria, compliance_build_report (P3).

The acceptance test (test_agentic_path_matches_direct_service_call) is P3's
stated criterion: the same criterion, same mocked repo/llm, evaluated via the
agentic tool sequence and via DiscretionaryCheckService directly, must
produce the same DiscretionaryResult. If they diverge, this module
reimplemented something it should have delegated.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from tilellm.models import Engine
from tilellm.models.schemas.retrieval_schemas import RetrievalChunksResult
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.agentic_compliance_checker.services.tools_core import (
    build_report_core,
    evaluate_criteria_core,
    retrieve_evidence_core,
)
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    DiscretionaryCriterion,
    OperatorRef,
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


def _lot() -> TenderLotRequirements:
    return TenderLotRequirements(
        tender=TenderInfo(title="Gara test", lot_id="L1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(
            discretionary=[
                DiscretionaryCriterion(id="P1", text="plasticità", mode="variabile", max_points=8),
            ],
        ),
    )


def _judge_response(coefficient=0.8, confidence=0.9):
    resp = MagicMock()
    resp.content = json.dumps({
        "coefficient": coefficient, "measured_value": None, "measured_quantity": None,
        "motivation": "ok", "confidence": confidence, "source_chunk_index": 1,
        "evidence_text": "chunk uno", "capitolato_discrepancy": None,
    })
    return resp


# ---------------------------------------------------------------------------
# compliance_retrieve_evidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_evidence_by_criterion_id_stores_and_previews(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    repo = AsyncMock()
    repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
        namespace="ns-oe1", chunks=["chunk uno " * 30], metadata=[{"file_name": "offerta.pdf", "page": 3}],
    ))

    with patch("tilellm.modules.agentic_compliance_checker.services.tools_core._resolve_deps",
               AsyncMock(return_value=(repo, AsyncMock()))):
        raw = await retrieve_evidence_core(session_id=session_id, criterion_id="P1")

    body = json.loads(raw)
    assert body["chunk_count"] == 1
    assert body["query_kind"] == "criterion"
    assert len(body["preview"][0]["excerpt"]) <= 200
    evidence = await SessionStore.get_evidence(session_id, body["evidence_ref"])
    assert evidence.chunks[0].startswith("chunk uno")  # full text preserved, not truncated in storage
    assert evidence.criterion_id == "P1"


@pytest.mark.asyncio
async def test_retrieve_evidence_requires_exactly_one_of_criterion_id_or_query(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())

    with pytest.raises(ValueError):
        await retrieve_evidence_core(session_id=session_id)
    with pytest.raises(ValueError):
        await retrieve_evidence_core(session_id=session_id, criterion_id="P1", query="x")


# ---------------------------------------------------------------------------
# compliance_evaluate_criteria
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_criteria_default_path_stores_result(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    repo = AsyncMock()
    repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
        namespace="ns-oe1", chunks=["chunk uno"], metadata=[{"file_name": "offerta.pdf", "page": 3}],
    ))
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=_judge_response(coefficient=0.8))

    with patch("tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
               AsyncMock(return_value=(repo, llm))):
        raw = await evaluate_criteria_core(session_id=session_id, criterion_ids=["P1"])

    body = json.loads(raw)
    assert body["results"][0]["coefficient"] == 0.8
    assert body["results"][0]["score"] == 6.4
    stored = await SessionStore.get_result(session_id, "ns-oe1", "P1")
    assert stored.coefficient == 0.8


@pytest.mark.asyncio
async def test_evaluate_criteria_with_evidence_ref_skips_retrieval(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    repo = AsyncMock()
    repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
        namespace="ns-oe1", chunks=["c"], metadata=[{"file_name": "x.pdf", "page": 1}],
    ))
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=_judge_response(coefficient=0.75))

    with patch("tilellm.modules.agentic_compliance_checker.services.tools_core._resolve_deps",
               AsyncMock(return_value=(repo, AsyncMock()))):
        preview_raw = await retrieve_evidence_core(session_id=session_id, criterion_id="P1")
    evidence_ref = json.loads(preview_raw)["evidence_ref"]
    assert repo.get_chunks_from_repo.await_count == 1  # the one retrieval, during compliance_retrieve_evidence

    with patch("tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
               AsyncMock(return_value=(repo, llm))):
        raw = await evaluate_criteria_core(
            session_id=session_id, criterion_ids=["P1"], evidence_ref=evidence_ref,
        )

    body = json.loads(raw)
    assert body["results"][0]["coefficient"] == 0.75
    assert repo.get_chunks_from_repo.await_count == 1  # unchanged — evaluate_criteria did NOT retrieve again


@pytest.mark.asyncio
async def test_evaluate_criteria_evidence_ref_multi_criterion_rejected(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())

    with pytest.raises(ValueError):
        await evaluate_criteria_core(
            session_id=session_id, criterion_ids=["P1", "P2"], evidence_ref="ev-x",
        )


@pytest.mark.asyncio
async def test_evaluate_criteria_evidence_ref_from_different_criterion_flags_override(fake_redis):
    """A guardrail this module adds: mismatched evidence_ref is permitted but
    never silent — must flag human_review with an explicit reason."""
    session_id = await SessionStore.create(_request(), _lot())
    from tilellm.modules.agentic_compliance_checker.models import EvidenceEntry
    await SessionStore.store_evidence(session_id, EvidenceEntry(
        evidence_ref="ev-other", namespace="ns-oe1", criterion_id="P2",  # different criterion
        query="q", query_kind="custom", chunks=["c"], metadata=[{"file_name": "x.pdf", "page": 1}],
    ))
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=_judge_response(coefficient=0.9))

    with patch("tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
               AsyncMock(return_value=(AsyncMock(), llm))):
        raw = await evaluate_criteria_core(
            session_id=session_id, criterion_ids=["P1"], evidence_ref="ev-other",
        )

    body = json.loads(raw)
    assert body["results"][0]["human_review_required"] is True
    assert "un altro criterio" in body["results"][0]["human_review_reason"]


@pytest.mark.asyncio
async def test_evaluate_criteria_increments_attempts_on_reevaluation(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    repo = AsyncMock()
    repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
        namespace="ns-oe1", chunks=["c"], metadata=[{"file_name": "x.pdf", "page": 1}],
    ))
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=_judge_response())

    with patch("tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
               AsyncMock(return_value=(repo, llm))):
        first = json.loads(await evaluate_criteria_core(session_id=session_id, criterion_ids=["P1"]))
        second = json.loads(await evaluate_criteria_core(session_id=session_id, criterion_ids=["P1"]))

    assert first["results"][0]["attempt"] == 1
    assert second["results"][0]["attempt"] == 2


# ---------------------------------------------------------------------------
# compliance_build_report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_report_reflects_stored_results(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    repo = AsyncMock()
    repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
        namespace="ns-oe1", chunks=["c"], metadata=[{"file_name": "x.pdf", "page": 1}],
    ))
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=_judge_response(coefficient=0.8))

    with patch("tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
               AsyncMock(return_value=(repo, llm))):
        await evaluate_criteria_core(session_id=session_id, criterion_ids=["P1"])
        raw = await build_report_core(session_id=session_id)

    body = json.loads(raw)
    assert body["unevaluated_criteria"] == []
    assert body["summary"]["ai_scored_count"] == 1
    assert body["summary"]["ai_scored_points"] == 6.4


@pytest.mark.asyncio
async def test_build_report_lists_unevaluated_criteria(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())

    raw = await build_report_core(session_id=session_id)

    body = json.loads(raw)
    assert body["unevaluated_criteria"] == ["P1"]
    # discretionary_total reflects EVALUATED results (ComplianceSummaryV2's own
    # semantics, unchanged) — 0 here is correct, unevaluated_criteria is what
    # a caller uses to see what's still outstanding against the lot.
    assert body["summary"]["discretionary_total"] == 0


# ---------------------------------------------------------------------------
# Anti-bypass regression test
# ---------------------------------------------------------------------------

def test_no_tool_args_schema_exposes_a_verdict_field():
    """No tool in this module can ever accept a score, coefficient, confidence,
    judgment or gptkey as an argument — the agent chooses WHAT to evaluate and
    WITH WHICH evidence, never the verdict itself. This is what makes the
    guardrails non-bypassable regardless of tool-call order."""
    from tilellm.modules.agentic_compliance_checker.services.langchain_tools import (
        AGENTIC_COMPLIANCE_TOOLS,
    )
    forbidden = {"score", "coefficient", "confidence", "human_review_required", "judgment", "gptkey"}
    for name, tool_obj in AGENTIC_COMPLIANCE_TOOLS.items():
        fields = set(tool_obj.args_schema.model_fields.keys())
        assert not (fields & forbidden), f"{name} exposes forbidden field(s): {fields & forbidden}"


# ---------------------------------------------------------------------------
# Acceptance test (P3's stated criterion)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agentic_path_matches_direct_service_call(fake_redis):
    """Same criterion, same mocked repo/llm responses: the agentic tool
    sequence and a direct DiscretionaryCheckService call must produce the
    same DiscretionaryResult. Divergence here means this module reimplemented
    scoring/guardrail logic instead of delegating to compliance_checker."""
    lot = _lot()
    criterion = lot.requirements.discretionary[0]
    bulk_request = _request()
    request = bulk_request.to_operator_request(bulk_request.operators[0])

    def _fresh_repo():
        repo = AsyncMock()
        repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
            namespace="ns-oe1", chunks=["ME-CC-011 plasticità 90"],
            metadata=[{"file_name": "scheda.pdf", "page": 12}],
        ))
        return repo

    def _fresh_llm():
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_judge_response(coefficient=0.85, confidence=0.92))
        return llm

    # Path A: direct service call (the existing, unchanged v2 engine)
    direct_service = DiscretionaryCheckService(repo=_fresh_repo(), llm=_fresh_llm(), request=request)
    direct_result = await direct_service._evaluate_criterion(criterion)

    # Path B: through the agentic tools, same mocked responses
    session_id = await SessionStore.create(bulk_request, lot)
    with patch("tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
               AsyncMock(return_value=(_fresh_repo(), _fresh_llm()))):
        raw = await evaluate_criteria_core(session_id=session_id, criterion_ids=["P1"])
    agentic_result = await SessionStore.get_result(session_id, "ns-oe1", "P1")

    assert agentic_result.coefficient == direct_result.coefficient
    assert agentic_result.score == direct_result.score
    assert agentic_result.confidence == direct_result.confidence
    assert agentic_result.human_review_required == direct_result.human_review_required
    assert agentic_result.citation_attributed == direct_result.citation_attributed
    assert agentic_result.evidence_document == direct_result.evidence_document
