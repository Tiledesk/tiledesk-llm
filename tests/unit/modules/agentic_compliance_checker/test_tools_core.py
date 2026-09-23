"""
tools_core.py — @traced_tool + compliance_list_requirements (P2).
"""
import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from tilellm.modules.agentic_compliance_checker.models import SessionNotFound
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.agentic_compliance_checker.services.tools_core import (
    list_requirements_core,
    record_trace_detail,
    traced_tool,
)
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    DiscretionaryCriterion,
    OperatorRef,
    TabularRequirementV2,
    TenderInfo,
    TenderLotRequirements,
    _RequirementsBlock,
)
from tilellm.models import Engine

import json


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
            tabular=[TabularRequirementV2(id="T1", text="prodotto sterile")],
            discretionary=[
                DiscretionaryCriterion(id="P1", text="plasticità", mode="variabile", max_points=8),
            ],
        ),
    )


@pytest.mark.asyncio
async def test_list_requirements_returns_all_by_default(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())

    raw = await list_requirements_core(session_id=session_id)
    body = json.loads(raw)

    assert body["tender"]["lot_id"] == "L1"
    assert body["operators"] == ["OE 1"]
    assert body["totals"] == {"tabular": 1, "discretionary": 1}
    ids = {r["id"] for r in body["requirements"]}
    assert ids == {"T1", "P1"}


@pytest.mark.asyncio
async def test_list_requirements_filters_by_kind(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())

    raw = await list_requirements_core(session_id=session_id, kind="discretionary")
    body = json.loads(raw)

    assert [r["id"] for r in body["requirements"]] == ["P1"]
    assert body["requirements"][0]["mode"] == "variabile"
    assert body["requirements"][0]["max_points"] == 8


@pytest.mark.asyncio
async def test_list_requirements_appends_ok_trace_record(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())

    await list_requirements_core(session_id=session_id)

    trace = await SessionStore.get_trace(session_id)
    assert len(trace) == 1
    record = trace[0]
    assert record.tool == "compliance_list_requirements"
    assert record.outcome == "ok"
    assert record.session_id == session_id
    assert record.seq == 1
    assert record.duration_ms >= 0
    assert "session_id" not in record.args  # redacted — it's the trace's own key, not an arg to log


@pytest.mark.asyncio
async def test_list_requirements_unknown_session_raises_and_appends_no_trace(fake_redis):
    with pytest.raises(SessionNotFound):
        await list_requirements_core(session_id="does-not-exist")
    # No session ever existed to append an error trace against — must not raise
    # a second, confusing exception out of the tracing machinery itself.


@pytest.mark.asyncio
async def test_traced_tool_requires_session_id_as_keyword():
    @traced_tool("dummy")
    async def core(session_id: str) -> str:
        return "ok"

    with pytest.raises(TypeError):
        await core("positional-not-allowed")


@pytest.mark.asyncio
async def test_traced_tool_records_error_outcome_on_exception(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())

    @traced_tool("dummy_failing")
    async def core(*, session_id: str) -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await core(session_id=session_id)

    trace = await SessionStore.get_trace(session_id)
    assert len(trace) == 1
    assert trace[0].outcome == "error"
    assert trace[0].error == "boom"


@pytest.mark.asyncio
async def test_record_trace_detail_is_attached_to_the_trace_record(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())

    @traced_tool("dummy_with_detail")
    async def core(*, session_id: str) -> str:
        record_trace_detail(llm={"model": "gpt-4o-mini", "total_tokens": 42})
        return "ok"

    await core(session_id=session_id)

    trace = await SessionStore.get_trace(session_id)
    assert trace[0].llm == {"model": "gpt-4o-mini", "total_tokens": 42}


@pytest.mark.asyncio
async def test_record_trace_detail_outside_traced_call_is_a_noop():
    record_trace_detail(llm={"should": "be ignored, no crash"})
