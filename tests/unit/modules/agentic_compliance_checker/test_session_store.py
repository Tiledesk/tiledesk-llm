"""
SessionStore — Redis-backed session state (P1).

Why fakeredis and not a mocked client: the whole point of this store is Redis'
own atomicity guarantees (HSET, RPUSH, INCR) under concurrent access from
different gunicorn workers — a MagicMock would happily "succeed" on a broken
implementation. fakeredis exercises the real Redis command semantics in-process.
"""
import asyncio

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from tilellm.modules.agentic_compliance_checker.models import SessionNotFound, TraceRecord
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    DiscretionaryCriterion,
    OperatorRef,
    TenderInfo,
    TenderLotRequirements,
    _RequirementsBlock,
)
from tilellm.models import Engine


@pytest.fixture
def fake_redis():
    """Points SessionStore at an in-process fakeredis instance for the duration
    of the test, then resets the lazy singleton so other tests get a clean one."""
    SessionStore._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield SessionStore._client
    SessionStore._client = None


def _request(**overrides) -> BulkComplianceRequestV2:
    kwargs = dict(
        requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\nrequirements: {}\n",
        operators=[OperatorRef(namespace="ns-oe1")],
        engine=Engine(name="qdrant"),
        llm="openai",
        gptkey=SecretStr("sk-real-secret-value"),
        model="gpt-4o-mini",
    )
    kwargs.update(overrides)
    return BulkComplianceRequestV2(**kwargs)


def _lot() -> TenderLotRequirements:
    return TenderLotRequirements(
        tender=TenderInfo(title="Gara test", lot_id="L1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(
            discretionary=[
                DiscretionaryCriterion(id="P1", text="criterio", mode="variabile", max_points=8)
            ]
        ),
    )


@pytest.mark.asyncio
async def test_create_and_get_request_roundtrip_preserves_real_secret(fake_redis):
    """The gptkey must survive the round-trip as the REAL secret, not the
    "**********" pydantic model_dump(mode="json") would mask it as — this is
    the whole reason _serialize_request patches it back in after dumping."""
    session_id = await SessionStore.create(_request(), _lot())

    reloaded = await SessionStore.get_request(session_id)

    assert reloaded.gptkey.get_secret_value() == "sk-real-secret-value"
    assert reloaded.operators[0].namespace == "ns-oe1"


@pytest.mark.asyncio
async def test_get_lot_roundtrip(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())

    lot = await SessionStore.get_lot(session_id)

    assert lot.tender.lot_id == "L1"
    assert lot.requirements.discretionary[0].id == "P1"


@pytest.mark.asyncio
async def test_unknown_session_raises_session_not_found(fake_redis):
    with pytest.raises(SessionNotFound):
        await SessionStore.get_request("does-not-exist")
    with pytest.raises(SessionNotFound):
        await SessionStore.get_lot("does-not-exist")
    with pytest.raises(SessionNotFound):
        await SessionStore.append_trace("does-not-exist", TraceRecord(seq=0, tool="x", session_id="does-not-exist"))


@pytest.mark.asyncio
async def test_concurrent_trace_appends_both_land_with_distinct_seq(fake_redis):
    """Regression test for the lost-update failure mode a single JSON blob would
    have: compliance_evaluate_criteria (P3) fans out with asyncio.gather, so two
    tool calls append to the same session's trace concurrently."""
    session_id = await SessionStore.create(_request(), _lot())

    records = [TraceRecord(seq=0, tool=f"tool_{i}", session_id=session_id) for i in range(10)]
    await asyncio.gather(*[SessionStore.append_trace(session_id, r) for r in records])

    trace = await SessionStore.get_trace(session_id)
    assert len(trace) == 10
    assert sorted(r.seq for r in trace) == list(range(1, 11))  # all distinct, none lost


@pytest.mark.asyncio
async def test_delete_returns_trace_and_removes_session(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    await SessionStore.append_trace(session_id, TraceRecord(seq=0, tool="compliance_list_requirements", session_id=session_id))

    trace = await SessionStore.delete(session_id)

    assert len(trace) == 1
    assert trace[0].tool == "compliance_list_requirements"
    assert not await SessionStore.exists(session_id)


@pytest.mark.asyncio
async def test_touch_refreshes_ttl_on_all_session_keys(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    await SessionStore.append_trace(session_id, TraceRecord(seq=0, tool="x", session_id=session_id))

    await SessionStore.touch(session_id)

    from tilellm.modules.agentic_compliance_checker.services.session_store import (
        SESSION_TTL_SECONDS,
        _session_key,
        _trace_key,
    )
    session_ttl = await fake_redis.ttl(_session_key(session_id))
    trace_ttl = await fake_redis.ttl(_trace_key(session_id))
    assert 0 < session_ttl <= SESSION_TTL_SECONDS
    assert 0 < trace_ttl <= SESSION_TTL_SECONDS
