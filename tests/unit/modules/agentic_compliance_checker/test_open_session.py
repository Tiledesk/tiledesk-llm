"""
logic.py — session lifecycle (open/status/close) and the _resolve_deps DI seam (P1).
"""
import fakeredis.aioredis
import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from tilellm.modules.agentic_compliance_checker.logic import (
    _resolve_deps,
    close_session,
    get_session_status,
    open_session,
)
from tilellm.modules.agentic_compliance_checker.models import SessionNotFound
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.compliance_checker.models_v2 import BulkComplianceRequestV2, ComplianceRequestV2, OperatorRef
from tilellm.models import Engine
from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository

_MINIMAL_YAML = """\
tender:
  title: Gara test
  lot_id: L1
  lot_name: Lotto 1
requirements:
  tabular:
    - id: T1
      text: prodotto sterile
  discretionary:
    - id: P1
      text: plasticità
      mode: variabile
      max_points: 8
"""


@pytest.fixture
def fake_redis():
    SessionStore._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield SessionStore._client
    SessionStore._client = None


def _bulk_request(**overrides) -> BulkComplianceRequestV2:
    kwargs = dict(
        requirements_yaml=_MINIMAL_YAML,
        operators=[OperatorRef(namespace="ns-oe1", operator_label="OE 1")],
        engine=Engine(name="qdrant"),
        llm="openai",
        gptkey=SecretStr("sk-test"),
        model="gpt-4o-mini",
    )
    kwargs.update(overrides)
    return BulkComplianceRequestV2(**kwargs)


@pytest.mark.asyncio
async def test_open_session_counts_and_operators(fake_redis):
    response = await open_session(_bulk_request())

    assert response.session_id
    assert response.tender.lot_id == "L1"
    assert response.tabular_count == 1
    assert response.discretionary_count == 1
    assert response.operators == ["OE 1"]
    assert await SessionStore.exists(response.session_id)


@pytest.mark.asyncio
async def test_open_session_falls_back_to_namespace_when_no_operator_label(fake_redis):
    request = _bulk_request(operators=[OperatorRef(namespace="ns-oe1")])

    response = await open_session(request)

    assert response.operators == ["ns-oe1"]


@pytest.mark.asyncio
async def test_get_session_status_roundtrip(fake_redis):
    opened = await open_session(_bulk_request())

    status = await get_session_status(opened.session_id)

    assert status.session_id == opened.session_id
    assert status.tender.lot_id == "L1"
    assert status.created_at != ""


@pytest.mark.asyncio
async def test_get_session_status_unknown_raises(fake_redis):
    with pytest.raises(SessionNotFound):
        await get_session_status("nope")


@pytest.mark.asyncio
async def test_close_session_returns_trace_and_deletes(fake_redis):
    opened = await open_session(_bulk_request())

    trace = await close_session(opened.session_id)

    assert trace == []  # nothing was recorded against this session yet in P1
    assert not await SessionStore.exists(opened.session_id)


@pytest.mark.asyncio
async def test_close_session_unknown_raises(fake_redis):
    with pytest.raises(SessionNotFound):
        await close_session("nope")


@pytest.mark.asyncio
async def test_resolve_deps_builds_repo_and_llm_from_request():
    """The DI seam every tool from P3 onward relies on: a ComplianceRequestV2
    (same shape check_compliance_v2 already uses) resolves to a real repo/llm
    pair via the SAME inject_repo_async/inject_llm_chat_async decorators the
    rest of the app uses — no parallel construction logic in this module."""
    request = ComplianceRequestV2(
        namespace="ns-oe1",
        engine=Engine(name="qdrant"),
        llm="openai",
        gptkey=SecretStr("sk-test"),
        model="gpt-4o-mini",
        requirements_yaml=_MINIMAL_YAML,
    )

    repo, llm = await _resolve_deps(request)

    assert isinstance(repo, QdrantRepository)
    assert isinstance(llm, ChatOpenAI)
