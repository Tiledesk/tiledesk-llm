"""
P7: trace-completeness guardrail (compute_result_digest /
verify_trace_completeness), MinIO archival (archive_report), and the two new
HTTP-only report views (GET .../report, GET .../report/xlsx,
GET .../trace?format=md) that sit alongside the agent-facing
compliance_build_report tool.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from tilellm.models import Engine
from tilellm.models.schemas.retrieval_schemas import RetrievalChunksResult
from tilellm.modules.agentic_compliance_checker.logic import build_full_report, render_trace_markdown
from tilellm.modules.agentic_compliance_checker.models import TraceIncompleteError, TraceRecord
from tilellm.modules.agentic_compliance_checker.services.audit_archive import (
    archive_report,
    compute_result_digest,
    verify_trace_completeness,
)
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.agentic_compliance_checker.services.langchain_tools import (
    AGENTIC_COMPLIANCE_TOOLS,
)
from tilellm.modules.agentic_compliance_checker.services.tools_core import (
    build_report_core,
    check_l01_core,
    evaluate_criteria_core,
)
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    DiscretionaryCriterion,
    DiscretionaryResult,
    OperatorRef,
    TenderInfo,
    TenderLotRequirements,
    _RequirementsBlock,
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
        requirements=_RequirementsBlock(discretionary=[
            DiscretionaryCriterion(id="P1", text="plasticità", mode="variabile", max_points=8),
        ]),
    )


def _judge_response(coefficient=0.8, confidence=0.9):
    resp = MagicMock()
    resp.content = json.dumps({
        "coefficient": coefficient, "measured_value": None, "measured_quantity": None,
        "motivation": "ok", "confidence": confidence, "source_chunk_index": 1,
        "evidence_text": "chunk uno", "capitolato_discrepancy": None,
    })
    return resp


async def _evaluate_p1(session_id):
    repo = AsyncMock()
    repo.get_chunks_from_repo = AsyncMock(return_value=RetrievalChunksResult(
        namespace="ns-oe1", chunks=["chunk uno"], metadata=[{"file_name": "offerta.pdf", "page": 3}],
    ))
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=_judge_response())
    with patch(
        "tilellm.modules.agentic_compliance_checker.services.runner._resolve_deps",
        AsyncMock(return_value=(repo, llm)),
    ):
        await evaluate_criteria_core(session_id=session_id, criterion_ids=["P1"])


# ---------------------------------------------------------------------------
# compute_result_digest / verify_trace_completeness — pure, no I/O
# ---------------------------------------------------------------------------

def _make_result(coefficient=0.8):
    return DiscretionaryResult(
        criterion_id="P1", criterion_text="plasticità", mode="variabile", max_points=8.0,
        coefficient=coefficient, score=coefficient * 8.0, motivation="m", confidence=0.9,
    )


def test_compute_result_digest_is_deterministic_and_content_sensitive():
    a, b, c = _make_result(0.8), _make_result(0.8), _make_result(0.5)
    assert compute_result_digest(a) == compute_result_digest(b)
    assert compute_result_digest(a) != compute_result_digest(c)


def test_verify_trace_completeness_passes_when_digest_is_recorded():
    result = _make_result()
    trace = [TraceRecord(
        seq=1, tool="compliance_evaluate_criteria", session_id="s1", outcome="ok",
        result_digests=[compute_result_digest(result)],
    )]
    verify_trace_completeness("s1", trace, [result], [], None)  # must not raise


def test_verify_trace_completeness_rejects_a_result_with_no_matching_digest():
    result = _make_result()
    trace = [TraceRecord(seq=1, tool="compliance_evaluate_criteria", session_id="s1", outcome="ok")]
    with pytest.raises(TraceIncompleteError) as exc_info:
        verify_trace_completeness("s1", trace, [result], [], None)
    assert exc_info.value.orphans == ["disc:P1"]


def test_verify_trace_completeness_catches_tampering():
    """A digest recorded for the ORIGINAL result does not cover a result
    silently edited afterwards (e.g. a manual Redis write) — the whole point
    of hashing the result's own content."""
    original = _make_result(coefficient=0.8)
    trace = [TraceRecord(
        seq=1, tool="compliance_evaluate_criteria", session_id="s1", outcome="ok",
        result_digests=[compute_result_digest(original)],
    )]
    tampered = _make_result(coefficient=1.0)  # same criterion_id, different content
    with pytest.raises(TraceIncompleteError):
        verify_trace_completeness("s1", trace, [tampered], [], None)


# ---------------------------------------------------------------------------
# End-to-end: evaluate via the traced tool, then verify/report — both the
# happy path and the tamper path, through the real session store.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_report_core_succeeds_after_a_legitimate_evaluation(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    await _evaluate_p1(session_id)

    raw = await build_report_core(session_id=session_id)  # must not raise

    body = json.loads(raw)
    assert body["summary"]["ai_scored_count"] == 1


@pytest.mark.asyncio
async def test_build_report_core_rejects_a_result_written_outside_a_traced_tool(fake_redis):
    """Simulates exactly the threat this guardrail exists for: a result that
    reached Redis WITHOUT going through @traced_tool (a manual edit, a bug
    bypassing evaluate_criteria_core). The bare core raises — this IS the
    guardrail firing at its source, not a caught/reported error."""
    session_id = await SessionStore.create(_request(), _lot())
    await SessionStore.store_result(session_id, "ns-oe1", "P1", _make_result())  # bypasses the traced tool

    with pytest.raises(TraceIncompleteError):
        await build_report_core(session_id=session_id)


@pytest.mark.asyncio
async def test_compliance_build_report_tool_turns_tampering_into_a_json_error(fake_redis):
    """The LangChain adapter's _safe() wrapping (services/langchain_tools.py)
    must catch TraceIncompleteError the same way it already catches
    SessionNotFound/ValueError — an agent gets a readable error string back,
    not a raised exception ending its turn."""
    session_id = await SessionStore.create(_request(), _lot())
    await SessionStore.store_result(session_id, "ns-oe1", "P1", _make_result())

    result = await AGENTIC_COMPLIANCE_TOOLS["compliance_build_report"].ainvoke({"session_id": session_id})

    assert "error" in json.loads(result)


@pytest.mark.asyncio
async def test_build_full_report_matches_stored_state_and_includes_tabular_l01(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    await _evaluate_p1(session_id)
    await check_l01_core(session_id=session_id)  # no l01_xlsx_url configured -> used=False, still stored

    report = await build_full_report(session_id, None)

    assert report.namespace == "ns-oe1"
    assert len(report.discretionary_results) == 1
    assert report.discretionary_results[0].coefficient == 0.8
    assert report.l01_check is not None and report.l01_check.used is False


@pytest.mark.asyncio
async def test_build_full_report_l01_check_is_none_when_never_run(fake_redis):
    """Unlike v1/v2's all-in-one evaluate_lot (which always runs _check_l01,
    defaulting to used=False), the agentic flow only has an L01 result to
    report if compliance_check_l01 was actually called — nobody having asked
    for it is a real, different state from "asked and found nothing to
    check", and the report should say so (None, not a synthesized default)."""
    session_id = await SessionStore.create(_request(), _lot())
    await _evaluate_p1(session_id)

    report = await build_full_report(session_id, None)

    assert report.l01_check is None


@pytest.mark.asyncio
async def test_build_full_report_raises_on_tampering(fake_redis):
    session_id = await SessionStore.create(_request(), _lot())
    await SessionStore.store_result(session_id, "ns-oe1", "P1", _make_result())

    with pytest.raises(TraceIncompleteError):
        await build_full_report(session_id, None)


# ---------------------------------------------------------------------------
# render_trace_markdown
# ---------------------------------------------------------------------------

def test_render_trace_markdown_includes_every_record():
    trace = [
        TraceRecord(seq=1, tool="compliance_list_requirements", session_id="s1", outcome="ok", duration_ms=5),
        TraceRecord(seq=2, tool="compliance_evaluate_criteria", session_id="s1", outcome="error", error="boom"),
    ]
    md = render_trace_markdown(trace)
    assert "compliance_list_requirements" in md
    assert "compliance_evaluate_criteria" in md
    assert "boom" in md


# ---------------------------------------------------------------------------
# archive_report — MinIO, best-effort
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_archive_report_returns_none_when_minio_unconfigured():
    with patch(
        "tilellm.shared.minio_storage.get_minio_storage_service",
        side_effect=ValueError("MinIO configuration not found"),
    ):
        uri = await archive_report("sess1", "ns-oe1", "{}", [])
    assert uri is None


@pytest.mark.asyncio
async def test_archive_report_returns_none_on_upload_failure():
    fake_svc = MagicMock()
    fake_svc.upload_data.side_effect = RuntimeError("s3 down")
    with patch("tilellm.shared.minio_storage.get_minio_storage_service", return_value=fake_svc):
        uri = await archive_report("sess1", "ns-oe1", "{}", [])
    assert uri is None


@pytest.mark.asyncio
async def test_archive_report_uploads_and_returns_s3_uri():
    fake_svc = MagicMock()
    with patch("tilellm.shared.minio_storage.get_minio_storage_service", return_value=fake_svc):
        uri = await archive_report("sess1", "ns-oe1", '{"summary": {}}', [])

    assert uri is not None
    assert uri.startswith("s3://")
    assert "sess1" in uri and "ns-oe1" in uri
    fake_svc.upload_data.assert_called_once()
    call_kwargs = fake_svc.upload_data.call_args
    uploaded_bytes = call_kwargs.args[2] if len(call_kwargs.args) > 2 else call_kwargs.kwargs["data"]
    payload = json.loads(uploaded_bytes)
    assert payload["session_id"] == "sess1"
    assert payload["report"] == {"summary": {}}


# ---------------------------------------------------------------------------
# HTTP endpoints (real ASGI app via the global `client` fixture)
# ---------------------------------------------------------------------------

_MINIMAL_YAML = """\
tender:
  title: Gara test
  lot_id: L1
  lot_name: Lotto 1
requirements:
  discretionary:
    - id: P1
      text: plasticità
      mode: variabile
      max_points: 8
"""


def _open_session_http(client):
    payload = {
        "requirements_yaml": _MINIMAL_YAML,
        "operators": [{"namespace": "ns-oe1", "operator_label": "OE 1"}],
        "engine": {"name": "qdrant"},
        "llm": "openai",
        "gptkey": "sk-test-secret",
        "model": "gpt-4o-mini",
    }
    return client.post("/api/agentic-compliance/sessions", json=payload).json()["session_id"]


def test_get_report_returns_full_report_with_no_artifact_when_minio_unavailable(client, fake_redis):
    session_id = _open_session_http(client)

    resp = client.get(f"/api/agentic-compliance/sessions/{session_id}/report")

    assert resp.status_code == 200
    body = resp.json()
    assert body["report"]["namespace"] == "ns-oe1"
    assert body["trace"] == []
    assert body["artifact_uri"] is None  # no real MinIO in this test env


def test_get_report_409s_on_tampered_result(client, fake_redis):
    session_id = _open_session_http(client)
    import asyncio
    asyncio.get_event_loop().run_until_complete(
        SessionStore.store_result(session_id, "ns-oe1", "P1", _make_result())
    )

    resp = client.get(f"/api/agentic-compliance/sessions/{session_id}/report")

    assert resp.status_code == 409


def test_get_report_xlsx_returns_a_workbook(client, fake_redis):
    session_id = _open_session_http(client)

    resp = client.get(f"/api/agentic-compliance/sessions/{session_id}/report/xlsx")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert len(resp.content) > 0


def test_get_trace_format_md_returns_markdown(client, fake_redis):
    session_id = _open_session_http(client)

    resp = client.get(f"/api/agentic-compliance/sessions/{session_id}/trace?format=md")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "# Traccia di audit" in resp.text


def test_get_trace_default_format_is_unchanged_json(client, fake_redis):
    session_id = _open_session_http(client)

    resp = client.get(f"/api/agentic-compliance/sessions/{session_id}/trace")

    assert resp.status_code == 200
    assert resp.json() == []
