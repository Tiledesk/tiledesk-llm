"""
HTTP layer (P1): session open/status/delete/trace through the real FastAPI app.
"""
import fakeredis.aioredis
import pytest

from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore

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


@pytest.fixture
def fake_redis():
    SessionStore._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield SessionStore._client
    SessionStore._client = None


def _open_payload():
    return {
        "requirements_yaml": _MINIMAL_YAML,
        "operators": [{"namespace": "ns-oe1", "operator_label": "OE 1"}],
        "engine": {"name": "qdrant"},
        "llm": "openai",
        "gptkey": "sk-test-secret",
        "model": "gpt-4o-mini",
    }


def test_open_session_returns_id_and_counts(client, fake_redis):
    resp = client.post("/api/agentic-compliance/sessions", json=_open_payload())

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"]
    assert body["discretionary_count"] == 1
    assert body["operators"] == ["OE 1"]


def test_open_session_response_never_echoes_gptkey(client, fake_redis):
    resp = client.post("/api/agentic-compliance/sessions", json=_open_payload())

    assert "sk-test-secret" not in resp.text


def test_get_session_roundtrip(client, fake_redis):
    session_id = client.post("/api/agentic-compliance/sessions", json=_open_payload()).json()["session_id"]

    resp = client.get(f"/api/agentic-compliance/sessions/{session_id}")

    assert resp.status_code == 200
    assert resp.json()["tender"]["lot_id"] == "L1"


def test_get_unknown_session_is_404(client, fake_redis):
    resp = client.get("/api/agentic-compliance/sessions/does-not-exist")
    assert resp.status_code == 404


def test_trace_starts_empty(client, fake_redis):
    session_id = client.post("/api/agentic-compliance/sessions", json=_open_payload()).json()["session_id"]

    resp = client.get(f"/api/agentic-compliance/sessions/{session_id}/trace")

    assert resp.status_code == 200
    assert resp.json() == []


def test_trace_for_unknown_session_is_404(client, fake_redis):
    resp = client.get("/api/agentic-compliance/sessions/does-not-exist/trace")
    assert resp.status_code == 404


def test_delete_session_closes_it(client, fake_redis):
    session_id = client.post("/api/agentic-compliance/sessions", json=_open_payload()).json()["session_id"]

    del_resp = client.delete(f"/api/agentic-compliance/sessions/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == []

    assert client.get(f"/api/agentic-compliance/sessions/{session_id}").status_code == 404
