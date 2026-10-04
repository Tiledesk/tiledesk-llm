"""
POST /api/v1/systemone — one client for every server speaking the Jev wire protocol
(TypeSafe Jev, laya-serve, clm-serve): state + typed questions (noul / choice / score)
in, typed answers with calibrated probabilities out.
"""
import json

import httpx
import pytest
from pydantic import ValidationError

from tilellm.modules.system_one import service
from tilellm.modules.system_one.models import SystemOneRequest
from tilellm.shared.timed_cache import TimedCache

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this",
                   "criteria": {"billing": "Payment issues", "technical": "Bugs or integration problems"}},
    "frustration": {"type": "score", "instructions": "How frustrated the customer is",
                    "criteria": ["calm", "frustrated", "angry"]},
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
}
JEV_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {"type": "choice", "choice": "technical", "confidence": 0.78,
                       "probabilities": {"technical": 0.85, "billing": 0.15}},
        "frustration": {"type": "score", "score": 1.0, "confidence": 1.0,
                        "legend": {"0": "calm", "1": "frustrated", "2": "angry"},
                        "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}},
        "is_urgent": {"type": "noul", "noul": 0.97},
    },
    "usage": {"input_tokens": 392, "output_tokens": 65},
}


def _request(**over):
    body = {"model": {"provider": "jev", "api_key": "k"}, "state": "Integration keeps failing.",
            "questions": QUESTIONS}
    body.update(over)
    return body


@pytest.fixture(autouse=True)
def _fresh_cache():
    TimedCache._caches.pop(service.CACHE_TYPE, None)
    yield
    TimedCache._caches.pop(service.CACHE_TYPE, None)


@pytest.fixture
def upstream(monkeypatch):
    """Fake System One server: records requests, answers with `upstream.reply`."""
    calls = []

    def handler(request):
        calls.append(request)
        reply = upstream.reply
        if isinstance(reply, Exception):
            raise reply
        return reply

    upstream.reply = httpx.Response(200, json=JEV_RESPONSE)
    upstream.calls = calls
    monkeypatch.setattr(service, "_transport", httpx.MockTransport(handler))
    return upstream


# ---------------------------------------------------------------------------
# Validation at the API boundary: nothing is silently dropped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad, why", [
    ({"questions": {}}, "at least one question"),
    ({"questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}}, "choice without options"),
    ({"questions": {"q": {"type": "choice", "instructions": "x",
                          "criteria": {str(i): "o" for i in range(256)}}}}, "choice over 255 options"),
    ({"questions": {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}}, "score with 1 level"),
    ({"questions": {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}}},
     "score over 10 levels"),
    ({"questions": {"q": {"type": "rank", "instructions": "x"}}}, "unknown question type"),
    ({"questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}}}, "noul criteria keys"),
    ({"model": {"provider": "openai", "api_key": "k"}}, "unknown provider"),
    ({"model": {"provider": "jev"}}, "jev without api_key"),
    ({"model": {"provider": "laya"}}, "self-hosted provider without url"),
    ({"stream": True}, "unknown field"),
])
def test_invalid_requests_are_rejected(bad, why):
    with pytest.raises(ValidationError):
        SystemOneRequest(**_request(**bad))


def test_self_hosted_providers_need_no_api_key():
    SystemOneRequest(**_request(model={"provider": "laya", "url": "http://laya-serve:8000"}))
    SystemOneRequest(**_request(model={"provider": "clm", "url": "http://clm-serve:8700"}))


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jev_request_goes_to_the_default_endpoint_with_bearer_and_default_model(upstream):
    result = await service.decide(SystemOneRequest(**_request()))

    sent = upstream.calls[0]
    assert str(sent.url) == "https://api.typesafe.ai/v1/systemone"
    assert sent.headers["Authorization"] == "Bearer k"
    body = json.loads(sent.content)
    assert body["model"] == "jev-latest"
    assert body["state"] == "Integration keeps failing."
    assert body["questions"]["department"]["criteria"] == QUESTIONS["department"]["criteria"]
    assert result.answers["department"].choice == "technical"
    assert result.answers["frustration"].score == 1.0
    assert result.answers["is_urgent"].noul == 0.97
    assert result.usage == {"input_tokens": 392, "output_tokens": 65}
    assert result.provider == "jev" and result.model == "jev-1.13.0"
    assert result.raw is None


@pytest.mark.asyncio
async def test_self_hosted_server_url_model_and_parameters(upstream):
    request = SystemOneRequest(**_request(
        model={"provider": "laya", "url": "http://laya-serve:8000/", "name": "/models/laya-gare-ft"},
        parameters={"lang": "it", "max_len": 1024},
    ))

    await service.decide(request)

    sent = upstream.calls[0]
    assert str(sent.url) == "http://laya-serve:8000/v1/systemone"
    assert "Authorization" not in sent.headers
    body = json.loads(sent.content)
    assert body["model"] == "/models/laya-gare-ft"
    assert body["lang"] == "it" and body["max_len"] == 1024


@pytest.mark.asyncio
async def test_parameters_cannot_override_the_protocol_fields(upstream):
    request = SystemOneRequest(**_request(parameters={"state": "other", "questions": {}, "model": "x"}))

    with pytest.raises(service.SystemOneError) as e:
        await service.decide(request)

    assert e.value.status_code == 422
    assert not upstream.calls


@pytest.mark.asyncio
async def test_answer_without_type_takes_it_from_the_question(upstream):
    """clm-serve answers a noul as {"noul": p_true}, with no "type"."""
    upstream.reply = httpx.Response(200, json={"model": "clm-latest", "answers": {"is_urgent": {"noul": 0.4}}})
    request = SystemOneRequest(**_request(model={"provider": "clm", "url": "http://clm:8700"},
                                          questions={"is_urgent": QUESTIONS["is_urgent"]}))

    result = await service.decide(request)

    assert result.answers["is_urgent"].type == "noul"
    assert result.answers["is_urgent"].noul == 0.4


@pytest.mark.asyncio
async def test_missing_answer_is_an_error_not_a_silent_gap(upstream):
    upstream.reply = httpx.Response(200, json={"model": "m", "answers": {"department": JEV_RESPONSE["answers"]["department"]}})

    with pytest.raises(service.SystemOneError) as e:
        await service.decide(SystemOneRequest(**_request()))

    assert e.value.status_code == 502
    assert "frustration" in e.value.detail


@pytest.mark.asyncio
async def test_debug_returns_the_raw_provider_response(upstream):
    upstream.reply = httpx.Response(200, json={**JEV_RESPONSE, "routing": {"model": "english"}})

    result = await service.decide(SystemOneRequest(**_request(debug=True)))

    assert result.raw["routing"] == {"model": "english"}


# ---------------------------------------------------------------------------
# Errors: mapped explicitly, never an empty 200
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("reply, status", [
    (httpx.Response(401, json={"detail": "bad key"}), 401),
    (httpx.Response(422, json={"detail": "bad question"}), 422),
    (httpx.Response(429, headers={"Retry-After": "7"}), 503),
    (httpx.Response(529), 503),
    (httpx.Response(500), 502),
    (httpx.ConnectError("refused"), 502),
    (httpx.ReadTimeout("slow"), 504),
    (httpx.Response(200, text="not json"), 502),
])
async def test_provider_errors_are_mapped(upstream, reply, status):
    upstream.reply = reply

    with pytest.raises(service.SystemOneError) as e:
        await service.decide(SystemOneRequest(**_request()))

    assert e.value.status_code == status
    if status == 503 and isinstance(reply, httpx.Response) and reply.headers.get("Retry-After"):
        assert e.value.headers == {"Retry-After": "7"}


# ---------------------------------------------------------------------------
# Cached client, like inject_llm_async: one per (provider, url, api key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_client_is_cached_per_provider_url_and_key():
    jev = SystemOneRequest(**_request())
    same = SystemOneRequest(**_request(questions={"is_urgent": QUESTIONS["is_urgent"]}, parameters={"x": 1}))
    other_key = SystemOneRequest(**_request(model={"provider": "jev", "api_key": "k2"}))
    other_url = SystemOneRequest(**_request(model={"provider": "jev", "api_key": "k", "url": "https://eu.example"}))

    first = await service.get_client(jev)

    assert await service.get_client(same) is first
    assert await service.get_client(other_key) is not first
    assert await service.get_client(other_url) is not first


# ---------------------------------------------------------------------------
# HTTP layer and default registration
# ---------------------------------------------------------------------------

def test_route_is_registered_by_default():
    from tilellm.__main__ import app

    assert "/api/v1/systemone" in {getattr(r, "path", None) for r in app.routes}


def test_endpoint_returns_normalized_answers(client, upstream):
    response = client.post("/api/v1/systemone", json=_request())

    assert response.status_code == 200
    body = response.json()
    assert body["answers"]["department"]["choice"] == "technical"
    assert "api_key" not in json.dumps(body)


def test_endpoint_maps_provider_errors(client, upstream):
    upstream.reply = httpx.Response(429, headers={"Retry-After": "3"})

    response = client.post("/api/v1/systemone", json=_request())

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "3"


def test_endpoint_rejects_invalid_requests(client, upstream):
    response = client.post("/api/v1/systemone", json=_request(questions={}))

    assert response.status_code == 422
    assert not upstream.calls


def test_providers_endpoint_lists_the_registry(client):
    response = client.get("/api/v1/systemone/providers")

    assert response.status_code == 200
    names = {p["name"] for p in response.json()}
    assert {"jev", "laya", "clm"} <= names


# ---------------------------------------------------------------------------
# Cached client across event loops (found with a real laya-serve, invisible to
# MockTransport): httpx pools connections per event loop, so a client cached in one
# loop and reused from another failed with "Event loop is closed". Real sockets here.
# ---------------------------------------------------------------------------

def test_cached_client_survives_a_new_event_loop():
    import asyncio
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, like laya-serve: the pooled connection is reused

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(JEV_RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        request = SystemOneRequest(**_request(
            model={"provider": "laya", "url": f"http://127.0.0.1:{server.server_address[1]}"}))

        first = asyncio.run(service.decide(request))
        second = asyncio.run(service.decide(request))  # new loop, same cached entry

        assert first.answers["department"].choice == second.answers["department"].choice == "technical"
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Laya-specific response checks (found with a real laya-serve 0.3.27): an unknown
# model name is ignored and the request is silently answered by the base checkpoint
# chosen by language — a mistyped fine-tuned path would return base-model answers with
# HTTP 200. The response's "routing" says which checkpoint really answered and why.
# ---------------------------------------------------------------------------

def _laya_request(name="/models/laya-gare-ft"):
    return SystemOneRequest(**_request(model={"provider": "laya", "url": "http://laya:8000", "name": name}))


def _laya_reply(routing=None, usage=None):
    payload = {**JEV_RESPONSE, "model": "laya-rl-agent", "usage": usage or {"input_tokens": 10}}
    if routing is not None:
        payload["routing"] = routing
    return httpx.Response(200, json=payload)


@pytest.mark.asyncio
async def test_laya_requested_model_not_served_is_an_error(upstream):
    upstream.reply = _laya_reply(routing={"model": "english", "repo": "convaiinnovations/laya",
                                          "reason": "Latin script, ...; using default (english)"})

    with pytest.raises(service.SystemOneError) as e:
        await service.decide(_laya_request())

    assert e.value.status_code == 422
    assert "/models/laya-gare-ft" in e.value.detail and "english" in e.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("name, routing", [
    ("english", {"model": "english", "repo": "convaiinnovations/laya", "reason": "explicit model='english'"}),
    ("/models/laya-gare-ft", {"model": "laya-gare-ft", "repo": "/models/laya-gare-ft", "reason": "explicit model"}),
])
async def test_laya_served_model_matches(upstream, name, routing):
    upstream.reply = _laya_reply(routing=routing)

    result = await service.decide(_laya_request(name))

    assert result.warnings == []


@pytest.mark.asyncio
async def test_laya_without_routing_cannot_verify_the_model(upstream):
    """LAYA_JEV_STRICT=1 drops "routing": the request still succeeds, with a warning."""
    upstream.reply = _laya_reply(routing=None)

    result = await service.decide(_laya_request())

    assert any("non verificabile" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_laya_truncated_state_is_reported(upstream):
    upstream.reply = _laya_reply(
        routing={"model": "english", "repo": "convaiinnovations/laya", "reason": "explicit model='english'"},
        usage={"input_tokens": 600, "state_tokens": 900, "state_tokens_dropped": 388, "truncated": 1})

    result = await service.decide(_laya_request("english"))

    assert any("troncato" in w and "388" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_other_providers_have_no_laya_checks(upstream):
    result = await service.decide(SystemOneRequest(**_request()))

    assert result.warnings == []
