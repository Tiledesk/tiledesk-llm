"""
System One decisions over the Jev wire protocol (POST {url}/v1/systemone).

One cached httpx client per (provider, url, api key) — the same pattern as
inject_llm_async. Only what goes into building the client is in the key: questions,
model name and parameters travel with each request, so a cached client never serves
one request's values to another (the 0.12.3-rc6 max_tokens bug).
"""
import asyncio
import hashlib
import logging
import time
from typing import Dict, Optional

import httpx

from tilellm.modules.system_one.models import Answer, SystemOneRequest, SystemOneResponse
from tilellm.modules.system_one.providers import ProviderResponseError, ProviderSpec, get_provider
from tilellm.shared import token_tracking
from tilellm.shared.timed_cache import TimedCache
from tilellm.shared.token_tracking import TokenUsageCollector, TokenUsageRecord

logger = logging.getLogger(__name__)

CACHE_TYPE = "system_one"
ENDPOINT = "/v1/systemone"
PROTOCOL_FIELDS = {"model", "state", "questions"}
# Model latency is 30 ms (GPU) to ~650 ms (Laya on CPU); Jev 70-500 ms end to end.
TIMEOUT = httpx.Timeout(30.0, connect=5.0)
# Test seam: an httpx transport used by every new client (None = real network).
_transport: Optional[httpx.AsyncBaseTransport] = None

# ponytail: clients live for the process (like embeddings) and are never closed on
# eviction; they are bounded by the distinct (provider, url, key) triples in use. Add
# close-on-evict to TimedCache if that set ever becomes large or churns.
TimedCache.set_policy(CACHE_TYPE, timeout_seconds=None, max_size=64, refresh_on_access=True,
                      close_on_evict=False)


class SystemOneError(Exception):
    """A provider failure mapped to the HTTP status the caller should see."""

    def __init__(self, status_code: int, detail: str, headers: Optional[Dict[str, str]] = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


class _LoopBoundClient:
    """One httpx.AsyncClient per event loop. httpx pools connections per loop: a client
    cached in one loop and reused from another failed with "Event loop is closed"
    (found against a real laya-serve; any caller running asyncio.run more than once —
    scripts, workers, TestClient — hits it)."""

    def __init__(self, factory):
        self._factory = factory
        self._loop = None
        self._client: Optional[httpx.AsyncClient] = None

    def get(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            # ponytail: the previous client is dropped, not closed — its loop is gone.
            self._client, self._loop = self._factory(), loop
        return self._client


def _base_url(request: SystemOneRequest, spec: ProviderSpec) -> str:
    return (request.model.url or spec.default_url).rstrip("/")


def _api_key(request: SystemOneRequest) -> Optional[str]:
    key = request.model.api_key.get_secret_value() if request.model.api_key else ""
    return key or None


async def get_client(request: SystemOneRequest) -> httpx.AsyncClient:
    spec = get_provider(request.model.provider)
    base_url, api_key = _base_url(request, spec), _api_key(request)
    key = (spec.name, base_url, hashlib.sha256(api_key.encode()).hexdigest() if api_key else "no_key")

    def _new_client() -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        return httpx.AsyncClient(base_url=base_url, headers=headers, timeout=TIMEOUT, transport=_transport)

    async def _create() -> _LoopBoundClient:
        return _LoopBoundClient(_new_client)

    bound = await TimedCache.async_get(object_type=CACHE_TYPE, key=key, constructor=_create)
    return bound.get()


def _provider_error(spec: ProviderSpec, response: httpx.Response) -> SystemOneError:
    detail = f"{spec.name}: HTTP {response.status_code} {response.text[:300]}".strip()
    status = response.status_code
    if status in (401, 403):
        return SystemOneError(401, detail)
    if status in (400, 422):
        return SystemOneError(422, detail)
    if status in (429, 503, 529):
        retry = response.headers.get("Retry-After")
        return SystemOneError(503, detail, {"Retry-After": retry} if retry else None)
    return SystemOneError(502, detail)  # other 4xx (wrong url/model) and 5xx: upstream's fault


def _normalize(request: SystemOneRequest, spec: ProviderSpec, payload) -> Dict[str, Answer]:
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise SystemOneError(502, f"{spec.name}: risposta senza 'answers'")
    missing = sorted(set(request.questions) - set(answers))
    if missing:
        raise SystemOneError(502, f"{spec.name}: nessuna risposta per {missing}")
    normalized = {}
    for key, question in request.questions.items():
        raw = answers[key]
        if not isinstance(raw, dict):
            raise SystemOneError(502, f"{spec.name}: risposta non valida per '{key}'")
        # clm-serve answers a noul as {"noul": p} without "type": take it from the question.
        try:
            normalized[key] = Answer(**{**raw, "type": raw.get("type", question.type)})
        except ValueError as e:
            raise SystemOneError(502, f"{spec.name}: risposta non valida per '{key}': {e}")
    return normalized


def _track_tokens(request: SystemOneRequest, spec: ProviderSpec, model: str, usage: Dict[str, int]) -> None:
    prompt, completion = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    collector = TokenUsageCollector()
    collector.add(TokenUsageRecord(operation="system_one", model=model, prompt_tokens=prompt,
                                   completion_tokens=completion, total_tokens=prompt + completion))
    token_tracking.emit_analytics(collector, id_project=request.id_project, source="system_one",
                                  provider=spec.name, request_id=request.request_id)


async def decide(request: SystemOneRequest) -> SystemOneResponse:
    spec = get_provider(request.model.provider)
    parameters = request.parameters or {}
    clash = sorted(PROTOCOL_FIELDS & parameters.keys())
    if clash:
        raise SystemOneError(422, f"parameters non può ridefinire i campi del protocollo {clash}")

    body = {
        **parameters,
        "state": request.state,
        "questions": {k: q.model_dump(exclude_none=True) for k, q in request.questions.items()},
    }
    model_name = request.model.name or spec.default_model
    if model_name:
        body["model"] = model_name

    client = await get_client(request)
    started = time.perf_counter()
    try:
        response = await client.post(ENDPOINT, json=body)
    except httpx.TimeoutException as e:
        raise SystemOneError(504, f"{spec.name}: timeout ({e})")
    except httpx.HTTPError as e:
        raise SystemOneError(502, f"{spec.name}: server non raggiungibile ({e})")
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if response.status_code >= 400:
        raise _provider_error(spec, response)
    try:
        payload = response.json()
    except ValueError:
        raise SystemOneError(502, f"{spec.name}: risposta non JSON")

    answers = _normalize(request, spec, payload)
    try:
        warnings = spec.check_response(model_name, payload) if spec.check_response else []
    except ProviderResponseError as e:
        raise SystemOneError(422, str(e))
    usage = {k: int(v) for k, v in (payload.get("usage") or {}).items() if isinstance(v, (int, float))}
    reported_model = payload.get("model") or model_name
    _track_tokens(request, spec, reported_model or spec.name, usage)
    return SystemOneResponse(
        provider=spec.name, model=reported_model, answers=answers, usage=usage,
        latency_ms=latency_ms, warnings=warnings, raw=payload if request.debug else None,
    )
