"""
Redis-backed session state for agentic compliance runs.

In-process state (TimedCache) is NOT an option here: entrypoint.sh starts
gunicorn with WORKERS=3 by default (2-3 in every docker-compose*.yml), worker
class uvicorn.workers.UvicornWorker. Two tool calls of the same agent
conversation can land on different worker processes, and a worker can be
recycled mid-session (--max-requests + jitter). A session that only lives in
one process's memory would work in dev (1 worker) and fail intermittently in
production — and a trace that dies with a worker restart isn't a trace.

Keys per session (never one JSON blob — HSET/RPUSH/INCR are atomic per field,
a single blob would lose writes under the asyncio.gather fan-out later phases
use for batched evaluation):

    acc:sess:{sid}                HASH  config, lot, created_at, id_project, request_id
    acc:sess:{sid}:trace          LIST  RPUSH-only, one TraceRecord JSON per tool call
    acc:sess:{sid}:seq            STR   INCR counter, source of TraceRecord.seq (race-free)
    acc:sess:{sid}:evidence       HASH  evidence_ref -> EvidenceEntry JSON
    acc:sess:{sid}:evidence:order LIST  evidence_ref insertion order, for FIFO eviction
    acc:sess:{sid}:results        HASH  "{namespace}|disc|{criterion_id}" -> DiscretionaryResult JSON
                                         "{namespace}|tab|{requirement_id}" -> ComplianceResult JSON
                                         "{namespace}|l01" -> L01CheckResult JSON
    acc:sess:{sid}:attempts       HASH  "{namespace}|{criterion_id}" -> int

TTL is refreshed on every read/write across all of a session's keys together,
so a session in active use never expires mid-run; one that goes idle expires
after AGENTIC_COMPLIANCE_SESSION_TTL (default 6h — a bulk run with human
pauses between batches is hours, not minutes). Durable audit retention past
that TTL is handled separately (see services/audit_archive.py, P7): Redis is
working state, not the archive.
"""
import json
import logging
import os
import secrets
import time
from typing import List, Optional

import redis.asyncio as aioredis

from tilellm.modules.agentic_compliance_checker.models import (
    EvidenceEntry,
    EvidenceNotFound,
    SessionNotFound,
    TraceRecord,
)
from tilellm.modules.compliance_checker.models import ComplianceResult
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    DiscretionaryResult,
    L01CheckResult,
    TenderLotRequirements,
)

logger = logging.getLogger(__name__)

_DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS = int(os.environ.get("AGENTIC_COMPLIANCE_SESSION_TTL", "21600"))  # 6h
# Caps the COUNT of cached evidence_refs, never the text of a chunk — full
# chunk text is never truncated (see EvidenceEntry docstring). FIFO eviction:
# oldest evidence_ref is dropped first when the cap is hit.
MAX_EVIDENCE_REFS = int(os.environ.get("AGENTIC_COMPLIANCE_MAX_EVIDENCE_REFS", "200"))
_KEY_PREFIX = "acc:sess"


def _session_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}"


def _trace_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}:trace"


def _seq_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}:seq"


def _evidence_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}:evidence"


def _evidence_order_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}:evidence:order"


def _results_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}:results"


def _attempts_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}:attempts"


def _all_keys(session_id: str) -> List[str]:
    return [
        _session_key(session_id),
        _trace_key(session_id),
        _seq_key(session_id),
        _evidence_key(session_id),
        _evidence_order_key(session_id),
        _results_key(session_id),
        _attempts_key(session_id),
    ]


def _result_field(namespace: str, criterion_id: str) -> str:
    return f"{namespace}|disc|{criterion_id}"


def _tabular_result_field(namespace: str, requirement_id: str) -> str:
    return f"{namespace}|tab|{requirement_id}"


def _l01_field(namespace: str) -> str:
    return f"{namespace}|l01"


def _attempts_field(namespace: str, criterion_id: str) -> str:
    return f"{namespace}|{criterion_id}"


def _serialize_request(request: BulkComplianceRequestV2) -> str:
    """model_dump(mode="json") masks SecretStr as "**********" — restore the
    real gptkey value before persisting. Round-tripped by _deserialize_request,
    where a plain string is accepted back into the SecretStr field by pydantic."""
    data = request.model_dump(mode="json")
    if request.gptkey is not None:
        data["gptkey"] = request.gptkey.get_secret_value()
    return json.dumps(data)


def _deserialize_request(raw: str) -> BulkComplianceRequestV2:
    return BulkComplianceRequestV2.model_validate(json.loads(raw))


class SessionStore:
    """Redis-backed session state. All methods are classmethods — no per-instance
    state beyond the shared lazy Redis client (same pattern as SemanticCache)."""

    _client: Optional[aioredis.Redis] = None

    @classmethod
    async def _get_client(cls) -> aioredis.Redis:
        if cls._client is None:
            cls._client = aioredis.from_url(
                _DEFAULT_REDIS_URL, decode_responses=True, socket_connect_timeout=5,
            )
            await cls._client.ping()
        return cls._client

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, request: BulkComplianceRequestV2, lot: TenderLotRequirements) -> str:
        session_id = secrets.token_urlsafe(32)
        client = await cls._get_client()
        mapping = {
            "config": _serialize_request(request),
            "lot": lot.model_dump_json(),
            "created_at": str(time.time()),
            "id_project": request.id_project or "",
            "request_id": request.request_id or "",
        }
        key = _session_key(session_id)
        await client.hset(key, mapping=mapping)
        await client.expire(key, SESSION_TTL_SECONDS)
        return session_id

    @classmethod
    async def exists(cls, session_id: str) -> bool:
        client = await cls._get_client()
        return bool(await client.exists(_session_key(session_id)))

    @classmethod
    async def get_request(cls, session_id: str) -> BulkComplianceRequestV2:
        client = await cls._get_client()
        raw = await client.hget(_session_key(session_id), "config")
        if raw is None:
            raise SessionNotFound(session_id)
        await cls.touch(session_id)
        return _deserialize_request(raw)

    @classmethod
    async def get_lot(cls, session_id: str) -> TenderLotRequirements:
        client = await cls._get_client()
        raw = await client.hget(_session_key(session_id), "lot")
        if raw is None:
            raise SessionNotFound(session_id)
        return TenderLotRequirements.model_validate_json(raw)

    @classmethod
    async def get_meta(cls, session_id: str) -> dict:
        client = await cls._get_client()
        data = await client.hgetall(_session_key(session_id))
        if not data:
            raise SessionNotFound(session_id)
        return data

    @classmethod
    async def touch(cls, session_id: str) -> None:
        """Refresh the TTL on every key of this session together."""
        client = await cls._get_client()
        pipe = client.pipeline()
        for key in _all_keys(session_id):
            pipe.expire(key, SESSION_TTL_SECONDS)
        await pipe.execute()

    @classmethod
    async def delete(cls, session_id: str) -> List[TraceRecord]:
        """Close the session and return its full trace — a DELETE must never
        silently drop the audit trail, only the caller decides to discard it."""
        trace = await cls.get_trace(session_id)
        client = await cls._get_client()
        await client.delete(*_all_keys(session_id))
        return trace

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    @classmethod
    async def append_trace(cls, session_id: str, record: TraceRecord) -> TraceRecord:
        """Assigns record.seq via INCR (atomic, race-free even when two tool
        calls append concurrently via asyncio.gather) then RPUSHes the fully
        formed record. Raises SessionNotFound rather than silently creating an
        orphan trace for a session that never existed or already expired."""
        if not await cls.exists(session_id):
            raise SessionNotFound(session_id)
        client = await cls._get_client()
        record.seq = await client.incr(_seq_key(session_id))
        await client.rpush(_trace_key(session_id), record.model_dump_json())
        await cls.touch(session_id)
        return record

    @classmethod
    async def get_trace(cls, session_id: str) -> List[TraceRecord]:
        client = await cls._get_client()
        raw_records = await client.lrange(_trace_key(session_id), 0, -1)
        return [TraceRecord.model_validate_json(r) for r in raw_records]

    # ------------------------------------------------------------------
    # Evidence (P3) — chunks cached under an opaque ref so an agent's context
    # only ever sees a preview + handle, never the full retrieved text.
    # ------------------------------------------------------------------

    @classmethod
    async def store_evidence(cls, session_id: str, entry: EvidenceEntry) -> None:
        if not await cls.exists(session_id):
            raise SessionNotFound(session_id)
        client = await cls._get_client()
        await client.hset(_evidence_key(session_id), entry.evidence_ref, entry.model_dump_json())
        await client.rpush(_evidence_order_key(session_id), entry.evidence_ref)
        # FIFO eviction on count, never on chunk text (see MAX_EVIDENCE_REFS).
        while await client.llen(_evidence_order_key(session_id)) > MAX_EVIDENCE_REFS:
            oldest = await client.lpop(_evidence_order_key(session_id))
            if oldest is not None:
                await client.hdel(_evidence_key(session_id), oldest)
        await cls.touch(session_id)

    @classmethod
    async def get_evidence(cls, session_id: str, evidence_ref: str) -> EvidenceEntry:
        client = await cls._get_client()
        raw = await client.hget(_evidence_key(session_id), evidence_ref)
        if raw is None:
            if not await cls.exists(session_id):
                raise SessionNotFound(session_id)
            raise EvidenceNotFound(session_id, evidence_ref)
        return EvidenceEntry.model_validate_json(raw)

    # ------------------------------------------------------------------
    # Results (P3) — one DiscretionaryResult per (namespace, criterion_id).
    # ------------------------------------------------------------------

    @classmethod
    async def store_result(
        cls, session_id: str, namespace: str, criterion_id: str, result: DiscretionaryResult,
    ) -> None:
        if not await cls.exists(session_id):
            raise SessionNotFound(session_id)
        client = await cls._get_client()
        await client.hset(
            _results_key(session_id), _result_field(namespace, criterion_id), result.model_dump_json(),
        )
        await cls.touch(session_id)

    @classmethod
    async def get_result(
        cls, session_id: str, namespace: str, criterion_id: str,
    ) -> Optional[DiscretionaryResult]:
        client = await cls._get_client()
        raw = await client.hget(_results_key(session_id), _result_field(namespace, criterion_id))
        return DiscretionaryResult.model_validate_json(raw) if raw is not None else None

    @classmethod
    async def get_results(cls, session_id: str, namespace: Optional[str] = None) -> List[DiscretionaryResult]:
        """All stored discretionary results, optionally filtered to one operator's
        namespace — used by compliance_build_report (P3) and, unfiltered, by
        compliance_resolve_proportional (P5, needs every operator together).

        Always filters to "|disc|" fields — the results HASH also holds tabular
        ("|tab|") and L01 ("|l01") entries (P4) under the same key, which are a
        different schema and would fail DiscretionaryResult validation if let through."""
        client = await cls._get_client()
        raw_map = await client.hgetall(_results_key(session_id))
        prefix = f"{namespace}|disc|" if namespace is not None else None
        results = []
        for field, raw in raw_map.items():
            if prefix is not None:
                if not field.startswith(prefix):
                    continue
            elif "|disc|" not in field:
                continue
            results.append(DiscretionaryResult.model_validate_json(raw))
        return results

    # ------------------------------------------------------------------
    # Tabular results (P4) — one ComplianceResult per (namespace, requirement_id).
    # ------------------------------------------------------------------

    @classmethod
    async def store_tabular_result(
        cls, session_id: str, namespace: str, requirement_id: str, result: ComplianceResult,
    ) -> None:
        if not await cls.exists(session_id):
            raise SessionNotFound(session_id)
        client = await cls._get_client()
        await client.hset(
            _results_key(session_id), _tabular_result_field(namespace, requirement_id), result.model_dump_json(),
        )
        await cls.touch(session_id)

    @classmethod
    async def get_tabular_results(cls, session_id: str, namespace: Optional[str] = None) -> List[ComplianceResult]:
        client = await cls._get_client()
        raw_map = await client.hgetall(_results_key(session_id))
        prefix = f"{namespace}|tab|" if namespace is not None else None
        results = []
        for field, raw in raw_map.items():
            if prefix is not None:
                if not field.startswith(prefix):
                    continue
            elif "|tab|" not in field:
                continue
            results.append(ComplianceResult.model_validate_json(raw))
        return results

    # ------------------------------------------------------------------
    # L01 result (P4) — one L01CheckResult per namespace, zero LLM.
    # ------------------------------------------------------------------

    @classmethod
    async def store_l01_result(cls, session_id: str, namespace: str, result: L01CheckResult) -> None:
        if not await cls.exists(session_id):
            raise SessionNotFound(session_id)
        client = await cls._get_client()
        await client.hset(_results_key(session_id), _l01_field(namespace), result.model_dump_json())
        await cls.touch(session_id)

    @classmethod
    async def get_l01_result(cls, session_id: str, namespace: str) -> Optional[L01CheckResult]:
        client = await cls._get_client()
        raw = await client.hget(_results_key(session_id), _l01_field(namespace))
        return L01CheckResult.model_validate_json(raw) if raw is not None else None

    # ------------------------------------------------------------------
    # Attempts (P3) — how many times a criterion has been (re-)evaluated,
    # for the trace and for the P5 re-roll policy.
    # ------------------------------------------------------------------

    @classmethod
    async def increment_attempts(cls, session_id: str, namespace: str, criterion_id: str) -> int:
        client = await cls._get_client()
        count = await client.hincrby(_attempts_key(session_id), _attempts_field(namespace, criterion_id), 1)
        await cls.touch(session_id)
        return count
