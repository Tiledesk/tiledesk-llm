"""Redis (Query Engine / RediSearch) as a dense-only vector store for small namespaces.

Layout (ADR-0009): one HNSW/COSINE index per (index_name, namespace), HASH keys
``tlvs:{index_name}:{namespace}:{chunk_id}`` with ``embedding`` (FLOAT32), ``text``, ``metadata`` (JSON) and a
fixed set of indexed copies of metadata fields (TAG/NUMERIC) for filters. No hybrid/BM25/SPLADE: the
"hybrid" methods run the dense KNN only. Redis Search indexes only DB 0.
"""
import asyncio
import json
import logging
import math
import os
import re
import uuid
from collections import OrderedDict
from array import array
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import quote, unquote

import redis.asyncio as aioredis
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ResponseError

from tilellm.models import Engine, ItemSingle, QuestionAnswer
from tilellm.models.llm import TEIConfig
from tilellm.models.schemas import (IndexingResult, RepositoryDescNamespaceResult, RepositoryIdSummaryResult,
                                    RepositoryItemNamespaceResult, RepositoryItems, RepositoryNamespace,
                                    RepositoryNamespaceResult, RepositoryQueryResult, RetrievalChunksResult)
from tilellm.shared.embeddings.embedding_client_manager import (inject_embedding_async_optimized,
                                                                inject_embedding_qa_async_optimized)
from tilellm.shared.untrusted_content import strip_invisible_documents
from tilellm.shared.utility import _hash_api_key
from tilellm.store.vector_store_repository import VectorStoreIndexingError, VectorStoreRepository
from tilellm.tools.document_tools import calc_embedding_cost

logger = logging.getLogger(__name__)

KEY_PREFIX = "tlvs"
# Metadata fields copied into indexed hash fields: the only fields a filter may use (ADR-0009).
TAG_FIELDS = ("id", "metadata_id", "doc_id", "source", "type", "tags", "doc_type", "file_name",
              "digest_type", "digest_granularity")
# "date" is temporal_digest's default date_metadata_field; tenants with their own date field list it in
# REDIS_VS_EXTRA_DATE_FIELDS (indexes created before the change must be rebuilt to pick it up).
NUMERIC_FIELDS = ("chunk_index", "page", "date", "digest_date_from", "digest_date_to") + tuple(
    f.strip() for f in os.environ.get("REDIS_VS_EXTRA_DATE_FIELDS", "").split(",") if f.strip())
TAG_SEP = "\x1f"  # tag values may contain commas
# RediSearch has no "match nothing" ("-*" is a syntax error): an empty exclusive range never matches.
NO_MATCH = "@chunk_index:[(0 (0]"
HNSW_M = int(os.environ.get("REDIS_VS_HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.environ.get("REDIS_VS_HNSW_EF_CONSTRUCTION", "200"))
# Redis' default EF_RUNTIME (10) lost ~16% of the true top-5 in RV0; 50 kept recall 0.996 at ~0.6 ms.
HNSW_EF_RUNTIME = int(os.environ.get("REDIS_VS_HNSW_EF_RUNTIME", "50"))
_BATCH = 500
# The host comes from the request: a server that accepts and never answers must not hang the request.
SOCKET_TIMEOUT = float(os.environ.get("REDIS_VS_SOCKET_TIMEOUT", "5"))
MAX_CONNECTIONS = int(os.environ.get("REDIS_VS_MAX_CONNECTIONS", "50"))
MAX_CLIENTS_PER_LOOP = 32

# One client per (event loop, connection): redis.asyncio clients are bound to the loop that created them.
# LRU-bounded: hosts/passwords come from requests, an unbounded dict would grow without limit.
# A WeakKeyDictionary would never drop a loop (its clients reference it): closed loops are purged instead.
_clients: "Dict[asyncio.AbstractEventLoop, OrderedDict[tuple, aioredis.Redis]]" = {}


def _client(engine: Engine) -> aioredis.Redis:
    password = engine.apikey.get_secret_value() if engine.apikey else None
    ssl = engine.deployment == "cloud"
    key = (engine.host, engine.port, _hash_api_key(password) if password else None, ssl)
    per_loop = _clients.setdefault(asyncio.get_running_loop(), OrderedDict())
    if key in per_loop:
        per_loop.move_to_end(key)
        return per_loop[key]
    for loop in [lp for lp in _clients if lp.is_closed()]:
        del _clients[loop]
    per_loop[key] = aioredis.Redis(host=engine.host, port=engine.port, password=password, db=0, ssl=ssl,
                                   socket_timeout=SOCKET_TIMEOUT, socket_connect_timeout=SOCKET_TIMEOUT,
                                   max_connections=MAX_CONNECTIONS,
                                   # one retry for a dropped connection; the default (3 + backoff) multiplies the timeout
                                   retry=Retry(ExponentialBackoff(cap=0.2), 1))
    while len(per_loop) > MAX_CLIENTS_PER_LOOP:
        _, old = per_loop.popitem(last=False)
        asyncio.get_running_loop().create_task(old.aclose())
    return per_loop[key]


def _index(engine: Engine, namespace: str) -> str:
    # quote() turns ':' into %3A, so namespace "a" can never prefix-match the keys of namespace "a:b".
    return f"{KEY_PREFIX}:{quote(engine.index_name, safe='')}:{quote(namespace, safe='')}"


def _key(engine: Engine, namespace: str, chunk_id: str) -> str:
    return f"{_index(engine, namespace)}:{quote(str(chunk_id), safe='')}"


def _chunk_id(key: bytes) -> str:
    # Every part of the key is quoted, so the last ':' is the one before the chunk id.
    return unquote(key.decode().rsplit(":", 1)[1])


def _missing_index(e: Exception) -> bool:
    msg = str(e).lower()
    return "no such index" in msg or "unknown index" in msg


# --- filters: Pinecone-style dict -> RediSearch query (values escaped, fields whitelisted) ------------

def _esc(value: Any) -> str:
    text = str(value)
    if not text:
        raise ValueError("Redis vector store: empty tag value in filter")
    return re.sub(r"([^\w])", r"\\\1", text)


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None  # Redis refuses to index the whole hash
    if isinstance(value, str):
        try:
            n = float(value)
            return n if math.isfinite(n) else None
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
    return None


def _num_or_raise(field: str, value: Any) -> float:
    n = _num(value)
    if n is None:
        raise ValueError(f"Redis vector store: '{field}' needs a number or an ISO date, got {value!r}")
    return n


def _tag_cond(field: str, op: str, val: Any) -> str:
    values = val if isinstance(val, (list, tuple)) else [val]
    if op in ("$in", "$nin") and not values:
        return "*" if op == "$nin" else NO_MATCH
    expr = f"@{field}:{{{' | '.join(_esc(v) for v in values)}}}"
    if op in ("$eq", "$in"):
        return expr
    if op in ("$ne", "$nin"):
        return f"-{expr}"
    raise ValueError(f"Redis vector store: operator {op} not supported on '{field}'")


def _num_cond(field: str, op: str, val: Any) -> str:
    if op in ("$in", "$nin") and not val:
        return "*" if op == "$nin" else NO_MATCH
    if op in ("$in", "$nin"):
        alts = " | ".join(f"@{field}:[{n} {n}]" for n in (_num_or_raise(field, v) for v in val))
        return f"({alts})" if op == "$in" else f"-({alts})"
    n = _num_or_raise(field, val)
    rng = {"$eq": f"[{n} {n}]", "$ne": f"[{n} {n}]", "$gte": f"[{n} +inf]", "$gt": f"[({n} +inf]",
           "$lte": f"[-inf {n}]", "$lt": f"[-inf ({n}]"}.get(op)
    if rng is None:
        raise ValueError(f"Redis vector store: operator {op} not supported on '{field}'")
    return f"{'-' if op == '$ne' else ''}@{field}:{rng}"


def filter_to_query(filter_dict: Optional[Dict]) -> str:
    """Translate a Pinecone-style filter. Fields outside TAG_FIELDS/NUMERIC_FIELDS are rejected:
    a post-filter would cost latency and silently lose matches."""
    if not filter_dict:
        return "*"
    parts = []
    for field, cond in filter_dict.items():
        if field == "$and":
            parts.extend(f"({filter_to_query(c)})" for c in cond)  # empty $and: no constraint
        elif field == "$or":
            parts.append("(" + " | ".join(f"({filter_to_query(c)})" for c in cond) + ")" if cond else NO_MATCH)
        elif field == "$not":
            parts.append(f"-({filter_to_query(cond)})")
        elif field in TAG_FIELDS or field in NUMERIC_FIELDS:
            ops = cond if isinstance(cond, dict) else {"$eq": cond}
            build = _tag_cond if field in TAG_FIELDS else _num_cond
            parts.extend(build(field, op, val) for op, val in ops.items())
        else:
            raise ValueError(f"Redis vector store: cannot filter on non-indexed field '{field}' "
                             f"(indexed: {', '.join(TAG_FIELDS + NUMERIC_FIELDS)})")
    return " ".join(parts) or "*"


def _merge_filters(*filters: Optional[Dict]) -> Optional[Dict]:
    present = [f for f in filters if f]
    if len(present) > 1:
        return {"$and": present}
    return present[0] if present else None


def _indexed_fields(meta: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for f in TAG_FIELDS:
        v = meta.get(f)
        vals = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v]) if x not in (None, "")]
        if vals:
            out[f] = TAG_SEP.join(vals)
    for f in NUMERIC_FIELDS:
        n = _num(meta.get(f)) if meta.get(f) not in (None, "") else None
        if n is not None:
            out[f] = repr(n)
    return out


def _decode(raw: Dict[bytes, bytes]) -> Tuple[str, Dict[str, Any]]:
    text = (raw.get(b"text") or b"").decode()
    meta = json.loads(raw.get(b"metadata") or b"{}")
    return text, meta


class RedisVectorStore(VectorStore):
    """Minimal LangChain VectorStore over RedisRepository, for as_retriever() callers."""

    def __init__(self, repo: "RedisRepository", engine: Engine, embeddings):
        self.repo, self.engine, self._embeddings = repo, engine, embeddings

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def client(self):
        return _client(self.engine)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        raise NotImplementedError("Redis vector store is async-only")

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        raise NotImplementedError("Use RedisRepository.aadd_documents")

    async def asimilarity_search_with_score(self, query: str, k: int = 4, namespace: Optional[str] = None,
                                            filter: Optional[Dict] = None, **kwargs) -> List[Tuple[Document, float]]:
        if namespace is None:
            raise ValueError("Redis vector store: namespace is required")
        vector = await self._embeddings.aembed_query(query)
        return await self.repo.knn(self.engine, namespace, vector, k, filter)

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        return [d for d, _ in await self.asimilarity_search_with_score(query, k, **kwargs)]

    def _select_relevance_score_fn(self):
        return lambda score: score  # knn() already returns cosine similarity


class RedisRepository(VectorStoreRepository):
    sparse_enabled = False

    # --- low level ---------------------------------------------------------------------------------

    @staticmethod
    async def _ensure_index(engine: Engine, namespace: str, dim: int) -> None:
        """Create the namespace index, or check the existing one has the same dimension: Redis skips
        (does not reject) a vector of the wrong size, so a mismatch would lose chunks silently."""
        r, name = _client(engine), _index(engine, namespace)
        schema = []
        for f in TAG_FIELDS:
            schema += [f, "TAG", "SEPARATOR", TAG_SEP, "CASESENSITIVE"]
        for f in NUMERIC_FIELDS:
            schema += [f, "NUMERIC"]
        schema += ["embedding", "VECTOR", "HNSW", 10, "TYPE", "FLOAT32", "DIM", dim, "DISTANCE_METRIC", "COSINE",
                   "M", HNSW_M, "EF_CONSTRUCTION", HNSW_EF_CONSTRUCTION]
        try:
            await r.execute_command("FT.CREATE", name, "ON", "HASH", "PREFIX", 1, f"{name}:", "SCHEMA", *schema)
            return
        except ResponseError as e:
            if "already exists" not in str(e).lower():
                raise
        existing = await RedisRepository._index_dim(r, name)
        if existing is not None and existing != dim:
            raise ValueError(f"Redis vector store: namespace '{namespace}' has dimension {existing}, "
                             f"embeddings have {dim}")

    @staticmethod
    async def _index_dim(r: aioredis.Redis, name: str) -> Optional[int]:
        info = await r.execute_command("FT.INFO", name)
        info = dict(zip(info[::2], info[1::2]))
        for attr in info.get(b"attributes", []):
            a = dict(zip(attr[::2], attr[1::2]))
            if a.get(b"attribute") == b"embedding" and b"dim" in a:
                return int(a[b"dim"])
        return None

    async def _write(self, engine: Engine, namespace: str, docs: List[Document], vectors: List[List[float]]) -> List[str]:
        if not docs:
            return []
        await self._ensure_index(engine, namespace, len(vectors[0]))
        r, ids = _client(engine), []
        for start in range(0, len(docs), _BATCH):
            pipe = r.pipeline(transaction=False)
            for doc, vec in zip(docs[start:start + _BATCH], vectors[start:start + _BATCH]):
                cid = str(doc.id or uuid.uuid4().hex)
                meta = {k: v for k, v in doc.metadata.items() if v is not None}
                meta["namespace"] = namespace
                pipe.hset(_key(engine, namespace, cid), mapping={
                    "embedding": array("f", vec).tobytes(), "text": doc.page_content,
                    "metadata": json.dumps(meta, default=str), **_indexed_fields(meta)})
                ids.append(cid)
            await pipe.execute()
        return ids

    async def knn(self, engine: Engine, namespace: str, vector: List[float], k: int,
                  filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """The latency path: one FT.SEARCH, HNSW, no index checks. Returns (doc, cosine similarity)."""
        if k <= 0:
            return []
        query = f"({filter_to_query(filter_dict)})=>[KNN {k} @embedding $vec EF_RUNTIME {max(HNSW_EF_RUNTIME, k)} AS __dist]"
        try:
            res = await _client(engine).execute_command(
                "FT.SEARCH", _index(engine, namespace), query, "PARAMS", 2, "vec", array("f", vector).tobytes(),
                "SORTBY", "__dist", "RETURN", 3, "text", "metadata", "__dist", "LIMIT", 0, k,
                # with ON_TIMEOUT return (the default) a slow query would come back partial, silently
                "TIMEOUT", 0, "DIALECT", 2)
        except ResponseError as e:
            if _missing_index(e):
                return []
            raise
        out = []
        for key, fields in zip(res[1::2], res[2::2]):
            raw = dict(zip(fields[::2], fields[1::2]))
            text, meta = _decode(raw)
            out.append((Document(id=_chunk_id(key), page_content=text, metadata=meta),
                        1.0 - float(raw[b"__dist"])))
        return out

    async def _scan(self, engine: Engine, namespace: str, query: str = "*",
                    with_text: bool = True) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Every match, with no 10k cap (FT.SEARCH silently returns nothing past offset 10000)."""
        r, name = _client(engine), _index(engine, namespace)
        load = ["@__key", "@metadata"] + (["@text"] if with_text else [])
        try:
            res = await r.execute_command("FT.AGGREGATE", name, query, "LOAD", len(load), *load,
                                          "WITHCURSOR", "COUNT", 1000, "TIMEOUT", 0, "DIALECT", 2)
        except ResponseError as e:
            if _missing_index(e):
                return []
            raise
        out = []
        while True:
            rows, cursor = res
            for row in rows[1:]:
                raw = dict(zip(row[::2], row[1::2]))
                text, meta = _decode(raw)
                out.append((_chunk_id(raw[b"__key"]), text, meta))
            if not cursor:
                return out
            res = await r.execute_command("FT.CURSOR", "READ", name, cursor)

    @staticmethod
    def _items(rows: Iterable[Tuple[str, str, Dict[str, Any]]], with_text: bool = True,
               with_metadata: bool = False) -> RepositoryItems:
        return RepositoryItems(matches=[RepositoryQueryResult(
            id=cid, metadata_id=str(meta.get("id", "")), metadata_source=str(meta.get("source", "")),
            metadata_type=str(meta.get("type", "")), date=str(meta.get("date", "Date not defined")),
            text=text if with_text else None, metadata=meta if with_metadata else None) for cid, text, meta in rows])

    # --- writes ------------------------------------------------------------------------------------

    @inject_embedding_async_optimized()
    async def add_item(self, item: ItemSingle, embedding_obj=None, embedding_dimension=None):
        chunks, total_tokens, cost = [], 0, 0
        try:
            chunks, sc = await self.prepare_item_chunks(item, embedding_obj)
            chunks = strip_invisible_documents(chunks)
            if not chunks:
                raise Exception("No chunks generated from source")
            total_tokens, cost = calc_embedding_cost(chunks, item.embedding)
            vectors = await embedding_obj.aembed_documents([c.page_content for c in chunks])
            # Old chunks go only once the new ones are ready and the index accepts them (dimension check):
            # a failed fetch, embedding or dimension mismatch keeps the document.
            await self._ensure_index(item.engine, item.namespace, len(vectors[0]))
            await self.delete_ids_namespace(item.engine, item.id, item.namespace)
            await self._write(item.engine, item.namespace, chunks, vectors)
            return IndexingResult(id=item.id, chunks=len(chunks), total_tokens=total_tokens, cost=f"{cost:.6f}",
                                  sc_input_tokens=sc["input_tokens"] if sc else None,
                                  sc_output_tokens=sc["output_tokens"] if sc else None,
                                  sc_total_tokens=sc["total_tokens"] if sc else None)
        except Exception as ex:
            logger.error(f"Redis add_item failed: {ex!r}")
            raise VectorStoreIndexingError(IndexingResult(
                id=item.id, chunks=len(chunks), total_tokens=total_tokens, status=400,
                cost=f"{cost:.6f}", error=str(ex)).model_dump()) from ex

    async def add_item_hybrid(self, item):
        logger.warning("Redis vector store is dense-only: hybrid indexing runs as dense")
        return await self.add_item(item)

    async def aadd_documents(self, engine: Engine, documents: list, namespace: str, embedding_model: Any,
                             sparse_encoder: Union[str, TEIConfig, None] = "splade", **kwargs):
        documents = strip_invisible_documents(documents)  # F1
        vectors = await embedding_model.aembed_documents([d.page_content for d in documents]) if documents else []
        if not kwargs.get("skip_delete", False):
            if kwargs.get("metadata_id"):
                if vectors:  # dimension check before deleting the old version
                    await self._ensure_index(engine, namespace, len(vectors[0]))
                await self.delete_ids_namespace(engine, kwargs["metadata_id"], namespace)
            else:
                await self.delete_namespace(RepositoryNamespace(namespace=namespace, engine=engine))
        return await self._write(engine, namespace, documents, vectors)

    async def delete_namespace(self, namespace_to_delete: RepositoryNamespace):
        engine, ns = namespace_to_delete.engine, namespace_to_delete.namespace
        r, name = _client(engine), _index(engine, ns)
        try:
            await r.execute_command("FT.DROPINDEX", name, "DD")
        except ResponseError as e:
            if not _missing_index(e):
                raise
        # DD deletes only indexed documents; hashes Redis failed to index would stay forever.
        # The prefix is quote()d, so it holds no glob characters.
        leftovers = [k async for k in r.scan_iter(match=f"{name}:*", count=1000)]
        for start in range(0, len(leftovers), _BATCH):
            await r.delete(*leftovers[start:start + _BATCH])

    async def _delete_matching(self, engine: Engine, namespace: str, query: str) -> int:
        ids = [cid for cid, _, _ in await self._scan(engine, namespace, query, with_text=False)]
        r = _client(engine)
        for start in range(0, len(ids), _BATCH):
            await r.delete(*(_key(engine, namespace, cid) for cid in ids[start:start + _BATCH]))
        return len(ids)

    async def delete_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str):
        await self._delete_matching(engine, namespace, filter_to_query({"id": metadata_id}))

    async def delete_chunk_id_namespace(self, engine: Engine, chunk_id: str, namespace: str):
        await _client(engine).delete(_key(engine, namespace, chunk_id))

    # --- reads -------------------------------------------------------------------------------------

    @inject_embedding_qa_async_optimized()
    async def get_chunks_from_repo(self, question_answer: QuestionAnswer, embedding_obj=None,
                                   embedding_dimension=None):
        from tilellm.shared.tags_query_parser import build_tags_filter
        start = datetime.now()
        if question_answer.search_type == "hybrid":
            logger.warning("Redis vector store is dense-only: hybrid search runs as dense KNN")
        vector = getattr(question_answer, "precomputed_query_embedding", None) or await embedding_obj.aembed_query(
            question_answer.retrieval_query or question_answer.question)
        flt = _merge_filters(build_tags_filter(question_answer.tags) if question_answer.tags else None,
                             getattr(question_answer, "_metadata_filter", None))
        hits = await self.knn(question_answer.engine, question_answer.namespace, vector, question_answer.top_k, flt)
        return RetrievalChunksResult(
            success=True, namespace=question_answer.namespace, chunks=[d.page_content for d, _ in hits],
            metadata=[d.metadata for d, _ in hits], chunk_ids=[d.id for d, _ in hits],
            duration=(datetime.now() - start).total_seconds())

    async def get_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str) -> RepositoryItems:
        return self._items(await self._scan(engine, namespace, filter_to_query({"id": metadata_id})))

    async def get_sources_namespace(self, engine: Engine, source: str, namespace: str) -> RepositoryItems:
        return self._items(await self._scan(engine, namespace, filter_to_query({"source": source})))

    async def get_all_obj_namespace(self, engine: Engine, namespace: str, with_text: bool = False) -> RepositoryItems:
        return self._items(await self._scan(engine, namespace, with_text=with_text), with_text, with_metadata=True)

    async def get_by_doc_id(self, engine: Engine, namespace: str, doc_id: str) -> List[Document]:
        rows = (await self._scan(engine, namespace, filter_to_query({"doc_id": doc_id}))
                or await self._scan(engine, namespace, filter_to_query({"id": doc_id})))
        return [Document(id=cid, page_content=text, metadata=meta) for cid, text, meta in rows]

    async def get_chunks_by_index(self, engine: Engine, namespace: str, doc_id: str,
                                  chunk_indexes: List[int]) -> List[Document]:
        if not chunk_indexes:
            return []
        query = filter_to_query({"doc_id": doc_id, "chunk_index": {"$in": [int(i) for i in chunk_indexes]}})
        return [Document(id=cid, page_content=text, metadata=meta)
                for cid, text, meta in await self._scan(engine, namespace, query)]

    async def list_namespaces(self, engine: Engine) -> RepositoryNamespaceResult:
        r = _client(engine)
        prefix = f"{KEY_PREFIX}:{quote(engine.index_name, safe='')}:"
        namespaces = []
        for raw in await r.execute_command("FT._LIST"):
            name = raw.decode()
            if name.startswith(prefix) and ":" not in name[len(prefix):]:
                info = await r.execute_command("FT.INFO", name)
                info = dict(zip(info[::2], info[1::2]))
                namespaces.append(RepositoryItemNamespaceResult(
                    namespace=unquote(name[len(prefix):]), vector_count=int(info.get(b"num_docs", 0))))
        return RepositoryNamespaceResult(namespaces=namespaces)

    async def get_desc_namespace(self, engine: Engine, namespace: str) -> RepositoryDescNamespaceResult:
        rows = await self._scan(engine, namespace, with_text=False)
        ids: Dict[str, RepositoryIdSummaryResult] = {}
        for _, _, meta in rows:
            mid = str(meta.get("id", ""))
            if mid in ids:
                ids[mid].chunks_count += 1
            elif mid:
                ids[mid] = RepositoryIdSummaryResult(metadata_id=mid, source=str(meta.get("source", "")),
                                                     chunks_count=1)
        return RepositoryDescNamespaceResult(
            namespace_desc=RepositoryItemNamespaceResult(namespace=namespace, vector_count=len(rows)),
            ids=list(ids.values()))

    # --- LangChain / QA plumbing -------------------------------------------------------------------

    async def create_index(self, engine, embeddings, emb_dimension=None, embedding_config_key=None,
                           cache_suffix=None):
        return RedisVectorStore(self, engine, embeddings)

    async def initialize_embeddings_and_index(self, question_answer, llm_embeddings, emb_dimension=None,
                                              embedding_config_key=None, cache_suffix=None):
        if question_answer.search_type == "hybrid":
            logger.warning("Redis vector store is dense-only: hybrid search runs as dense KNN")
        dim = emb_dimension or getattr(question_answer.embedding, "dimension", None) or question_answer.engine.vector_size
        return dim, None, RedisVectorStore(self, question_answer.engine, llm_embeddings)

    async def perform_hybrid_search(self, question_answer, index, dense_vector, sparse_vector, filter=None):
        flt = _merge_filters(filter, getattr(question_answer, "_metadata_filter", None))
        hits = await self.knn(question_answer.engine, question_answer.namespace, dense_vector,
                              question_answer.top_k, flt)
        return {"matches": [{"id": d.id, "score": score, "text": d.page_content,
                             "metadata": {**d.metadata, "text": d.page_content}} for d, score in hits]}

    async def search_community_report(self, question_answer, index, dense_vector, sparse_vector):
        return await self.perform_hybrid_search(question_answer, index, dense_vector, sparse_vector)

    def build_filter(self, namespace: str, filter_dict: Optional[Dict] = None) -> Optional[Dict]:
        return filter_dict  # namespace is the index itself; filter_to_query() runs inside knn()

    @staticmethod
    async def get_embeddings_dimension(embedding) -> int:
        dim = getattr(embedding, "dimension", None)
        if dim:
            return dim
        if hasattr(embedding, "aembed_query"):
            return len(await embedding.aembed_query("dimension probe"))
        raise ValueError("Redis vector store: cannot infer the embedding dimension; set embedding.dimension")
