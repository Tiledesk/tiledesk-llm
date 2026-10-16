"""Latency of /api/qa and /api/v2/qa: LLM calls per request, history size, streaming pace,
event-loop blocking and per-request client creation.

The LLM is a fake with a fixed latency that records every call: what is left of the request
time is our code. Vector store and embeddings are in-memory (no network).
"""
import asyncio
import json
import time
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.vectorstores import InMemoryVectorStore

import tilellm.shared.utility as utility
from tilellm.models import QuestionAnswer
from tilellm.models.chat import ChatEntry
from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless

LLM_DELAY = 0.05
CALLS: List[list] = []
QUERIES: List[str] = []  # what reached the vector store


class FakeLLM(BaseChatModel):
    stream_chunks: int = 5

    @property
    def _llm_type(self):
        return "fake-latency"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        raise AssertionError("sync LLM call on the async request path")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
        CALLS.append(messages)
        await asyncio.sleep(LLM_DELAY)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Riparte a settembre."))])

    async def _astream(self, messages, stop=None, run_manager=None, **kw):
        CALLS.append(messages)
        await asyncio.sleep(LLM_DELAY)
        for i in range(self.stream_chunks):
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"t{i} "))


class _Store(InMemoryVectorStore):
    async def asimilarity_search(self, query, k=4, **kwargs):
        QUERIES.append(query)
        kwargs.pop("namespace", None)
        return await super().asimilarity_search(query, k=k, **kwargs)


DOCS = [Document(page_content=t, metadata={"source": f"s{i}", "id": f"d{i}"}) for i, t in enumerate(
    ["Il trasporto scolastico riparte a settembre.", "La biblioteca chiude ad agosto.",
     "Il bilancio ha un avanzo di 120.000 euro."] * 3)]
HISTORY = {str(i): ChatEntry(question=f"domanda {i}", answer=f"risposta {i} " * 20) for i in range(30)}
ENGINE = {"name": "pinecone", "type": "serverless", "apikey": "pc-fake", "index_name": "latency-test"}


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    emb = DeterministicFakeEmbedding(size=8)
    store = _Store(emb)
    store.add_documents(DOCS)

    async def _llm(question):
        return fake

    async def _emb(question):
        return emb

    async def _create_index(self, *a, **k):
        return store

    async def _dim(*a, **k):
        return 8

    async def _init(self, qa, llm_embeddings, *a, **k):
        return 8, None, None

    async def _hybrid(self, qa, index, dense, sparse, filter=None):
        hits = await store.asimilarity_search(qa.question, k=qa.top_k)
        return {"matches": [{"id": d.metadata["id"], "score": 1.0, "metadata": {**d.metadata, "text": d.page_content}}
                            for d in hits]}

    monkeypatch.setattr(utility, "_create_llm_instance", _llm)
    monkeypatch.setattr(utility, "_create_embedding_instance", _emb)
    monkeypatch.setattr(PineconeRepositoryServerless, "create_index", _create_index)
    monkeypatch.setattr(PineconeRepositoryServerless, "get_embeddings_dimension", staticmethod(_dim))
    monkeypatch.setattr(PineconeRepositoryServerless, "initialize_embeddings_and_index", _init)
    monkeypatch.setattr(PineconeRepositoryServerless, "perform_hybrid_search", _hybrid)
    CALLS.clear()
    QUERIES.clear()
    return fake


def _qa(n=[0], **kw):
    n[0] += 1  # unique model name per test: the LLM client cache must not return another test's fake
    return QuestionAnswer(question="Quando riparte il trasporto?", namespace="ns", engine=ENGINE,
                          embedding="text-embedding-3-small", llm="openai", model=f"fake-latency-{n[0]}",
                          gptkey="sk-fake", top_k=3, **kw)


async def _ask(qa):
    from tilellm.controller.controller import ask_hybrid_with_memory, ask_with_memory
    fn = ask_hybrid_with_memory if qa.search_type == "hybrid" else ask_with_memory
    res = await fn(qa)
    if qa.stream:
        body = "".join([c if isinstance(c, str) else c.decode() if isinstance(c, bytes) else json.dumps(c)
                         async for c in res.body_iterator])
        return res, body
    return res, json.loads(res.body)


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", ["similarity", "hybrid"])
async def test_history_is_contextualized_once(llm, search_type):
    res, body = await _ask(_qa(search_type=search_type, chat_history_dict=HISTORY))
    assert res.status_code == 200, body
    assert len(CALLS) == 2  # one rewrite of the question + the answer


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", ["similarity", "hybrid"])
async def test_every_llm_call_respects_max_history_messages(llm, search_type):
    res, body = await _ask(_qa(search_type=search_type, chat_history_dict=HISTORY, max_history_messages=2))
    assert res.status_code == 200, body
    # system + 2 turns (2 messages each) + the question
    assert all(len(messages) <= 1 + 2 * 2 + 1 for messages in CALLS), [len(m) for m in CALLS]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_contextualize_prompt_false_answers(llm, stream):
    res, body = await _ask(_qa(contextualize_prompt=False, chat_history_dict=HISTORY, stream=stream))
    assert res.status_code == 200, body
    assert ("t4" if stream else "settembre") in str(body)
    assert len(CALLS) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("contextualize", [True, False])
async def test_streaming_adds_no_delay_per_chunk(llm, contextualize):
    llm.stream_chunks = 200  # the old 20 ms pause per chunk = 4 s
    t = time.perf_counter()
    res, body = await _ask(_qa(stream=True, contextualize_prompt=contextualize))
    assert res.status_code == 200
    assert "t199" in body
    assert time.perf_counter() - t < 1.5


# --- reranking -------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reranked_retriever_reranks_asynchronously():
    from tilellm.tools.reranker import RerankedRetriever

    class Base(InMemoryVectorStore):
        pass

    store = Base(DeterministicFakeEmbedding(size=8))
    store.add_documents(DOCS)

    class Reranker:
        def rerank_documents(self, *a, **k):
            raise AssertionError("sync rerank on the event loop")

        async def arerank_documents(self, query, documents, top_k, batch_size=8):
            return documents[:top_k]

    r = RerankedRetriever.model_construct(base_retriever=store.as_retriever(search_kwargs={"k": 9}),
                                          reranker=Reranker(), top_k=3, use_reranking=True,
                                          contextualize_query=None)
    assert len(await r.ainvoke("trasporto")) == 3


@pytest.mark.asyncio
async def test_reranker_model_is_loaded_lazily_off_the_event_loop(monkeypatch):
    import threading
    from tilellm.tools.reranker import TileReranker

    loaded_on = []

    class CE:
        def predict(self, pairs, **kw):
            return [0.0] * len(pairs)

    def _load(cls, *a, **k):
        loaded_on.append(threading.current_thread().name)
        return CE()

    monkeypatch.setattr(TileReranker, "_get_cached_model", classmethod(_load))
    r = TileReranker(model_name="local-model")
    assert loaded_on == []  # building it does not load the model on the caller's thread
    await r.arerank_documents("q", DOCS[:5], top_k=2)
    assert loaded_on and loaded_on[0] != threading.current_thread().name


@pytest.mark.asyncio
async def test_tei_reranker_reuses_its_http_client(monkeypatch):
    import httpx
    from tilellm.models.llm import TEIConfig
    from tilellm.tools.reranker import TEIReranker

    created = []
    real = httpx.AsyncClient

    def handler(request):
        texts = json.loads(request.content)["texts"]
        return httpx.Response(200, json=[{"index": i, "score": 1.0 - i / 10} for i in range(len(texts))])

    def factory(*a, **k):
        created.append(1)
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    rr = TEIReranker(TEIConfig(provider="tei", name="r", url="http://tei.example"), 200, 40)
    await rr.arerank_documents("q", DOCS[:3], top_k=2)
    await rr.arerank_documents("q", DOCS[:3], top_k=2)
    assert len(created) == 1


# --- Pinecone ----------------------------------------------------------------------------------------

@pytest.fixture
def pinecone_server():
    """Real HTTP/1.1 keep-alive server answering Pinecone queries; records each client connection."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    connections = set()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            connections.add(self.client_address)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"matches": [{"id": "a", "score": 0.9, "metadata": {"text": "ciao"}}],
                               "namespace": "ns"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", connections
    srv.shutdown()


def _cvs(host):
    from tilellm.models.vector_store import Engine
    from tilellm.store.pinecone.pinecone_repository_base import CachedVectorStore
    cvs = CachedVectorStore(Engine(name="pinecone", type="serverless", apikey="pc-fake", index_name="i"),
                            DeterministicFakeEmbedding(size=8), 8)
    cvs._host = host
    return cvs


@pytest.mark.asyncio
async def test_pinecone_requests_share_one_session(pinecone_server):
    # The real path: langchain_pinecone wraps every async call in `async with index`, whose __aexit__
    # closes the session; the caller then closes the index too. Both must leave the shared one open.
    host, connections = pinecone_server
    cvs = _cvs(host)
    for _ in range(3):
        vs = await cvs.get_vector_store()
        docs = await vs.as_retriever(search_kwargs={"k": 1, "namespace": "ns"}).ainvoke("q")
        assert [d.page_content for d in docs] == ["ciao"]
        await (await cvs.get_index()).close()
    assert len(connections) == 1  # keep-alive: one TCP connection for every request
    await cvs.close()


@pytest.mark.asyncio
async def test_pinecone_shared_index_follows_host_changes_and_really_closes(pinecone_server):
    host, _ = pinecone_server
    cvs = _cvs(host)
    old = await cvs.get_index()
    cvs._host = host.replace("127.0.0.1", "localhost")  # index recreated / key rotated
    new = await cvs.get_index()
    assert new is not old and old._api_client.rest_client._session.closed
    await cvs.close()
    assert new._api_client.rest_client._session.closed


# --- semantic cache L2 -------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_lookup_does_not_round_trip_per_entry(monkeypatch):
    import fakeredis.aioredis
    from tilellm.shared.cache import semantic_cache as sc

    r = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(sc.SemanticCache, "_client", r)
    for i in range(40):
        vec = [1.0 if j == i % 8 else 0.0 for j in range(8)]
        await sc.SemanticCache.store(namespace="ns", question=f"q{i}", embedding=vec,
                                     body={"answer": f"a{i}"}, store_l1=False, store_l2=True)
    calls = []
    real_hgetall = r.hgetall

    async def counting(*a, **k):
        calls.append(a)
        return await real_hgetall(*a, **k)

    monkeypatch.setattr(r, "hgetall", counting)
    hit = await sc.SemanticCache.lookup(namespace="ns", question="x", embedding=[0, 0, 1.0, 0, 0, 0, 0, 0],
                                        check_l1=False, check_l2=True)
    assert hit is not None and hit["answer"] in {f"a{i}" for i in range(2, 40, 8)}
    assert len(calls) <= 1  # one fetch of the winning body, not one per stored entry


# --- /api/v2/qa ----------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_v2_guard_and_intent_router_run_concurrently(llm, monkeypatch):
    from tilellm.agents.workflow import simple_app
    from tilellm.modules.api_v2.models import QASimpleRequest

    class Structured:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, prompt, *a, **k):
            await asyncio.sleep(0.3)
            fields = self.schema.model_fields
            return self.schema(**({"score": "yes", "explanation": "ok"} if "score" in fields else {"intent": "qa"}))

    monkeypatch.setattr(FakeLLM, "with_structured_output", lambda self, schema, **k: Structured(schema),
                        raising=False)
    qa = QASimpleRequest(question="Quando riparte il trasporto?", namespace="ns", engine=ENGINE,
                         embedding="text-embedding-3-small", llm="openai", model="fake-latency-v2",
                         gptkey="sk-fake", top_k=3, use_hallucination_grader=False)
    t = time.perf_counter()
    state = await simple_app.ainvoke({"question_answer": qa, "retry_count": 0, "max_retries": 3,
                                      "metadata": {"search_type": "similarity", "trace": []}})
    elapsed = time.perf_counter() - t
    assert state.get("retrieval_result") is not None
    steps = [s["step"] for s in state["metadata"]["trace"]]
    assert "input_guard" in steps and "intent_router" in steps
    assert elapsed < 0.3 * 2  # guard and intent overlap instead of 0.6 s in sequence


@pytest.mark.asyncio
@pytest.mark.parametrize("graph", ["simple_app", "app"])
async def test_v2_guard_refusal_still_stops_the_graph(llm, monkeypatch, graph):
    from tilellm.agents import workflow
    from tilellm.modules.api_v2.models import QASimpleRequest

    class Structured:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, prompt, *a, **k):
            fields = self.schema.model_fields
            return self.schema(**({"score": "no", "explanation": "off topic"} if "score" in fields
                                  else {"intent": "qa"}))

    monkeypatch.setattr(FakeLLM, "with_structured_output", lambda self, schema, **k: Structured(schema),
                        raising=False)
    qa = QASimpleRequest(question="insulto", namespace="ns", engine=ENGINE, embedding="text-embedding-3-small",
                         llm="openai", model=f"fake-latency-refuse-{graph}", gptkey="sk-fake", top_k=3)
    state = await getattr(workflow, graph).ainvoke({"question_answer": qa, "retry_count": 0, "max_retries": 3,
                                                    "metadata": {"search_type": "similarity", "trace": []}})
    assert state.get("is_on_topic") == "no" and state.get("retrieval_result") is None
    assert CALLS == []  # no RAG answer generated


# --- reranker cache isolation (security review) ---------------------------------------------------------

def test_remote_rerankers_with_same_name_but_other_endpoint_or_key_are_not_shared():
    from pydantic import SecretStr
    from tilellm.models.llm import TEIConfig
    from tilellm.tools.reranker import TileReranker

    TileReranker.clear_cache()
    a = TileReranker(TEIConfig(provider="tei", name="bge", url="https://tenant-a.example",
                               api_key=SecretStr("KEY_A"), custom_headers={"X-T": "a"}))
    b_headers = {"X-T": "b"}
    b = TileReranker(TEIConfig(provider="tei", name="bge", url="https://tenant-b.example",
                               api_key=SecretStr("KEY_B"), custom_headers=b_headers))
    assert a.model is not b.model
    assert b.model.url == "https://tenant-b.example"
    assert "KEY_A" not in str(b.model.headers) and b.model.headers["X-T"] == "b"
    assert b_headers == {"X-T": "b"}  # the request's own dict is not mutated with the Authorization header
    assert "KEY_B" not in b.model_name  # the cache key never carries the secret in clear


@pytest.mark.asyncio
async def test_shared_tei_client_does_not_carry_cookies_between_requests(monkeypatch):
    import httpx
    from tilellm.models.llm import TEIConfig
    from tilellm.tools.reranker import TEIReranker

    seen = []
    real = httpx.AsyncClient

    def handler(request):
        seen.append(request.headers.get("cookie"))
        texts = json.loads(request.content)["texts"]
        return httpx.Response(200, json=[{"index": i, "score": 1.0} for i in range(len(texts))],
                              headers={"set-cookie": "session=tenant-a; Path=/"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handler)))
    rr = TEIReranker(TEIConfig(provider="tei", name="r", url="http://tei.example"), 200, 40)
    await rr.arerank_documents("q", DOCS[:2], top_k=1)
    await rr.arerank_documents("q", DOCS[:2], top_k=1)
    assert seen == [None, None]


# --- retrieval query and history (RAG review) -------------------------------------------------------------

@pytest.mark.asyncio
async def test_conversation_summary_keeps_the_full_history_and_skips_the_discarded_summary_call(llm):
    # On the default path the summary was generated and then dropped (`pass`): its LLM call was pure
    # latency, and trimming the history there would lose the old turns entirely.
    res, body = await _ask(_qa(chat_history_dict=HISTORY, conversation_summary=True, max_history_messages=10))
    assert res.status_code == 200, body
    assert len(CALLS) == 2
    assert len(CALLS[-1]) == 1 + 30 * 2 + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", ["similarity", "hybrid"])
async def test_hyde_passage_is_used_as_is_for_retrieval(llm, search_type):
    hyde = "Il servizio riprende a settembre con nuove linee verso le frazioni."
    res, body = await _ask(_qa(search_type=search_type, chat_history_dict=HISTORY, retrieval_query=hyde))
    assert res.status_code == 200, body
    assert len(CALLS) == 1  # no rewrite of a passage as if it were a question
    if search_type == "similarity":
        assert QUERIES == [hyde]


@pytest.mark.asyncio
async def test_citations_retrieve_with_the_rewritten_question(llm):
    res, body = await _ask(_qa(chat_history_dict=HISTORY, citations=True, stream=True))
    assert res.status_code == 200
    assert QUERIES == ["Riparte a settembre."]  # the fake LLM's rewrite, not the raw question + citation tail


@pytest.mark.asyncio
async def test_empty_rewrite_falls_back_to_the_question(llm, monkeypatch):
    from tilellm.controller import controller

    async def empty_rewrite(llm, qa):
        return ""

    monkeypatch.setattr(controller, "create_contextualize_query", empty_rewrite)
    res, body = await _ask(_qa(chat_history_dict=HISTORY))
    assert res.status_code == 200, body
    assert QUERIES == ["Quando riparte il trasporto?"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_hyde_also_drives_retrieval_without_contextualization(llm, stream):
    hyde = "Il servizio riprende a settembre."
    res, body = await _ask(_qa(contextualize_prompt=False, retrieval_query=hyde, stream=stream))
    assert res.status_code == 200, body
    assert QUERIES == [hyde]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_reranker_load_error_is_an_error_response_not_a_broken_stream(llm, stream, monkeypatch):
    from tilellm.tools.reranker import TileReranker

    def _boom(cls, *a, **k):
        raise ImportError("CrossEncoder is not available")

    monkeypatch.setattr(TileReranker, "_get_cached_model", classmethod(_boom))
    res = await __import__("tilellm.controller.controller", fromlist=["x"]).ask_with_memory(
        _qa(reranking=True, stream=stream))
    assert res.status_code >= 400 and "StreamingResponse" not in type(res).__name__


# --- Qdrant: the sync client must not run on the event loop -------------------------------------------------

class _SlowQdrant:
    """Sync client whose query takes 0.3 s, like a remote Qdrant under load."""
    def query_points(self, **kw):
        time.sleep(0.3)
        from types import SimpleNamespace
        return SimpleNamespace(points=[SimpleNamespace(id="p1", payload={"page_content": "ciao",
                                                                            "metadata": {"source": "s"}})])


async def _max_loop_lag(coro):
    lag = [0.0]
    done = asyncio.Event()

    async def ticker():
        while not done.is_set():
            t = time.perf_counter()
            await asyncio.sleep(0.01)
            lag[0] = max(lag[0], time.perf_counter() - t - 0.01)

    tick = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        await coro
    finally:
        done.set()
        await tick
    return lag[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["perform_hybrid_search", "search_community_report"])
async def test_qdrant_hybrid_queries_do_not_block_the_loop(method):
    from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository
    qa = QuestionAnswer(question="q", namespace="ns", top_k=2,
                        engine={"name": "qdrant", "deployment": "local", "host": "h", "port": 6333, "index_name": "c"})
    lag = await _max_loop_lag(getattr(QdrantRepository(), method)(qa, _SlowQdrant(), [0.1] * 8,
                                                                  {"indices": [1], "values": [0.5]}))
    assert lag < 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", ["similarity", "hybrid"])
async def test_qdrant_get_chunks_does_not_block_the_loop(monkeypatch, search_type):
    from types import SimpleNamespace
    from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository

    async def _create_index(self, *a, **k):
        return SimpleNamespace(client=_SlowQdrant())

    class _Sparse:
        def __init__(self, *a, **k):
            pass

        async def aencode_queries(self, q):
            return {"indices": [1], "values": [0.5]}

    monkeypatch.setattr(QdrantRepository, "create_index", _create_index)
    monkeypatch.setattr("tilellm.store.qdrant.qdrant_repository_local.TiledeskSparseEncoders", _Sparse)
    qa = QuestionAnswer(question="q", namespace="ns", top_k=2, search_type=search_type,
                        engine={"name": "qdrant", "deployment": "local", "host": "h", "port": 6333, "index_name": "c"})
    repo = QdrantRepository()
    repo.sparse_enabled = True
    fn = QdrantRepository.get_chunks_from_repo.__wrapped__
    lag = await _max_loop_lag(fn(repo, qa, embedding_obj=DeterministicFakeEmbedding(size=8), embedding_dimension=8))
    assert lag < 0.1


@pytest.mark.asyncio
async def test_tei_embeddings_reuse_one_http_client(monkeypatch):
    # /api/qa builds TEIEmbeddings without a shared client: every query embedding opened a new client
    # (SSL context built on the event loop + a new connection to TEI).
    import httpx
    from tilellm.shared.embeddings.embedding_client_manager import TEIEmbeddings

    created = []
    real = httpx.AsyncClient

    def handler(request):
        inputs = json.loads(request.content)["inputs"]
        n = 1 if isinstance(inputs, str) else len(inputs)
        return httpx.Response(200, json=[[0.1] * 4] * n)

    def factory(*a, **k):
        created.append(1)
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    emb = TEIEmbeddings(base_url="http://tei.example", model="m")
    for _ in range(3):
        assert len(await emb.aembed_query("q")) == 4
    await emb.aembed_documents(["a", "b"])
    assert len(created) == 1


# --- second review ------------------------------------------------------------------------------------------

def test_remote_reranker_configs_do_not_evict_the_local_model(monkeypatch):
    import tilellm.tools.reranker as rr
    from pydantic import SecretStr
    from tilellm.models.llm import TEIConfig

    loads = []

    class CE:
        def __init__(self, name, device=None):
            loads.append(name)

    monkeypatch.setattr(rr, "CrossEncoder", CE)
    monkeypatch.setattr(rr.TileReranker, "_warmup_on_load", False)
    rr.TileReranker.clear_cache()
    _ = rr.TileReranker("local-ce").model
    for tenant in "abc":
        _ = rr.TileReranker(TEIConfig(provider="tei", name="bge", url=f"https://{tenant}.example",
                                      api_key=SecretStr(f"k-{tenant}"))).model
    _ = rr.TileReranker("local-ce").model
    assert loads == ["local-ce"]


@pytest.mark.asyncio
async def test_hyde_passage_is_written_for_the_contextualized_question(llm, monkeypatch):
    from tilellm.agents import nodes

    prompts = []

    class Structured:
        async def ainvoke(self, prompt, *a, **k):
            prompts.append(prompt)
            return nodes.HyDEDocument(hypothetical_document="passaggio")

    monkeypatch.setattr(FakeLLM, "with_structured_output", lambda self, schema, **k: Structured(), raising=False)
    qa = _qa(chat_history_dict=HISTORY, use_hyde=True)
    out = await nodes.hyde_node({"question_answer": qa})
    assert out["question_answer"].retrieval_query == "passaggio"
    assert "Riparte a settembre." in prompts[0]  # the fake LLM's rewrite of the follow-up question


@pytest.mark.asyncio
async def test_guard_error_keeps_its_own_type(llm, monkeypatch):
    from tilellm.agents import nodes

    async def guard(state):
        raise PermissionError("401 invalid api key")

    async def intent(state):
        await asyncio.sleep(5)

    monkeypatch.setattr(nodes, "input_guard_node", guard)
    monkeypatch.setattr(nodes, "intent_router_node", intent)
    t = time.perf_counter()
    with pytest.raises(PermissionError):
        await nodes.guard_and_intent_node({"question_answer": _qa(), "metadata": {"trace": []}})
    assert time.perf_counter() - t < 1  # the other task is cancelled, not awaited


def test_loop_bound_keeps_one_instance_per_live_loop():
    # Sync entry points run a coroutine on a temporary loop in another thread: that must not
    # replace the main loop's client (it would then be rebuilt on every call).
    import threading
    from tilellm.shared.loop_bound import LoopBound

    made = []
    lb = LoopBound(lambda: made.append(1) or object())

    async def get():
        return lb.get()

    async def main():
        first = lb.get()
        t = threading.Thread(target=lambda: asyncio.run(get()))
        t.start()
        t.join()
        assert lb.get() is first

    asyncio.run(main())
    assert len(made) == 2


@pytest.mark.asyncio
async def test_qdrant_vector_store_does_not_embed_on_every_request(monkeypatch):
    # QdrantVectorStore validates the collection by embedding a dummy text with a SYNC call from the
    # event loop: a blocking TEI round trip on every retrieval (it also deadlocked a worker on master).
    from types import SimpleNamespace
    from tilellm.store.qdrant.qdrant_repository_local import CachedVectorStore

    class Emb(DeterministicFakeEmbedding):
        def embed_documents(self, texts):
            raise AssertionError("sync embedding while building the vector store")

    class Client:
        def get_collection(self, *a, **k):
            raise AssertionError("collection validation on every request")

    cvs = CachedVectorStore.__new__(CachedVectorStore)
    cvs.engine = SimpleNamespace(index_name="c")
    cvs.embeddings = Emb(size=8)
    cvs._qdrant_client = Client()

    async def _noop():
        return None

    monkeypatch.setattr(cvs, "_ensure_client", _noop)
    store = await cvs.get_vector_store()
    assert store.collection_name == "c"
