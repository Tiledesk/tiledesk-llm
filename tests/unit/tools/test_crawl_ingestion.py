"""
type=crawl through the ingestion path: the request fields reach the crawler,
every backend routes `crawl` to the fetch branch, and each chunk cites the
page it came from (not the start URL).
"""
import pathlib
import re
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from tilellm.models import Engine
from tilellm.models.document_type import FETCHED_TYPES, DocumentType
from tilellm.models.llm import ItemSingle
from tilellm.tools import document_tools, web_crawler

START = "https://www.example.org/"


def _item(**over):
    kw = dict(id="site-1", source=START, type="crawl", namespace="ns",
              engine=Engine(name="qdrant", type="serverless", apikey="", vector_size=1024,
                            index_name="c", host="localhost", port=6333))
    kw.update(over)
    return ItemSingle(**kw)


def test_crawl_fields_defaults():
    item = _item()
    assert item.type == DocumentType.CRAWL
    assert (item.max_depth, item.max_pages, item.ignore_robots) == (3, 50, False)
    assert item.crawl_options() == {"max_depth": 3, "max_pages": 50, "ignore_robots": False,
                                    "file_types": ["pdf", "docx"]}


@pytest.mark.parametrize("field,value", [("max_depth", -1), ("max_depth", 11),
                                         ("max_pages", 0), ("max_pages", 501)])
def test_crawl_limits_are_bounded(field, value):
    with pytest.raises(ValidationError):
        _item(**{field: value})


@pytest.mark.asyncio
async def test_fetch_documents_passes_the_options_to_the_crawler(monkeypatch):
    calls = []

    async def fake_crawl(url, **kwargs):
        calls.append((url, kwargs))
        return [Document(page_content="testo della pagina", metadata={"source": url + "a"})]

    monkeypatch.setattr(web_crawler, "crawl_site", fake_crawl)
    docs = await document_tools.fetch_documents(
        "crawl", START, 0, None, {"user-agent": "x"},
        crawl_options={"max_depth": 1, "max_pages": 7, "ignore_robots": True})
    assert docs[0].metadata["source"] == START + "a"
    assert calls == [(START, {"browser_headers": {"user-agent": "x"}, "max_depth": 1,
                              "max_pages": 7, "ignore_robots": True})]


def test_crawl_is_a_fetched_type():
    assert DocumentType.CRAWL in FETCHED_TYPES and DocumentType.URL in FETCHED_TYPES


def test_no_backend_keeps_its_own_list_of_fetched_types():
    # a hard-coded ['url', 'pdf', ...] silently sends `crawl` to the direct-text branch
    root = pathlib.Path(document_tools.__file__).parents[1] / "store"
    offenders = [str(p) for p in root.rglob("*.py")
                 if re.search(r"in \[\s*'url',\s*'pdf'", p.read_text())]
    assert offenders == []


def _repos():
    from tilellm.store.milvus.milvus_repository import MilvusRepository
    from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
    from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository
    from tilellm.store.redis.redis_vector_repository import RedisRepository
    return [MilvusRepository, PineconeRepositoryServerless, QdrantRepository, RedisRepository]


def _chunk_item(item_type):
    from tilellm.models.llm import TableOptions
    item = MagicMock()
    item.id, item.source, item.type = "site-1", START, item_type
    item.embedding, item.tags, item.namespace = "text-embedding-3-small", None, "ns"
    item.chunk_size, item.chunk_overlap = 1000, 200
    item.semantic_chunk, item.breakpoint_threshold_type = False, "percentile"
    item.table_options = TableOptions()
    return item


@pytest.mark.asyncio
@pytest.mark.parametrize("repo_cls", _repos())
async def test_crawled_chunks_cite_their_page(repo_cls):
    repo = repo_cls.__new__(repo_cls)
    page = START + "servizi/anagrafe"
    doc = Document(page_content="Orari dello sportello anagrafe. " * 5,
                   metadata={"source": page, "file_name": "Anagrafe"})
    chunks = await repo.chunk_documents(_chunk_item("crawl"), [doc], embeddings=None)
    assert {c.metadata["source"] for c in chunks} == {page}
    assert {c.metadata["file_name"] for c in chunks} == {"Anagrafe"}


@pytest.mark.asyncio
@pytest.mark.parametrize("repo_cls", _repos())
async def test_other_types_still_cite_the_request_source(repo_cls):
    repo = repo_cls.__new__(repo_cls)
    doc = Document(page_content="Contenuto del file. " * 5, metadata={"source": "/tmp/tmpab12.pdf"})
    chunks = await repo.chunk_documents(_chunk_item("pdf"), [doc], embeddings=None)
    assert {c.metadata["source"] for c in chunks} == {START}


def test_every_backend_fetch_passes_the_crawl_options():
    # without them a backend crawls silently with the defaults (3 levels, 50 pages)
    root = pathlib.Path(document_tools.__file__).parents[1] / "store"
    counts = {p.name: (p.read_text().count("in FETCHED_TYPES"),
                       p.read_text().count("crawl_options=item.crawl_options()"))
              for p in root.rglob("*.py") if "in FETCHED_TYPES" in p.read_text()}
    assert len(counts) == 5, counts  # qdrant, milvus, pinecone serverless and pod, base (redis)
    assert all(branches == passed for branches, passed in counts.values()), counts


def test_pod_keeps_the_page_source():
    from tilellm.store.pinecone import pinecone_repository_pod
    assert "chunk_source(type_source, source" in pathlib.Path(pinecone_repository_pod.__file__).read_text()


def test_item_single_default_user_agent_is_the_shared_constant():
    # the crawler recognises ItemSingle's default user agent by this constant
    from tilellm.models.llm import DEFAULT_BROWSER_USER_AGENT
    assert _item().browser_headers["user-agent"] == DEFAULT_BROWSER_USER_AGENT
    assert _item(browser_headers={"Authorization": "x"}).browser_headers["user-agent"] == DEFAULT_BROWSER_USER_AGENT
