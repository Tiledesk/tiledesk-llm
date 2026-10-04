"""
TEIEmbeddings sync methods inside a running event loop.

Found on a real run: langchain_qdrant's QdrantVectorStore validates the collection
with a *sync* embed_documents call from inside the event loop. The sync bridge ran
the coroutine on a second loop in a worker thread, but with the shared
httpx.AsyncClient, whose connections belong to the main loop — which was blocked
waiting for that very thread. The gunicorn worker froze with no error and no timeout.
"""
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tilellm.shared.embeddings import embedding_client_manager as ecm


def _ephemeral_client_factory(calls):
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=[[0.1, 0.2]])

    real = httpx.AsyncClient
    return lambda *a, **kw: real(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_sync_embed_inside_running_loop_never_touches_the_shared_client():
    shared = httpx.AsyncClient()
    shared.post = AsyncMock(side_effect=AssertionError("shared loop-bound client used"))
    tei = ecm.TEIEmbeddings(base_url="http://tei", model="m", client=shared)
    calls = []

    with patch.object(ecm.httpx, "AsyncClient", _ephemeral_client_factory(calls)):
        vectors = tei.embed_documents(["x"])  # called with the loop running, like langchain_qdrant does
        query = tei.embed_query("y")

    assert vectors == [[0.1, 0.2]]
    assert query == [0.1, 0.2]
    assert len(calls) == 2
    shared.post.assert_not_awaited()
    await shared.aclose()


@pytest.mark.asyncio
async def test_async_embed_keeps_using_the_shared_client():
    shared = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json=[[0.3]])))
    tei = ecm.TEIEmbeddings(base_url="http://tei", model="m", client=shared)

    assert await tei.aembed_query("z") == [0.3]
    await shared.aclose()
