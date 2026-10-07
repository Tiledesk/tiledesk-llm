"""
Labels, relationship types and property keys cannot be Cypher parameters, so
they were interpolated into the query text as-is. They come from the API
(GET /api/kg/nodes?label=...) and from LLM extraction over user documents
(entity_type / relationship_type): a value like `X) DETACH DELETE n //` became
part of the query. API values are now rejected; LLM-derived types are
normalized to a valid identifier (they used to fail the query and the entity
was lost).
"""
from unittest.mock import AsyncMock

import pytest

from tilellm.modules.knowledge_graph_falkor.models import Node, Relationship
from tilellm.modules.knowledge_graph_falkor.repository.async_falkor_repository import (
    AsyncFalkorGraphRepository,
)

EVIL = "Entity) DETACH DELETE n //"


@pytest.fixture
def repo():
    r = AsyncFalkorGraphRepository.__new__(AsyncFalkorGraphRepository)
    r._execute_query = AsyncMock(return_value=[])
    return r


def _queries(repo):
    return " ".join(str(c.args[0]) for c in repo._execute_query.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", [
    lambda r: r.find_nodes_by_label(EVIL),
    lambda r: r.find_nodes_by_property(EVIL, "name", "x"),
    lambda r: r.find_nodes_by_property("Entity", "name = 'a' OR true //", "x"),
    lambda r: r.create_node(Node(label=EVIL, properties={"name": "a"})),
    lambda r: r.create_node(Node(label="Entity", properties={"a: 1}) DETACH DELETE n //": 1})),
    lambda r: r.create_relationship(Relationship(source_id="1", target_id="2", type=EVIL)),
    lambda r: r.create_relationship(Relationship(source_id="1", target_id="2", type="REL",
                                                 properties={"x}]->() DETACH DELETE n //": 1})),
])
async def test_api_identifiers_are_rejected_before_any_query(repo, call):
    with pytest.raises(ValueError, match="identifier"):
        await call(repo)
    repo._execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_update_node_rejects_property_keys(repo):
    repo.find_node_by_id = AsyncMock(return_value=Node(id="1", label="Entity", properties={}))
    with pytest.raises(ValueError, match="identifier"):
        await repo.update_node("1", properties={"a = 1 DETACH DELETE n //": 1})
    repo._execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_valid_identifiers_still_work_including_accents(repo):
    await repo.find_nodes_by_label("ATTIVITÀ")
    await repo.find_nodes_by_property("Entity", "entity_type", "x")

    assert "(n:ATTIVITÀ)" in _queries(repo)


@pytest.mark.asyncio
async def test_llm_entity_types_are_normalized_not_injected(repo):
    await repo.batch_create_nodes(
        [{"entity_name": "a", "entity_type": "medical device"},
         {"entity_name": "b", "entity_type": EVIL}],
        namespace="ns",
    )

    q = _queries(repo)
    assert "(n:MEDICAL_DEVICE" in q
    assert "DETACH DELETE" not in q


@pytest.mark.asyncio
async def test_llm_relationship_types_are_normalized_not_injected(repo):
    await repo.batch_create_relationships(
        [{"src_id": "a", "tgt_id": "b", "relationship_type": "works for"},
         {"src_id": "a", "tgt_id": "b", "relationship_type": "R]->() DETACH DELETE n //"}],
        entity_node_map={"a": "1", "b": "2"},
        namespace="ns",
    )

    q = _queries(repo)
    assert "[r:WORKS_FOR" in q
    assert "DETACH DELETE" not in q


# ---------------------------------------------------------------------------
# LLM-generated Cypher (Text2Cypher agent): the "starts with MATCH" check lets
# `MATCH (n) DETACH DELETE n` through. It must run as a read-only query, which
# FalkorDB enforces server side.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_only_execution_uses_ro_query():
    from unittest.mock import MagicMock

    r = AsyncFalkorGraphRepository.__new__(AsyncFalkorGraphRepository)
    graph = MagicMock(name="graph")
    graph.ro_query = AsyncMock(return_value=MagicMock(result_set=[], header=[]))
    graph.query = AsyncMock()
    r._get_graph = lambda **kw: graph

    await r._execute_query("MATCH (n) DETACH DELETE n", {}, namespace="ns", read_only=True)

    graph.ro_query.assert_awaited_once()
    graph.query.assert_not_called()


@pytest.mark.asyncio
async def test_query_graph_tool_runs_llm_cypher_read_only(repo):
    from tilellm.modules.knowledge_graph_falkor.agents.tools import create_cypher_executor_tool

    tool = create_cypher_executor_tool(repo, namespace="ns")
    await tool.ainvoke({"cypher_query": "MATCH (n) DETACH DELETE n"})

    assert repo._execute_query.call_args.kwargs.get("read_only") is True


@pytest.mark.asyncio
async def test_text2cypher_executor_runs_read_only(repo):
    from tilellm.modules.knowledge_graph_falkor.agents import nodes

    executor = nodes.create_nodes(repo, llm=AsyncMock())["executor"]
    await executor({"cypher_query": "MATCH (n) DETACH DELETE n", "namespace": "ns", "metadata": {}})

    assert repo._execute_query.call_args.kwargs.get("read_only") is True


# ponytail: /nodes/search is not exercised — /nodes/{node_id} is declared first
# and shadows it (pre-existing routing bug, the endpoint is unreachable today).
@pytest.mark.parametrize("path", ["/nodes?label=X"])
def test_invalid_identifier_is_a_client_error_not_a_500(path):
    from unittest.mock import patch

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tilellm.modules.knowledge_graph_falkor import controllers

    app = FastAPI()
    app.include_router(controllers.router)
    prefix = controllers.router.prefix
    boom = AsyncMock(side_effect=ValueError("invalid Cypher identifier: 'X'"))
    with patch.object(controllers.kg_logic, "get_nodes_by_label", boom), \
         patch.object(controllers.kg_logic, "search_nodes", boom):
        response = TestClient(app).get(prefix + path)

    assert response.status_code == 400
