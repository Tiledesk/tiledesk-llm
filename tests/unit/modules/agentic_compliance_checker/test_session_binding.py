"""
The session_id used to travel inside the LLM context: the agent received it in
the prompt and passed it as an argument to every tool, so it could leak (answers,
logs, prompt injection) and be replayed from another project — reading another
tender's evidence and spending the gptkey stored in that session.

Now the session is bound server side: /api/ask and /api/thinking take it in a
dedicated body field (QuestionToLLM.compliance_session_id) and build the tools
already bound to it; MCP takes it from the X-Compliance-Session header. No tool
schema exposes session_id, and one passed by the model anyway is ignored.
"""
import json

import fakeredis
import pytest
from pydantic import SecretStr, ValidationError

from tilellm.models import Engine
from tilellm.modules.agentic_compliance_checker.services.langchain_tools import (
    AGENTIC_COMPLIANCE_TOOLS,
    build_tool,
)
from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceRequestV2,
    DiscretionaryCriterion,
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


def _request() -> BulkComplianceRequestV2:
    return BulkComplianceRequestV2(
        requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\nrequirements: {}\n",
        operators=[OperatorRef(namespace="ns-oe1")], engine=Engine(name="qdrant"),
        llm="openai", gptkey=SecretStr("sk-x"), model="gpt-4o-mini",
    )


def _lot(criterion_id: str) -> TenderLotRequirements:
    return TenderLotRequirements(
        tender=TenderInfo(title="Gara", lot_id="L1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(discretionary=[
            DiscretionaryCriterion(id=criterion_id, text="criterio", mode="variabile", max_points=8),
        ]),
    )


def test_no_langchain_tool_schema_exposes_session_id():
    for name in AGENTIC_COMPLIANCE_TOOLS:
        assert "session_id" not in build_tool(name, "sid").args, name


@pytest.mark.asyncio
async def test_no_mcp_tool_schema_exposes_session_id():
    from tilellm.modules.agentic_compliance_checker.services.mcp_server import mcp

    for tool in await mcp.list_tools():
        assert "session_id" not in tool.inputSchema.get("properties", {}), tool.name


@pytest.mark.asyncio
async def test_bound_tool_ignores_a_session_id_passed_by_the_model(fake_redis):
    mine = await SessionStore.create(_request(), _lot("MIO"))
    other = await SessionStore.create(_request(), _lot("ALTRO"))

    result = await build_tool("compliance_list_requirements", mine).ainvoke({"session_id": other})

    assert "MIO" in result and "ALTRO" not in result


def test_registry_factory_needs_the_session():
    from tilellm.modules.tools_registry.services.tool_registry import resolve_tools

    with pytest.raises(ValueError, match="compliance_session_id"):
        resolve_tools(["compliance_list_requirements"], chat_model=None, base64_storage=None)

    [tool] = resolve_tools(["compliance_list_requirements"], chat_model=None, base64_storage=None,
                           compliance_session_id="sid")
    assert tool.name == "compliance_list_requirements" and "session_id" not in tool.args


def test_question_with_compliance_tools_requires_the_session():
    from tilellm.models.llm import QuestionToLLM

    base = dict(question="q", llm_key="sk", llm="openai", model="gpt-4o-mini")
    with pytest.raises(ValidationError, match="compliance_session_id"):
        QuestionToLLM(**base, tools=["compliance_list_requirements"])

    q = QuestionToLLM(**base, tools=["compliance_list_requirements"], compliance_session_id="sid-secret")
    assert "sid-secret" not in repr(q)
    assert QuestionToLLM(**base, tools=["web_search"]).compliance_session_id is None


def test_ask_and_thinking_pass_the_session_to_the_tool_factories():
    import inspect
    import re

    from tilellm.controller import controller

    src = inspect.getsource(controller)
    calls = re.findall(r"resolve_tools\((.*?)\)\n", src, re.S)
    assert len(calls) == 2 and all("compliance_session_id" in c for c in calls)


def test_every_registry_factory_accepts_the_shared_runtime_kwargs():
    """resolve_tools passes the same runtime kwargs to every factory. The
    multimodal one didn't accept them (chat_model vs llm_instance): TypeError,
    printed and swallowed — selecting it via `tools` silently did nothing."""
    from unittest.mock import MagicMock

    from tilellm.modules.tools_registry.services.tool_registry import TOOL_REGISTRY, resolve_tools

    import tilellm.modules.agentic_compliance_checker.controllers  # noqa: F401 — registers its tools

    factories = [n for n, c in TOOL_REGISTRY.items() if c["is_factory"]]
    tools = resolve_tools(factories, chat_model=MagicMock(), base64_storage={}, compliance_session_id="sid")
    assert sorted(t.name for t in tools) == sorted(
        TOOL_REGISTRY[n]["implementation"](chat_model=MagicMock(), base64_storage={},
                                            compliance_session_id="sid").name
        for n in factories
    )
    assert len(tools) == len(factories)
