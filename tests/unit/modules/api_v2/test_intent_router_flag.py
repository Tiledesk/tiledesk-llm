"""
/api/v2/qa sent ordinary FAQ questions ("how often must the password be
changed?") to the compliance branch: the intent router is an LLM call that
misclassifies them. use_intent_router=False goes straight to RAG; the default
(True) keeps today's behavior.
"""
from types import SimpleNamespace

from tilellm.agents import workflow
from tilellm.modules.api_v2.models import QASimpleRequest
from tilellm.modules.api_v2.node_registry import NODE_REGISTRY


def _state(**flags):
    return {"question_answer": SimpleNamespace(**flags)}


def test_entry_skips_guard_and_router_when_both_are_off():
    assert workflow._simple_entry_router(_state(use_guard=False, use_intent_router=False)) == "rag"
    assert workflow._simple_entry_router(_state(use_guard=False, use_intent_router=True)) == "intent"
    assert workflow._simple_entry_router(_state(use_guard=True, use_intent_router=False)) == "guard"


def test_after_guard_goes_to_rag_when_router_is_off():
    on_topic = {"is_on_topic": "si"}
    assert workflow._simple_after_guard_router({**_state(use_intent_router=False), **on_topic}) == "rag"
    assert workflow._simple_after_guard_router({**_state(use_intent_router=True), **on_topic}) == "intent"
    assert workflow._simple_after_guard_router({**_state(use_intent_router=False), "is_on_topic": "no"}) == "rifiuta"


def test_compiled_graph_can_reach_rag_without_the_router():
    graph = workflow.simple_workflow.compile().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("_start", "rag_core") in edges and ("guardia", "rag_core") in edges


def test_flag_defaults_to_today_behavior_and_is_documented():
    assert QASimpleRequest.model_fields["use_intent_router"].default is True
    entry = next(n for n in NODE_REGISTRY if n["name"] == "intent_router")
    assert entry["can_disable"] is True and entry["request_field"] == "use_intent_router"
