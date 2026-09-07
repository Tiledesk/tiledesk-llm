"""
gpt-5.x reasoning models reject temperature/top_p outright (HTTP 400) unless
reasoning.effort == "none", which QuestionAnswer/QuestionToLLM never default to.
The old "force temperature=1.0" behavior (needed on the legacy chat-completions
gpt-5 API) must NOT resurface, or every gpt-5.x call breaks. See llm.py
adjust_temperature_and_validate().
"""
from tilellm.models.llm import QuestionAnswer, QuestionToLLM
from tilellm.models.vector_store import Engine


def test_question_answer_gpt5_6_omits_temperature_and_top_p():
    qa = QuestionAnswer(question="hi", namespace="ns", model="gpt-5.6", engine=Engine())
    assert qa.temperature is None
    assert qa.top_p is None
    assert qa.thinking.reasoning_effort == "low"


def test_question_to_llm_gpt5_6_omits_temperature_and_top_p():
    q = QuestionToLLM(question="hi", model="gpt-5.6", engine=Engine(), llm="openai", llm_key="sk")
    assert q.temperature is None
    assert q.top_p is None
    assert q.thinking.reasoning_effort == "medium"


def test_question_answer_non_gpt5_keeps_default_temperature():
    qa = QuestionAnswer(question="hi", namespace="ns", model="gpt-4o", engine=Engine())
    assert qa.temperature == 0.0


if __name__ == "__main__":
    test_question_answer_gpt5_6_omits_temperature_and_top_p()
    test_question_to_llm_gpt5_6_omits_temperature_and_top_p()
    test_question_answer_non_gpt5_keeps_default_temperature()
    print("ok")
