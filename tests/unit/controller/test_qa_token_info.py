"""/api/qa returns input and output tokens separately (prompt_token_info, as /api/ask),
next to the unchanged total prompt_token_size."""
from unittest.mock import Mock

from langchain_core.documents import Document

from tilellm.controller.controller_utils import format_result
from tilellm.shared.tiledesk_chatmodel_info import TiledeskAICallbackHandler


def _format(handler):
    result = {"context": [Document(page_content="t", metadata={"id": "d1", "source": "s"})],
              "input": "q", "chat_history": [], "answer": "a"}
    qa = Mock(namespace="ns", debug=False, citations=False)
    return format_result(result=result, citations=None, question_answer=qa,
                         callback_handler=handler, question_answer_list=[], success=True)


def test_tokens_are_returned_separately():
    handler = TiledeskAICallbackHandler()
    handler.prompt_tokens, handler.completion_tokens, handler.total_tokens = 1200, 80, 1280
    out = _format(handler).model_dump()
    assert out["prompt_token_info"] == {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280}
    assert out["prompt_token_size"] == 1280  # unchanged for existing callers


def test_no_handler_gives_zero_counts():
    out = _format(None).model_dump()
    assert out["prompt_token_info"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert out["prompt_token_size"] == 0
