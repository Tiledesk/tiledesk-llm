"""
MCP server configs carry credentials (headers, tokens in the URL, fixed tool
arguments). The request is logged (`logger.info(question)`) and the MCP code
logs each server before and after get_tools: none of those values may reach
the logs. Names of keys and the host are fine (they are what makes a log useful).
"""
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import tilellm.__main__ as main_mod
from tilellm.controller.controller_utils import get_all_filtered_tools
from tilellm.models.base import _redact_url
from tilellm.models.llm import QuestionToLLM

AUTH_VALUE = "Bearer FAKE-XYZ"
PATH_TOKEN = "PATHTOKEN-123"
QUERY_TOKEN = "QUERYTOKEN-456"
USERINFO = "utente-finto:PWD-FINTA"
TOOL_ARG_VALUE = "VALORE-FITTIZIO-789"
LLM_KEY = "sk-FAKE-LLMKEY-000"
AWS_SECRET = "AWS-SEGRETO-FINTO-111"
URL = f"https://{USERINFO}@mcp-esempio.example:8443/t/{PATH_TOKEN}/mcp?token={QUERY_TOKEN}"
SECRETS = [AUTH_VALUE, "FAKE-XYZ", PATH_TOKEN, QUERY_TOKEN, "PWD-FINTA", TOOL_ARG_VALUE, LLM_KEY]


def _question(**kwargs):
    return QuestionToLLM(
        question="che ore sono?",
        llm="openai",
        llm_key=kwargs.pop("llm_key", LLM_KEY),
        servers={"negozio": {
            "transport": "streamable_http",
            "url": URL,
            "headers": {"Authorization": AUTH_VALUE},
            "parameters": {"tool_arguments": {"meta": {"profile": TOOL_ARG_VALUE}}},
        }},
        **kwargs,
    )


def _assert_no_secret(text):
    for secret in SECRETS:
        assert secret not in text, secret


@pytest.mark.asyncio
async def test_api_ask_request_log_has_no_secrets(caplog):
    caplog.set_level(logging.DEBUG)
    with patch.object(main_mod, "ask_mcp_agent", AsyncMock(return_value="ok")):
        await main_mod.post_ask_to_llm_main(_question())

    _assert_no_secret(caplog.text)
    # still useful: transport, host and key names are there
    assert "mcp-esempio.example:8443" in caplog.text
    assert "Authorization" in caplog.text and "tool_arguments" in caplog.text


def test_str_and_repr_of_the_request_have_no_secrets():
    question = _question()
    for text in (str(question), repr(question)):
        _assert_no_secret(text)
        assert "streamable_http" in text


def test_aws_secret_is_not_in_the_request_repr():
    question = _question(llm_key={
        "aws_access_key_id": "AKIAFINTO", "aws_secret_access_key": AWS_SECRET, "region_name": "eu-south-1",
    })
    assert AWS_SECRET not in str(question) and AWS_SECRET not in repr(question)
    assert question.llm_key.aws_secret_access_key == AWS_SECRET  # value still usable


@pytest.mark.asyncio
async def test_mcp_client_and_get_tools_logs_have_no_url_secrets(caplog):
    caplog.set_level(logging.DEBUG)
    question = _question()
    client = question.create_mcp_client()
    client.get_tools = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    await get_all_filtered_tools(client, question.servers)

    _assert_no_secret(caplog.text)
    assert "https://mcp-esempio.example:8443" in caplog.text


@pytest.mark.parametrize("url,expected", [
    (URL, "https://mcp-esempio.example:8443"),
    ("http://127.0.0.1/mcp?k=v", "http://127.0.0.1"),
    ("https://[::1]:9000/x", "https://[::1]:9000"),
    (None, None),
    ("", ""),
    ("non-un-url", "<url non valido>"),
    ("https://host.example:notaport/x", "<url non valido>"),
])
def test_redact_url_keeps_only_scheme_host_port(url, expected):
    assert _redact_url(url) == expected
