"""
MCP server configs carry credentials (headers, tokens in the URL, fixed tool
arguments). The request is logged (`logger.info(question)`) and the MCP loader
logs each server: none of those values may reach the logs. Key names and the
host are fine (they are what makes a log useful).
"""
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import httpx2
import pytest

import tilellm.__main__ as main_mod
from tilellm.controller.controller import handle_agent_exception
from tilellm.models.base import _redact_url
from tilellm.models.llm import QuestionToLLM
from tilellm.shared import mcp_client

AUTH_VALUE = "Bearer FAKE-XYZ"
PATH_TOKEN = "PATHTOKEN-123"
QUERY_TOKEN = "QUERYTOKEN-456"
USERINFO = "utente-finto:PWD-FINTA"
TOOL_ARG_VALUE = "VALORE-FITTIZIO-789"
LLM_KEY = "sk-FAKE-LLMKEY-000"
AWS_SECRET = "AWS-SEGRETO-FINTO-111"
URL = f"https://{USERINFO}@mcp-esempio.example:8443/t/{PATH_TOKEN}/mcp?token={QUERY_TOKEN}"
SECRETS = [AUTH_VALUE, "FAKE-XYZ", PATH_TOKEN, QUERY_TOKEN, "PWD-FINTA", TOOL_ARG_VALUE, LLM_KEY]


@pytest.fixture
def app_log(caplog):
    # the app's logging config stops "tilellm.*" from propagating to the root handler
    log = logging.getLogger("tilellm")
    log.addHandler(caplog.handler)
    level = log.level
    log.setLevel(logging.DEBUG)
    yield caplog
    log.removeHandler(caplog.handler)
    log.setLevel(level)


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
async def test_api_ask_request_log_has_no_secrets(app_log):
    with patch.object(main_mod, "ask_mcp_agent", AsyncMock(return_value="ok")):
        await main_mod.post_ask_to_llm_main(_question())

    _assert_no_secret(app_log.text)
    # still useful: transport, host and key names are there
    assert "mcp-esempio.example:8443" in app_log.text
    assert "Authorization" in app_log.text and "tool_arguments" in app_log.text


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
async def test_mcp_loader_logs_have_no_url_secrets(app_log, monkeypatch):
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "mcp-esempio.example")
    question = _question()
    with patch.object(mcp_client, "_load_server_tools",
                      AsyncMock(side_effect=httpx.ConnectError("connection refused"))):
        assert await mcp_client.load_tools(question.servers) == []

    _assert_no_secret(app_log.text)
    assert "https://mcp-esempio.example:8443" in app_log.text


@pytest.mark.asyncio
async def test_refused_url_log_has_no_url_secrets(app_log, monkeypatch):
    monkeypatch.delenv("OUTBOUND_URL_ALLOWED_HOSTS", raising=False)
    question = _question()
    question.servers["negozio"].url = f"http://169.254.169.254/t/{PATH_TOKEN}/mcp?token={QUERY_TOKEN}"

    assert await mcp_client.load_tools(question.servers) == []

    assert "rifiutato" in app_log.text
    _assert_no_secret(app_log.text)


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


# --- HTTP error texts carry the full URL (query token included) ---------------------

def _status_error(lib):
    url = f"https://mcp-esempio.example/mcp?token={QUERY_TOKEN}"
    request = lib.Request("POST", url)
    response = lib.Response(403, request=request)
    return lib.HTTPStatusError(f"Client error '403 Forbidden' for url '{url}'", request=request, response=response)


@pytest.mark.asyncio
@pytest.mark.parametrize("lib", [httpx, httpx2], ids=["httpx", "httpx2"])
async def test_mcp_loader_error_log_has_no_url_token(app_log, monkeypatch, lib):
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "mcp-esempio.example")
    group = BaseExceptionGroup("unhandled errors in a TaskGroup", [_status_error(lib)])
    with patch.object(mcp_client, "_load_server_tools", AsyncMock(side_effect=group)):
        assert await mcp_client.load_tools(_question().servers) == []

    assert "403" in app_log.text  # still useful
    _assert_no_secret(app_log.text)


@pytest.mark.parametrize("lib", [httpx, httpx2], ids=["httpx", "httpx2"])
@pytest.mark.parametrize("grouped", [True, False], ids=["group", "plain"])
def test_agent_exception_log_has_no_url_token(app_log, lib, grouped, capsys):
    exc = _status_error(lib)
    if grouped:
        exc = BaseExceptionGroup("unhandled errors in a TaskGroup", [exc])

    handle_agent_exception(exc, "test")

    assert "403" in app_log.text
    _assert_no_secret(app_log.text)
    _assert_no_secret(capsys.readouterr().err)  # no raw traceback on stderr either


# --- library loggers: httpx/httpx2 log every request URL at INFO -------------------

@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_http_library_loggers_stay_at_warning(monkeypatch, tmp_path, level):
    monkeypatch.chdir(Path(__file__).resolve().parents[3])  # log_conf.json is read from the cwd
    monkeypatch.setenv("LOG_FILE_PATH", str(tmp_path / "app.log"))
    for name in ("LOG_LEVEL_STDOUT", "LOG_LEVEL_FILE", "LOG_LEVEL_SYS"):
        monkeypatch.setenv(name, level)
    try:
        main_mod.setup_logging()
        for name in ("httpx", "httpx2", "httpcore", "httpcore2", "mcp"):
            assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING, name
    finally:
        for name in ("LOG_FILE_PATH", "LOG_LEVEL_STDOUT", "LOG_LEVEL_FILE", "LOG_LEVEL_SYS"):
            monkeypatch.delenv(name)
        main_mod.setup_logging()  # back to the defaults


def test_llm_model_url_and_key_are_not_in_the_request_repr():
    question = QuestionToLLM(question="ciao", llm="vllm", llm_key=LLM_KEY, model={
        "provider": "vllm", "name": "modello-finto", "api_key": AUTH_VALUE,
        "url": URL, "custom_headers": {"X-Token": TOOL_ARG_VALUE},
    })
    for text in (str(question), repr(question)):
        _assert_no_secret(text)
        assert "https://mcp-esempio.example:8443" in text and "modello-finto" in text and "X-Token" in text
