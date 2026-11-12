"""Every agent tool call is logged: name, argument names, status, time and output size at
INFO; argument values and an excerpt of the output only at DEBUG (they may hold user data)."""
import logging

import pytest
from langchain_core.messages import ToolMessage

from tilellm.controller.agent_middleware import MessageCleaningMiddleware, TOOL_LOG_MAX_CHARS


class _Req:
    tool_call = {"name": "lookup_order", "id": "call-1", "args": {"email": "cliente.finto@example.org"}}


async def _run(caplog, level, output="x" * (TOOL_LOG_MAX_CHARS + 50)):
    log = logging.getLogger("tilellm.controller.agent_middleware")
    log.addHandler(caplog.handler)
    caplog.set_level(level, logger="tilellm.controller.agent_middleware")
    result = ToolMessage(content=output, tool_call_id="call-1")

    async def handler(request):
        return result

    try:
        out = await MessageCleaningMiddleware({}).awrap_tool_call(_Req(), handler)
    finally:
        log.removeHandler(caplog.handler)
    return out, result, " ".join(r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_info_has_names_not_values(caplog):
    out, result, text = await _run(caplog, logging.INFO)
    assert out is result
    assert "lookup_order" in text and "email" in text and f"output_len={TOOL_LOG_MAX_CHARS + 50}" in text
    assert "cliente.finto@example.org" not in text and "xxxx" not in text


@pytest.mark.asyncio
async def test_debug_has_values_and_truncated_output(caplog):
    _, _, text = await _run(caplog, logging.DEBUG)
    assert "cliente.finto@example.org" in text
    assert "x" * TOOL_LOG_MAX_CHARS + "..." in text and "x" * (TOOL_LOG_MAX_CHARS + 1) not in text


@pytest.mark.asyncio
async def test_failure_is_logged_and_raised(caplog):
    async def handler(request):
        raise RuntimeError("server down")

    log = logging.getLogger("tilellm.controller.agent_middleware")
    log.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="tilellm.controller.agent_middleware")
    try:
        with pytest.raises(RuntimeError):
            await MessageCleaningMiddleware({}).awrap_tool_call(_Req(), handler)
    finally:
        log.removeHandler(caplog.handler)
    assert "FAILED" in caplog.text and "RuntimeError" in caplog.text
