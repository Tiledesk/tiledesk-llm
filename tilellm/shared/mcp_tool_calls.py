"""MCP tool calls: fixed per-server arguments and errors handed back to the model.

Two langchain-mcp-adapters tool interceptors plus the httpx client factory that
keeps the body of an HTTP error response (mcp streams it and raises before
reading it, so the server's JSON-RPC error message would otherwise be lost).
"""
import asyncio
import contextlib
import json
import logging
from typing import Any, Dict, Optional, Tuple

import httpx
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)

_ERROR_BODY_LIMIT = 8192
_ERROR_MESSAGE_LIMIT = 500


def root_exception(exc: BaseException) -> BaseException:
    """Innermost cause, unwrapping ExceptionGroup (first sub-exception) and __cause__ chains."""
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        return root_exception(exc.exceptions[0])
    if exc.__cause__ and exc.__cause__ is not exc:
        return root_exception(exc.__cause__)
    return exc


def describe_exception(exc: BaseException) -> str:
    """Short, secret-free description of the root cause (no headers, no URL)."""
    root = root_exception(exc)
    if isinstance(root, httpx.HTTPStatusError):
        response = root.response
        text = f"HTTP {response.status_code}"
        message = _jsonrpc_error_message(response)
        return f"{text} — {message}" if message else text
    return f"{type(root).__name__}: {root}"[:_ERROR_MESSAGE_LIMIT]


def _jsonrpc_error_message(response: httpx.Response) -> Optional[str]:
    body = getattr(response, "tilellm_error_body", None)
    if body is None:
        try:
            body = response.content
        except httpx.ResponseNotRead:
            return None
    try:
        message = json.loads(body)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return None
    return str(message)[:_ERROR_MESSAGE_LIMIT] if message else None


async def _keep_error_body(response: httpx.Response) -> None:
    """Response hook: read (bounded) the body of an error response before mcp raises."""
    if response.status_code < 400:
        return
    # ponytail: the body is consumed here; an httpx.Auth flow that re-reads the
    # error response (e.g. mcp's OAuthClientProvider on 401) would find it read.
    # No such auth is configured from a request today; if added, skip the hook
    # for the status codes that auth flow handles.
    body = b""
    async with contextlib.aclosing(response.aiter_bytes()) as chunks:
        async for chunk in chunks:
            body += chunk
            if len(body) >= _ERROR_BODY_LIMIT:
                break
    response.tilellm_error_body = body[:_ERROR_BODY_LIMIT]


def mcp_httpx_client_factory(
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[httpx.Timeout] = None,
    auth: Optional[httpx.Auth] = None,
) -> httpx.AsyncClient:
    """mcp's default client (same headers/timeout/auth/redirects) + the error-body hook."""
    client = create_mcp_http_client(headers=headers, timeout=timeout, auth=auth)
    client.event_hooks = {"request": [], "response": [_keep_error_body]}
    return client


class FixedToolArguments:
    """Interceptor: sets configured arguments on the calls of the tools that declare them.

    A configured top-level key replaces whatever the model sent under it.

    `per_tool[(server, tool)]` is filled by get_all_filtered_tools, which knows the
    tool schemas; a tool not in the map is called unchanged.
    """

    def __init__(self) -> None:
        self.per_tool: Dict[Tuple[str, str], Dict[str, Any]] = {}

    async def __call__(self, request, handler):
        fixed = self.per_tool.get((request.server_name, request.name))
        if fixed:
            request = request.override(args={**(request.args or {}), **fixed})
        return await handler(request)


async def tool_errors_to_result(request, handler):
    """Interceptor: a failing tool call becomes an isError result the model can read."""
    try:
        return await handler(request)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # includes BaseExceptionGroup from anyio
        if isinstance(root_exception(exc), (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
        message = f"MCP tool {request.name} failed: {describe_exception(exc)}"
        logger.warning("server=%r %s", request.server_name, message)
        return CallToolResult(isError=True, content=[TextContent(type="text", text=message)])
