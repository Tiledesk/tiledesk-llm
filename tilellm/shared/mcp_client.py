"""
MCP client: loads the tools of the MCP servers named in a request.

Built on langchain.mcp / fastmcp (protocol 2026-07-28, falling back to the initialize
handshake with older servers). The protocol is negotiated per server
(ServerConfig.protocol: "auto" by default, "legacy" or a modern version to pin it),
so one request may mix servers of different eras.

- Transports are always built explicitly from ServerConfig (streamable HTTP or SSE):
  never inferred from a config dict, where "command" would mean stdio.
- One adapter per server: a multi-server group would rename tools "{server}_{tool}",
  changing the names the model sees.
- Elicitation is declined: our APIs can't pause for the user (langchain.mcp would
  otherwise turn it into a LangGraph interrupt and leave the agent suspended).
- The server URL comes from the request payload: it passes the SSRF guard first.
- Header values are never logged (they carry tokens): only their keys. URLs are
  logged as scheme://host[:port].
- parameters.tool_arguments: fixed top-level arguments set on every call of the tools
  that declare them, and hidden from the schema the model sees.
- A tool call that fails (HTTP error, timeout, connection) reaches the model as an error
  ToolMessage instead of ending the agent run.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx
import httpx2

from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool, ToolException
from mcp.shared.exceptions import MCPError

from tilellm.models.base import ServerConfig, _redact_url
from tilellm.shared.outbound_url import BlockedURLError, validate_outbound_url

logger = logging.getLogger(__name__)

_ERROR_MESSAGE_LIMIT = 500
_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://\S+")
# mcp uses httpx2, the rest of the code (and LLM SDKs) httpx
_HTTP_STATUS_ERRORS = (httpx.HTTPStatusError, httpx2.HTTPStatusError)
_NEVER_SWALLOWED = (asyncio.CancelledError, KeyboardInterrupt, SystemExit)


def _request_timeout() -> float:
    return float(os.environ.get("MCP_REQUEST_TIMEOUT", "120"))


def _redact_urls(text: str) -> str:
    """Exception texts (httpx HTTPStatusError, ConnectError...) carry the full URL, query token included."""
    return _URL_RE.sub("<url>", text)


async def _decline_elicitation(message, response_type, params, context):
    return ElicitResult(action="decline")


def build_client(config: ServerConfig) -> Client:
    # ponytail: the URL is checked once here; the SDK follows redirects only within the
    # same origin, so later hops can't leave it (DNS rebinding: network policy).
    validate_outbound_url(config.url)
    headers = dict(config.headers or {})
    if config.transport == "sse":
        logger.warning("MCP: il trasporto HTTP+SSE è deprecato dalla specifica 2026-07-28 "
                       "(url=%r): passare a streamable_http.", _redact_url(config.url))
        transport = SSETransport(config.url, headers=headers)
    else:
        transport = StreamableHttpTransport(config.url, headers=headers)
    # Without a timeout a lost response waits forever: mcp's SSE transport drops a 4xx
    # on the POST and the stream (pings) stays open.
    # ponytail: one timeout for every request (initialize, list, call) of every server;
    # make it per-server (ServerConfig) if a tool legitimately needs longer.
    return Client(transport, mode=config.protocol, elicitation_handler=_decline_elicitation,
                  timeout=_request_timeout())


async def _load_server_tools(config: ServerConfig) -> List[BaseTool]:
    async with MCPAdapter(build_client(config)) as adapter:
        return await adapter.list_tools()


def _root_exception(exc: BaseException) -> BaseException:
    """Extract the innermost cause from ExceptionGroup / __cause__ chains."""
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        return _root_exception(exc.exceptions[0])
    if exc.__cause__ and exc.__cause__ is not exc:
        return _root_exception(exc.__cause__)
    return exc


def _jsonrpc_error_message(response: Any) -> Optional[str]:
    try:
        message = json.loads(response.content)["error"]["message"]
    except Exception:  # body not read, not JSON, not a JSON-RPC error
        return None
    return str(message) if message else None


def describe_exception(exc: BaseException) -> str:
    """Short description of the root cause, for the model or the client: no URLs, no headers.

    mcp turns a JSON-RPC error body (also on an HTTP 4xx/5xx) into an MCPError: its
    message is the server's own text. The HTTP status is not kept by mcp.
    """
    root = _root_exception(exc)
    if isinstance(root, MCPError):
        text = root.message
    elif isinstance(root, _HTTP_STATUS_ERRORS):
        text = f"HTTP {root.response.status_code}"
        message = _jsonrpc_error_message(root.response)
        if message:
            text += f" — {message}"
    else:
        text = f"{type(root).__name__}: {root}".rstrip(": ")
    return _redact_urls(text)[:_ERROR_MESSAGE_LIMIT]


class _ToolCallFailed(ToolException):
    """A tool call that raised (transport, timeout...): handed to the model as an error."""


def _prepare_tool(tool: BaseTool, server_name: str, tool_arguments: Dict[str, Any]) -> bool:
    """Set up one MCP tool, in place: fixed arguments and call errors back to the model.

    The configured top-level keys the tool declares are removed from its schema (a
    copy) and set on every call, replacing the model's value. Returns True if the
    tool declares at least one of them.
    """
    schema = tool.args_schema if isinstance(tool.args_schema, dict) else {}
    properties = schema.get("properties") or {}
    fixed = {k: v for k, v in tool_arguments.items() if k in properties}
    if fixed:
        tool.args_schema = {
            **schema,
            "properties": {k: v for k, v in properties.items() if k not in fixed},
            "required": [r for r in schema.get("required", []) if r not in fixed],
        }

    call = tool.coroutine
    name = tool.name

    async def call_with_fixed_arguments(**arguments: Any) -> Any:
        try:
            return await call(**{**arguments, **fixed})
        except ToolException:
            raise  # isError result: langchain.mcp already hands it to the model
        except BaseException as exc:  # includes BaseExceptionGroup from anyio task groups
            if isinstance(_root_exception(exc), _NEVER_SWALLOWED):
                raise
            message = f"MCP tool {name} failed: {describe_exception(exc)}"
            logger.warning("MCP server %r: %s", server_name, message)
            raise _ToolCallFailed(message) from exc

    handle_mcp_error = tool.handle_tool_error

    def handle_tool_error(error: ToolException) -> Any:
        if isinstance(error, _ToolCallFailed) or not callable(handle_mcp_error):
            return str(error)
        return handle_mcp_error(error)

    tool.coroutine = call_with_fixed_arguments
    tool.handle_tool_error = handle_tool_error
    return bool(fixed)


def _prepare_server_tools(server_name: str, config: ServerConfig, tools: List[BaseTool]) -> None:
    tool_arguments = (config.parameters or {}).get("tool_arguments") or {}
    if not isinstance(tool_arguments, dict):
        logger.error("MCP server %r: parameters.tool_arguments deve essere un oggetto "
                     "{chiave: valore}, ricevuto %s: configurazione ignorata",
                     server_name, type(tool_arguments).__name__)
        tool_arguments = {}
    declared = [_prepare_tool(t, server_name, tool_arguments) for t in tools]
    if tool_arguments and not any(declared):
        logger.warning("MCP server %r: nessun tool dichiara le chiavi di tool_arguments %s: "
                       "non verranno inviate", server_name, sorted(tool_arguments))


def _format_exception_details(exc: BaseException, *, indent: str = "") -> str:
    """Format exception text for logs, unwrapping ExceptionGroup / TaskGroup wrappers."""
    if isinstance(exc, BaseExceptionGroup):
        lines = [f"{indent}{type(exc).__name__}: {exc}"]
        for i, sub in enumerate(exc.exceptions):
            lines.append(f"{indent}  sub-exception [{i}]:")
            lines.append(_format_exception_details(sub, indent=indent + "    "))
        return "\n".join(lines)
    lines = [f"{indent}{type(exc).__name__}: {exc}"]
    if exc.__cause__:
        lines.append(f"{indent}  cause:")
        lines.append(_format_exception_details(exc.__cause__, indent=indent + "    "))
    elif exc.__context__ and not exc.__suppress_context__:
        lines.append(f"{indent}  context:")
        lines.append(_format_exception_details(exc.__context__, indent=indent + "    "))
    return "\n".join(lines)


def _describe_mcp_connection_error(exc: BaseException) -> str:
    """Return a short, actionable hint for common MCP connection failures."""
    root = _root_exception(exc)
    msg = str(root).lower()
    hints: List[str] = []

    if "missing an 'http://' or 'https://' protocol" in msg:
        hints.append(
            "URL malformato: spesso c'è uno spazio prima di 'https://' "
            "(es. url=' https://...'). Controlla servers.<nome>.url nel payload."
        )
    if "connection refused" in msg:
        hints.append(
            "Il server MCP non accetta connessioni (connection refused). "
            "Verifica che il servizio sia avviato e che host/porta siano corretti."
        )
    if "timeout" in msg or "timed out" in msg:
        hints.append(
            "Timeout verso il server MCP. Verifica rete, ngrok, firewall o che l'URL sia raggiungibile."
        )
    if "401" in msg or "unauthorized" in msg:
        hints.append(
            "Autenticazione fallita (401). Verifica x-chatbotToken e gli altri header."
        )
    if "403" in msg or "forbidden" in msg:
        hints.append("Accesso negato (403). Verifica permessi e token.")
    if "404" in msg or "not found" in msg:
        hints.append(
            "Endpoint non trovato (404). Verifica il path MCP nell'URL (es. /communicator/mcp)."
        )
    if any(
        token in msg
        for token in ("getaddrinfo", "name resolution", "nodename nor servname", "failed to resolve")
    ):
        hints.append("Host DNS non risolvibile. Verifica il dominio nell'URL.")
    if "ssl" in msg or "certificate" in msg:
        hints.append("Errore TLS/certificato SSL. Verifica HTTPS e certificati del server MCP.")
    if "illegal header value" in msg:
        hints.append(
            "Un valore di header contiene un carattere non ammesso (es. un ritorno "
            "a capo) — probabile testo lungo/multi-riga finito per errore in un "
            "header invece che nel body. Controlla servers.<nome>.headers nel payload."
        )

    if hints:
        return " | ".join(hints)
    return _redact_urls(f"Causa radice: {type(root).__name__}: {root}")


def _find_illegal_header_control_char(headers: Optional[Dict[str, str]]) -> Optional[str]:
    """Return a short description of the first header value containing a CR/LF
    or other control character illegal in an HTTP header value per RFC 7230,
    or None if headers are clean.

    httpcore/h11 already reject these (LocalProtocolError: Illegal header
    value) — but only after opening a connection, 40 stack frames inside an
    ExceptionGroup that names h11 internals, not the offending header. This
    check catches the same problem before spending a round-trip on a request
    that can only fail, with a message that names the exact header and server.
    """
    if not headers:
        return None
    for key, value in headers.items():
        if not isinstance(value, str):
            continue
        pos = next((i for i, c in enumerate(value) if ord(c) < 0x20 and c != "\t"), None)
        if pos is not None:
            # the value may be a token: describe it, never show it
            return (
                f"header '{key}' contiene un carattere di controllo non ammesso in "
                f"un header HTTP (RFC 7230), probabilmente un ritorno a capo: "
                f"{value[pos]!r} in posizione {pos} (len={len(value)})"
            )
    return None


def _log_server(server_name: str, config: ServerConfig) -> None:
    logger.info(
        "MCP list_tools — server=%r transport=%s protocol=%s url=%r enabled_tools=%s header_keys=%s",
        server_name, config.transport, config.protocol, _redact_url(config.url), config.enabled_tools,
        sorted(config.headers) if config.headers else None,
    )


def _log_error(server_name: str, config: ServerConfig, exc: Exception) -> None:
    root = _root_exception(exc)
    logger.error("Errore nel recupero tool per il server %r — sintesi: %s",
                 server_name, _describe_mcp_connection_error(exc))
    logger.error("Errore nel recupero tool per il server %r — config: transport=%s url=%r",
                 server_name, config.transport, _redact_url(config.url))
    logger.error("Errore nel recupero tool per il server %r — causa radice: %s",
                 server_name, _redact_urls(f"{type(root).__name__}: {root}"))
    # no exc_info: the traceback would repeat the exception texts, URLs included
    logger.error("Errore nel recupero tool per il server %r — dettaglio completo:\n%s",
                 server_name, _redact_urls(_format_exception_details(exc)))


async def load_tools(servers_config: Dict[str, ServerConfig]) -> List[BaseTool]:
    """Tools of every configured server, filtered by enabled_tools. A failing server
    is logged and skipped: it never takes the whole request down."""
    final_tools: List[BaseTool] = []
    for server_name, config in servers_config.items():
        illegal_header = _find_illegal_header_control_char(config.headers)
        if illegal_header:
            logger.error(
                "Errore nel recupero tool per il server %r — %s. Nessun tentativo "
                "di connessione effettuato. Probabile errore di configurazione a "
                "monte (es. il corpo di un messaggio finito in un header pensato "
                "per un valore breve).",
                server_name, illegal_header,
            )
            continue
        if not config.enabled_tools:
            continue
        try:
            _log_server(server_name, config)
            server_tools = await _load_server_tools(config)
        except BlockedURLError as e:
            logger.error("MCP server %r rifiutato: %s (url=%r)", server_name,
                         _redact_urls(str(e)), _redact_url(config.url))
            continue
        except Exception as e:
            _log_error(server_name, config, e)
            continue
        logger.info("MCP list_tools — server=%r OK: %d tool: %s",
                    server_name, len(server_tools), [t.name for t in server_tools])
        _prepare_server_tools(server_name, config, server_tools)
        if "all" in config.enabled_tools:
            final_tools.extend(server_tools)
        else:
            final_tools.extend(t for t in server_tools if t.name in config.enabled_tools)
    return final_tools
