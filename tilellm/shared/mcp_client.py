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
- Header values are never logged (they carry tokens): only their keys.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool

from tilellm.models.base import ServerConfig
from tilellm.shared.outbound_url import BlockedURLError, validate_outbound_url

logger = logging.getLogger(__name__)


async def _decline_elicitation(message, response_type, params, context):
    return ElicitResult(action="decline")


def build_client(config: ServerConfig) -> Client:
    # ponytail: the URL is checked once here; the SDK follows redirects only within the
    # same origin, so later hops can't leave it (DNS rebinding: network policy).
    validate_outbound_url(config.url)
    headers = dict(config.headers or {})
    if config.transport == "sse":
        logger.warning("MCP: il trasporto HTTP+SSE è deprecato dalla specifica 2026-07-28 "
                       "(url=%r): passare a streamable_http.", config.url)
        transport = SSETransport(config.url, headers=headers)
    else:
        transport = StreamableHttpTransport(config.url, headers=headers)
    return Client(transport, mode=config.protocol, elicitation_handler=_decline_elicitation)


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
    return f"Causa radice: {type(root).__name__}: {root}"


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
        if any(ord(c) < 0x20 and c != "\t" for c in value):
            preview = value[:80].replace("\n", "\\n").replace("\r", "\\r")
            if len(value) > 80:
                preview += "..."
            return (
                f"header '{key}' contiene un carattere di controllo non ammesso in "
                f"un header HTTP (RFC 7230), probabilmente un ritorno a capo: "
                f"'{preview}' (len={len(value)})"
            )
    return None


def _log_server(server_name: str, config: ServerConfig) -> None:
    logger.info(
        "MCP list_tools — server=%r transport=%s protocol=%s url=%r enabled_tools=%s header_keys=%s",
        server_name, config.transport, config.protocol, config.url, config.enabled_tools,
        sorted(config.headers) if config.headers else None,
    )


def _log_error(server_name: str, config: ServerConfig, exc: Exception) -> None:
    root = _root_exception(exc)
    logger.error("Errore nel recupero tool per il server %r — sintesi: %s",
                 server_name, _describe_mcp_connection_error(exc))
    logger.error("Errore nel recupero tool per il server %r — config: transport=%s url=%r",
                 server_name, config.transport, config.url)
    logger.error("Errore nel recupero tool per il server %r — causa radice: %s: %s",
                 server_name, type(root).__name__, root)
    logger.error("Errore nel recupero tool per il server %r — dettaglio completo:\n%s",
                 server_name, _format_exception_details(exc), exc_info=True)


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
            logger.error("MCP server %r rifiutato: %s (url=%r)", server_name, e, config.url)
            continue
        except Exception as e:
            _log_error(server_name, config, e)
            continue
        logger.info("MCP list_tools — server=%r OK: %d tool: %s",
                    server_name, len(server_tools), [t.name for t in server_tools])
        if "all" in config.enabled_tools:
            final_tools.extend(server_tools)
        else:
            final_tools.extend(t for t in server_tools if t.name in config.enabled_tools)
    return final_tools
