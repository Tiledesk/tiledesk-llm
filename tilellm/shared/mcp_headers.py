"""Helpers to inspect MCP HTTP headers passed to langchain-mcp-adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Headers typically required by Tiledesk MCP Communicator (for checklist in logs).
TILEDESK_COMMUNICATOR_HEADER_KEYS = [
    "x-chatbotToken",
    "x-project_id",
    "x-conversation_id",
    "x-department_id",
    "x-chatbot_name",
    "x-chatbot_id",
    "x-user_id",
    "x-last_user_text",
]


def _format_header_value(key: str, value: Any) -> str:
    if value is None:
        return f"  [{key}] MISSING (null)"
    if not isinstance(value, str):
        value = str(value)
    issues: List[str] = []
    if not value:
        issues.append("EMPTY")
    stripped = value.strip()
    if stripped != value:
        issues.append("WHITESPACE_PADDING")
    if key.lower() in ("x-chatbottoken",) and len(value) < 20:
        issues.append("SUSPICIOUSLY_SHORT_TOKEN")
    preview = value if len(value) <= 120 else f"{value[:120]}... (truncated)"
    issue_suffix = f" WARN={','.join(issues)}" if issues else ""
    return (
        f"  [{key}] present=yes len={len(value)} "
        f"value={preview!r}{issue_suffix}"
    )


def format_headers_inspection(
    headers: Optional[Dict[str, Any]],
    *,
    label: str,
    expected_keys: Optional[List[str]] = None,
) -> str:
    """Build a multi-line report of headers for debugging."""
    h = dict(headers or {})
    lines = [f"--- MCP headers: {label} (count={len(h)}) ---"]
    if not h:
        lines.append("  (no headers — dict is empty or None)")

    checklist = expected_keys or []
    for key in checklist:
        if key in h:
            lines.append(_format_header_value(key, h[key]))
        else:
            lines.append(f"  [{key}] MISSING (not in headers dict)")

    extra_keys = sorted(k for k in h if k not in checklist)
    for key in extra_keys:
        lines.append(_format_header_value(key, h[key]) + " (extra)")

    return "\n".join(lines)


def log_headers_inspection(
    headers: Optional[Dict[str, Any]],
    *,
    label: str,
    expected_keys: Optional[List[str]] = None,
    level: int = logging.INFO,
) -> None:
    logger.log(
        level,
        "%s",
        format_headers_inspection(
            headers, label=label, expected_keys=expected_keys
        ),
    )


def resolve_mcp_connection_headers(
    mcp_client: Any, server_name: str, config_headers: Optional[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Return headers from ServerConfig and from MultiServerMCPClient.connections."""
    from_config = dict(config_headers or {})
    from_connection: Dict[str, Any] = {}
    connections = getattr(mcp_client, "connections", None)
    if isinstance(connections, dict) and server_name in connections:
        conn = connections[server_name]
        if isinstance(conn, dict):
            from_connection = dict(conn.get("headers") or {})
    return {
        "server_config": from_config,
        "mcp_client_connection": from_connection,
    }


def log_mcp_headers_at_tool_call(
    mcp_client: Any,
    server_name: str,
    config_headers: Optional[Dict[str, Any]],
    *,
    tool_name: str,
    url: Optional[str] = None,
) -> None:
    """Log header details immediately before an MCP tool HTTP request."""
    bundles = resolve_mcp_connection_headers(mcp_client, server_name, config_headers)
    checklist = TILEDESK_COMMUNICATOR_HEADER_KEYS

    logger.info(
        "MCP tool call '%s' server='%s' url=%s — inspecting HTTP headers forwarded to MCP",
        tool_name,
        server_name,
        url,
    )
    log_headers_inspection(
        bundles["server_config"],
        label=f"tool={tool_name} source=QuestionToLLM.servers['{server_name}'].headers",
        expected_keys=checklist,
    )
    log_headers_inspection(
        bundles["mcp_client_connection"],
        label=f"tool={tool_name} source=MultiServerMCPClient.connections['{server_name}'].headers",
        expected_keys=checklist,
    )

    if bundles["server_config"] != bundles["mcp_client_connection"]:
        logger.warning(
            "MCP headers mismatch for server '%s': ServerConfig has %d keys, "
            "mcp_client connection has %d keys — tool HTTP calls use the connection dict",
            server_name,
            len(bundles["server_config"]),
            len(bundles["mcp_client_connection"]),
        )
