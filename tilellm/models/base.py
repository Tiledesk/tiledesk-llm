from enum import Enum
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing import Optional, List, Dict
from urllib.parse import urlsplit


def _redact_url(url: Optional[str]) -> Optional[str]:
    """scheme://host[:port] only: path, query and userinfo may carry credentials."""
    if not url:
        return url
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        return "<url non valido>"
    if not parts.scheme or not host:
        return "<url non valido>"
    if ":" in host:
        host = f"[{host}]"
    return f"{parts.scheme}://{host}" + (f":{port}" if port else "")


class LLMEmbeddingProviders(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GOOGLE = "google"
    COHERE = "cohere"
    VOYAGE = "voyage"
    VLLM = "vllm"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    TEI = "tei"
    DEEPSEEK="deepseek"
    OPENROUTER="openrouter"

class AWSAuthentication(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str = Field(repr=False)
    region_name: str

MCP_ALLOWED_TRANSPORTS = ("streamable_http", "sse")


class ServerConfig(BaseModel):
    """Modello per la configurazione di un server MCP"""
    transport: str
    url: Optional[str] = None
    api_key: Optional[SecretStr] = None
    headers: Optional[Dict[str, str]] = Field(default=None, description="HTTP headers to send to the MCP server (e.g. x-composio-user-id for Composio)")
    enabled_tools: Optional[List[str]] = Field(default_factory=lambda: ["all"])
    parameters: Optional[dict] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_url_and_headers(self):
        if self.url is not None:
            self.url = self.url.strip()
        if self.headers:
            self.headers = {
                k: (v.strip() if isinstance(v, str) else v)
                for k, v in self.headers.items()
            }
        return self

    def __repr_args__(self):
        # used by repr() and str(): the request is logged, keep values out
        yield "transport", self.transport
        yield "url", _redact_url(self.url)
        yield "headers", sorted(self.headers or {})
        yield "api_key", "**********" if self.api_key else None
        yield "enabled_tools", self.enabled_tools
        yield "parameters", sorted(self.parameters or {})

    @model_validator(mode='after')
    def validate_transport_specific_fields(self):
        if self.transport not in MCP_ALLOWED_TRANSPORTS:
            raise ValueError(
                f"MCP transport {self.transport!r} non ammesso: usare uno tra {MCP_ALLOWED_TRANSPORTS}"
            )
        if not self.url:
            raise ValueError(f"URL è obbligatorio per il trasporto {self.transport}")
        return self

