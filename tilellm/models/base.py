from enum import Enum
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing import Optional, List, Dict


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
    aws_secret_access_key: str
    region_name: str

# Solo trasporti di rete: la config arriva dal payload della richiesta, e con
# "stdio" il client MCP avvierebbe `command args` come processo nel nostro pod.
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

    @model_validator(mode='after')
    def validate_transport_specific_fields(self):
        if self.transport not in MCP_ALLOWED_TRANSPORTS:
            raise ValueError(
                f"MCP transport {self.transport!r} non ammesso: usare uno tra {MCP_ALLOWED_TRANSPORTS}"
            )
        if not self.url:
            raise ValueError(f"URL è obbligatorio per il trasporto {self.transport}")
        return self