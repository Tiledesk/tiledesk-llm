from enum import Enum
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing import Optional, List

class EmbeddingProviders(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GOOGLE = "google"
    COHERE = "cohere"
    VOYAGE = "voyage"
    VLLM = "vllm"

class AWSAuthentication(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str

class ServerConfig(BaseModel):
    """Modello per la configurazione di un server MCP"""
    transport: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    api_key: Optional[SecretStr] = None
    parameters: Optional[dict] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_transport_specific_fields(self):
        # Validazione per trasporto SSE
        if self.transport == "sse" or self.transport=="streamable_http":
            if not self.url:
                raise ValueError("URL è obbligatorio per il trasporto SSE")

        # Validazione per trasporto stdio
        elif self.transport == "stdio":
            if not self.command or not self.args:
                raise ValueError("Command e args sono obbligatori per il trasporto stdio")

        return self