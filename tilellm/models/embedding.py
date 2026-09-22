import logging

from pydantic import BaseModel, Field, SecretStr, model_validator #, field_validator, validator
from typing import Optional, Dict, Any
from huggingface_hub import snapshot_download # Potrebbe andare in utils/huggingface_utils.py

#from pydantic.v1 import validator

from tilellm.models.base import LLMEmbeddingProviders
logger = logging.getLogger(__name__)

# Potrebbe essere spostato in un file di utilities se non strettamente legato al modello
def prepare_huggingface_model(model_name: str):
    """Scarica e cachea il modello Hugging Face"""
    return snapshot_download(
        repo_id=model_name,
        #local_dir=f"./models/{model_name.replace('/', '_')}"
    )

EMBEDDING_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "normalize": True,
        "device": "auto"
    },
    "BAAI/bge-m3": {
        "dimension": 1024,
        "normalize": False,
        "device": "cuda"
    },
    "voyage-multilingual-2": {
        "dimension": 1024,
        "voyage_api_key": "your-default-key"
    }
}

class LlmEmbeddingModel(BaseModel):
    provider: LLMEmbeddingProviders
    name: str
    api_key: Optional[SecretStr] | None = None
    url: Optional[str] = Field(default_factory=lambda: "")
    dimension: Optional[int] = 1024 #qwel2-deepseek 3584, llama3.2 3072
    custom_headers: Optional[Dict[str, Any]] = None
    project: Optional[str] = None  # GCP project id, routes google provider to Vertex AI
    location: Optional[str] = None  # GCP region for Vertex AI (e.g. europe-west8)
    # OpenRouter only: which upstream providers may serve this model, and in what order.
    # Shape: {"order": ["azure", "openai"], "allow_fallbacks": true, "sort": "price"}.
    # Passed through to the OpenRouter API as the request's "provider" block.
    provider_routing: Optional[Dict[str, Any]] = None
    # vllm provider only: the "vllm" slot is also used for any custom OpenAI-compatible
    # endpoint (e.g. Cerebras via a custom base_url), not just genuine self-hosted vLLM.
    # utility.py auto-sends extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    # to every "vllm" request by default — needed for Qwen3/thinking models actually
    # served by vLLM (otherwise thinking eats max_tokens and the response comes back
    # empty), but some strict OpenAI-compatible backends (Cerebras confirmed 2026-09-22)
    # reject any unrecognized body field with a 400 instead of ignoring it. Default None
    # preserves that existing auto-apply behavior; set False to opt out for a backend that
    # rejects it, True to force it on (same as default, for explicitness in configs).
    disable_thinking_mode: Optional[bool] = None

    @model_validator(mode='after')
    def validate_model(self):
        logger.debug(f"Validazione dopo l'inizializzazione del modello: {self.name} con provider {self.provider}")
        if self.provider == LLMEmbeddingProviders.HUGGINGFACE:
            prepare_huggingface_model(self.name)
        return self


class EmbeddingModel(BaseModel):
    embedding_provider: str
    embedding_key: Optional[SecretStr]| None = None
    embedding_model: str
    embedding_host: Optional[str] = Field(default=None)
    embedding_dimension: Optional[int] = None
    embedding_custom_headers: Optional[Dict[str, Any]] = None