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