from functools import wraps, lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from huggingface_hub import snapshot_download
from langchain_community.embeddings import CohereEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, ValidationError
import torch
from langchain.embeddings.base import Embeddings
import logging

from tilellm.models.item_model import EmbeddingModel


class EmbeddingFactory:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.provider_map = {
            "openai": self._create_openai,
            "huggingface": self._create_huggingface,
            "ollama": self._create_ollama,
            "google": self._create_google,
            "cohere": self._create_cohere,
            "voyage": self._create_voyage
        }

    def create(self, config: Dict[str, Any]) ->Tuple[Embeddings, int]:
        """Metodo principale per creare gli embedding"""

        try:
            if config.get("legacy_mode", False):
                embedding, dimension = self._create_legacy(config)
                return embedding, dimension

            provider = config["provider"].lower()
            if provider not in self.provider_map:
                raise ValueError(f"Provider non supportato: {provider}")

            embedding, dimension = self.provider_map[provider](config)
            return embedding, dimension

        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}", exc_info=True)
            raise EmbeddingCreationError("Errore nella creazione degli embedding", e)

    def _create_openai(self, config: Dict) -> Tuple[Embeddings, int]:
        from langchain_openai import OpenAIEmbeddings
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        model_name = config.get("model_name", "text-embedding-ada-002")
        return OpenAIEmbeddings(
            api_key=config["api_key"],
            model=model_name
        ), model_dimensions.get(model_name, 1536)

    def _create_huggingface(self, config: Dict) -> Tuple[Embeddings, int]:
        from langchain_huggingface import HuggingFaceEmbeddings

        #self._prepare_huggingface_model(config["model_name"])

        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        return HuggingFaceEmbeddings(
            model_name=config["model_name"],
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": config.get("normalize", True)}
        ), config.get("dimension", 768)

    def _create_ollama(self, config: Dict) -> Tuple[Embeddings, int]:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config["model_name"],
            base_url=config.get("base_url", "http://localhost:11434")
        ), config.get("dimension", 4096)

    def _create_google(self, config: Dict) -> Tuple[Embeddings, int]:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config["model_name"],
            google_api_key=config["api_key"]
        ), config.get("dimension", 768)

    def _create_cohere(self, config: Dict) -> Tuple[Embeddings, int]:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(
            model=config["model_name"],
            cohere_api_key=config["api_key"]
        ), config.get("dimension", 1024)

    def _create_voyage(self, config: Dict) -> Tuple[Embeddings, int]:
        from langchain_voyageai import VoyageAIEmbeddings
        return VoyageAIEmbeddings(
            model=config["model_name"],
            voyage_api_key=config["api_key"]
        ), config.get("dimension", 1024)

    def _create_legacy(self, config: Dict) -> Tuple[Embeddings, int]:
        legacy_models = {
            "text-embedding-ada-002": (self._create_openai, 1536),
            "text-embedding-3-small": (self._create_openai, 1536),
            "text-embedding-3-large": (self._create_openai, 3072),
            "huggingface": (self._create_huggingface, 1024),
            "ollama": (self._create_ollama, 4096),
            "google": (self._create_google, 768),
            "cohere": (self._create_cohere, 1024),
            "voyage-multilingual-2": (self._create_voyage, 1024)
        }

        model_name = config["model_name"]
        if model_name not in legacy_models:
            embedding_obj, _ = self._create_openai(config)
            return embedding_obj, 1536  # Return esplicito senza nesting

        creator, default_dim = legacy_models[model_name]

        # Modifica chiave: spacchetta il risultato del creator
        embedding_obj, creator_dim = creator(config)
        return embedding_obj, default_dim  # Usa default_dim come override

    @staticmethod
    def _prepare_huggingface_model(model_name: str):
        return snapshot_download(
            repo_id=model_name,
            local_dir=f"./models/{model_name.replace('/', '_')}"
        )


class EmbeddingCreationError(Exception):
    """Eccezione personalizzata per errori nella creazione degli embedding"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(f"{message} - Original error: {str(original_error)}" if original_error else message)


def inject_embedding(factory: Optional[EmbeddingFactory] = None):
    """Decoratore per l'iniezione degli embedding"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, item, *args, **kwargs):
            nonlocal factory
            factory = factory or EmbeddingFactory()

            try:

                # Costruisci la configurazione
                if item.model:
                    config = {
                        "provider": item.model.provider,
                        "model_name": item.model.name,
                        "api_key": item.gptkey,
                        "dimension": item.model.dimension,
                        "base_url": item.model.url
                    }
                else:
                    config = {
                        "legacy_mode": True,
                        "model_name": item.embedding,
                        "api_key": item.gptkey
                    }

                # Crea gli embedding
                result = factory.create(config)
                # Gestione del tipo di ritorno
                if isinstance(result, tuple):
                    embedding_obj, dimension = result
                else:
                    embedding_obj = result
                    dimension = 1536  # Default

                # Inietta nei kwargs
                kwargs["embedding_obj"] = embedding_obj
                kwargs["embedding_dimension"] = dimension

                return await func(self, item, *args, **kwargs)

            except ValidationError as ve:
                raise EmbeddingCreationError("Validazione fallita", ve)
            except Exception as e:
                raise EmbeddingCreationError("Errore generico", e)

        return wrapper

    return decorator


def inject_embedding_qa(factory: Optional[EmbeddingFactory] = None):
    """Decoratore per l'iniezione degli embedding"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, question, *args, **kwargs):
            nonlocal factory
            factory = factory or EmbeddingFactory()

            try:

                if isinstance(question.embedding, EmbeddingModel):
                    provider = question.embedding.embedding_provider
                    key = question.embedding.embedding_key
                    model_name = question.embedding.embedding_model
                    base_url = question.embedding.embedding_host
                    dimensione = question.embedding.embedding_dimension
                    config = {
                        "provider": provider,
                        "model_name": model_name,
                        "api_key": key,
                        "dimension": dimensione,
                        "base_url": base_url
                    }
                else:
                    config = {
                        "legacy_mode": True,
                        "model_name": question.embedding,
                        "api_key": question.gptkey
                    }

                # Crea gli embedding
                result = factory.create(config)
                # Gestione del tipo di ritorno
                if isinstance(result, tuple):
                    embedding_obj, dimension = result
                else:
                    embedding_obj = result
                    dimension = 1536  # Default

                # Inietta nei kwargs
                kwargs["embedding_obj"] = embedding_obj
                kwargs["embedding_dimension"] = dimension

                return await func(self, question, *args, **kwargs)

            except ValidationError as ve:
                raise EmbeddingCreationError("Validazione fallita", ve)
            except Exception as e:
                raise EmbeddingCreationError("Errore generico", e)

        return wrapper

    return decorator

