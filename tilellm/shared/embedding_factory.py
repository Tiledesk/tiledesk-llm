from functools import wraps, lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from huggingface_hub import snapshot_download
from langchain_community.embeddings import CohereEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from openai import api_key
from pydantic import BaseModel, ValidationError, SecretStr
import torch
from langchain.embeddings.base import Embeddings
import logging

from langchain_huggingface import HuggingFaceEmbeddings

from tilellm.models import EmbeddingModel, LlmEmbeddingModel
from tilellm.shared.timed_cache import TimedCache


class EmbeddingFactory:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.provider_map = {
            "openai": self._create_openai,
            "huggingface": self._create_huggingface,
            "ollama": self._create_ollama,
            "google": self._create_google,
            "cohere": self._create_cohere,
            "vllm": self._create_vllm,
            "voyage": self._create_voyage
        }
        #self._gpu_model_cache: Dict[str, Embeddings] = {}
        self.logger.info("EmbeddingFactory initialized with cache ")

    def _get_cache_key(self, config: Dict[str, Any]) -> Tuple:
        """
        Genera una chiave di cache univoca e immutabile dalla configurazione.
        La chiave include i parametri che definiscono un'istanza unica del modello.
        """
        key_parts = []
        key_parts.append(config.get("provider"))
        key_parts.append(config.get("model_name"))

        # L'API key è fondamentale per l'univocità
        api_key = config.get("api_key")
        if api_key:
            # Gestisce sia stringhe che SecretStr
            key_parts.append(str(api_key))

        # Parametri specifici che alterano l'oggetto creato
        if config.get("provider") == "huggingface":
            key_parts.append(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            key_parts.append(config.get("normalize", True))

        if config.get("base_url"):
            key_parts.append(config.get("base_url"))

        return tuple(key_parts)

    def create(self, config: Dict[str, Any]) -> Tuple[Embeddings, int]:
        """
        Metodo principale per creare o recuperare gli embedding dalla cache.
        """
        try:
            cache_key = self._get_cache_key(config)

            # Funzione che sa come creare l'oggetto se non è in cache
            def _creator() -> Tuple[Embeddings, int]:
                self.logger.info(f"Cache miss for key {cache_key}. Creating new embedding object.")
                if config.get("legacy_mode", False):
                    return self._create_legacy(config)

                provider = config["provider"].lower()
                if provider not in self.provider_map:
                    raise ValueError(f"Provider non supportato: {provider}")

                return self.provider_map[provider](config)

            # Usa TimedCache.get per ottenere l'oggetto
            # Il costruttore (_creator) verrà chiamato solo se l'oggetto non è in cache
            embedding_obj, dimension = TimedCache.get(
                object_type="embedding",
                key=cache_key,
                constructor=_creator
            )

            return embedding_obj, dimension

        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}", exc_info=True)
            raise EmbeddingCreationError("Errore nella creazione degli embedding", e)


    def _generate_model_id(self, config: Dict[str, Any]) -> str:
        """Genera un ID univoco per il caching del modello."""
        # Un ID semplice basato su provider e nome del modello è spesso sufficiente.
        # Se device, normalize_embeddings, o altri parametri influenzano il modello caricato,
        # dovresti includerli nell'ID della cache.
        if config.get("legacy_mode", False):
            return f"legacy_{config.get('model_name')}"

        provider = config.get("provider", "unknown").lower()
        model_name = config.get("model_name", "default")

        # Per HuggingFace, includi il device e normalize_embeddings nell'ID della cache
        # se diverse combinazioni di questi parametri richiedono istanze diverse.
        if provider == "huggingface":
            device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            normalize = config.get("normalize", True)
            return f"{provider}_{model_name}_{device}_{normalize}"

        return f"{provider}_{model_name}"

    def _get_default_dimension(self, provider: str, model_name: str) -> int:
        """Restituisce la dimensione di default per un dato provider/modello."""
        # Puoi espandere questo dizionario con tutte le dimensioni note
        dimensions = {
            "openai": {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            },
            "huggingface": {
                "BAAI/bge-m3": 1024,  # Aggiunto esplicitamente per bge-m3
                # Aggiungi altri modelli HuggingFace con le loro dimensioni
            },
            "ollama": 4096,
            "google": 768,
            "cohere": 1024,
            "vllm": 3072,
            "voyage": 1024,
        }

        if provider in dimensions and isinstance(dimensions[provider], dict):
            return dimensions[provider].get(model_name, 768)  # Fallback per modelli specifici
        elif provider in dimensions:
            return dimensions[provider]  # Per provider con una singola dimensione predefinita
        return 768  # Default fallback

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
        #from langchain_huggingface import HuggingFaceEmbeddings

        #self._prepare_huggingface_model(config["model_name"])

        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        return HuggingFaceEmbeddings(
            model_name=config["model_name"],
            model_kwargs={"device": device, "trust_remote_code":True},
            encode_kwargs={"normalize_embeddings": config.get("normalize", True)}
        ), config.get("dimension", 768)

    def _create_ollama(self, config: Dict) -> Tuple[Embeddings, int]:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config["model_name"],
            base_url=config.get("base_url", "http://localhost:11434")
        ), config.get("dimension", 4096)

    def _create_vllm(self, config: Dict) -> Tuple[Embeddings, int]:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config["model_name"],
            base_url=config.get("base_url", "http://localhost:8001"),
            api_key=SecretStr("a")
        ), config.get("dimension", 3072)

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
            "vllm": (self._create_vllm, 3072),
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
    """Decoratore per l'iniezione degli embedding con supporto Union"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, item, *args, **kwargs):
            nonlocal factory
            factory = factory or EmbeddingFactory()

            try:
                # Gestione del tipo Union: stringa o modello
                if isinstance(item.embedding, str):
                    # Modalità legacy: usa la stringa come model_name
                    config = {
                        "legacy_mode": True,
                        "model_name": item.embedding,
                        "api_key": item.gptkey
                    }
                elif isinstance(item.embedding, LlmEmbeddingModel):
                    # Nuova modalità: usa l'oggetto LlmEmbeddingModel
                    config = {
                        "provider": item.embedding.provider,
                        "model_name": item.embedding.name,
                        "api_key": item.embedding.api_key,
                        "dimension": item.embedding.dimension,
                        "base_url": item.embedding.url
                    }
                else:
                    raise TypeError(f"Tipo non supportato per embedding: {type(item.embedding)}")

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

