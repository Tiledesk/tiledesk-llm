import asyncio
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
import hashlib

from huggingface_hub import snapshot_download

from pydantic import ValidationError
from langchain.embeddings.base import Embeddings
import logging

from langchain_huggingface import HuggingFaceEmbeddings

from tilellm.models import LlmEmbeddingModel #EmbeddingModel,
from tilellm.shared.embeddings.resilient_embeddings import ResilientEmbeddings
from tilellm.shared.timed_cache import TimedCache
from tilellm.store.vector_store_repository import VectorStoreIndexingError
from tilellm.shared.embeddings.embedding_client_manager import TEIEmbeddings


def _hash_api_key(api_key: str) -> str:
    """
    Crea un hash SHA256 della chiave API per utilizzarlo nella cache
    senza esporre la chiave completa nei log.
    """
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:16]


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
            "voyage": self._create_voyage,
            "tei": self._create_tei
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

        # L'API key è fondamentale per l'univocità - usa hash per sicurezza
        api_key = config.get("api_key")
        if api_key:
            # Gestisce sia stringhe che SecretStr e crea hash
            key_str = str(api_key.get_secret_value()) if hasattr(api_key, 'get_secret_value') else str(api_key)
            key_parts.append(_hash_api_key(key_str))

        # Parametri specifici che alterano l'oggetto creato
        if config.get("provider") == "huggingface":
            try:
                import torch
                default_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                default_device = "cpu"
            key_parts.append(config.get("device", default_device))
            key_parts.append(config.get("normalize", True))

        if config.get("base_url"):
            key_parts.append(config.get("base_url"))

        return tuple(key_parts)


    def create_embedding_cache(self, config: Dict[str, Any]) -> Tuple[Embeddings, int]:
        try:
            cache_key = self._get_cache_key(config)

            def _creator() -> Tuple[Embeddings, int]:
                self.logger.info(f"Cache miss for key {cache_key}. Creating new embedding object.")
                if config.get("legacy_mode", False):
                    emb, dim = self._create_legacy(config)
                    # legacy può essere OpenAI o simili: wrappa se rete
                    provider = "openai" if isinstance(config.get("model_name"), str) else "openai"
                    if provider in {"openai", "vllm", "ollama", "google", "cohere", "voyage"}:
                        def builder():
                            return self._create_openai({"api_key": config["api_key"], "model_name": config["model_name"]})[
                                0]

                        emb = ResilientEmbeddings(builder, seed=emb)
                    return emb, dim

                provider = config["provider"].lower()
                if provider not in self.provider_map:
                    raise ValueError(f"Provider non supportato: {provider}")
                emb, dim = self.provider_map[provider](config)

                # Wrappa solo provider “di rete”
                if provider in {"openai", "vllm", "ollama", "google", "cohere", "voyage"}:
                    def builder():
                        # ricrea l'oggetto embedding per quel provider
                        return self.provider_map[provider](config)[0]

                    emb = ResilientEmbeddings(builder, seed=emb)

                return emb, dim

            embedding_obj, dimension = TimedCache.get(
                object_type="embedding",
                key=cache_key,
                constructor=_creator
            )
            return embedding_obj, dimension

        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}", exc_info=True)
            raise EmbeddingCreationError("Errore nella creazione degli embedding", e)

    def create(self, config: Dict[str, Any]) -> Tuple[Embeddings, int]:
        return self.create_embedding_cache(config)

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
        # Per HuggingFace, includi il device e normalize_embeddings nell'ID della cache
        # se diverse combinazioni di questi parametri richiedono istanze diverse.
        if provider == "huggingface":
            try:
                import torch
                default_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                default_device = "cpu"
            device = config.get("device", default_device)
            normalize = config.get("normalize", True)
            return f"{provider}_{model_name}_{default_device}_{normalize}"

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

        try:
            import torch
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            default_device = "cpu"

        device = config.get("device", default_device)

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
        
        api_key = config.get("api_key")
        headers = config.get("custom_headers")
        return OpenAIEmbeddings(
            model=config["model_name"],
            base_url=config.get("base_url", "http://localhost:8001"),
            api_key=api_key,
            default_headers=headers
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

    def _create_tei(self, config: Dict) -> Tuple[Embeddings, int]:
        return TEIEmbeddings(
            base_url=config.get("base_url"),
            model=config.get("model_name"),
            api_key=config.get("api_key"),
            headers=config.get("custom_headers")
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
        return snapshot_download(repo_id=model_name,
                                 #local_dir=f"./models/{model_name.replace('/', '_')}"
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
                        "api_key": item.gptkey.get_secret_value()
                    }
                elif isinstance(item.embedding, LlmEmbeddingModel):
                    # Nuova modalità: usa l'oggetto LlmEmbeddingModel
                    config = {
                        "provider": item.embedding.provider,
                        "model_name": item.embedding.name,
                        "api_key": item.embedding.api_key.get_secret_value() if item.embedding.api_key else None,
                        "dimension": item.embedding.dimension,
                        "base_url": item.embedding.url,
                        "custom_headers": item.embedding.custom_headers
                    }
                else:
                    raise TypeError(f"Embedding type not supported: {type(item.embedding)}")

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

            except VectorStoreIndexingError as vsie:
                raise vsie
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

                if isinstance(question.embedding, LlmEmbeddingModel): #EmbeddingModel):
                    provider = question.embedding.provider
                    key = question.embedding.api_key
                    model_name = question.embedding.name
                    base_url = question.embedding.url
                    dimension = question.embedding.dimension
                    custom_headers = question.embedding.custom_headers
                    config = {
                        "provider": provider,
                        "model_name": model_name,
                        "api_key": key.get_secret_value() if key else None,
                        "dimension": dimension,
                        "base_url": base_url,
                        "custom_headers": custom_headers
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



class AsyncEmbeddingFactory:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.provider_map = {
            "openai": self._create_openai,
            "huggingface": self._create_huggingface,
            "ollama": self._create_ollama,
            "google": self._create_google,
            "cohere": self._create_cohere,
            "vllm": self._create_vllm,
            "voyage": self._create_voyage,
            "tei": self._create_tei
        }
        self.logger.info("AsyncEmbeddingFactory initialized with cache")

    @staticmethod
    def _get_cache_key(config: Dict[str, Any]) -> Tuple:
        """
        Genera una chiave di cache univoca e immutabile dalla configurazione.
        La chiave include i parametri che definiscono un'istanza unica del modello.
        """
        key_parts = []
        key_parts.append(config.get("provider"))
        key_parts.append(config.get("model_name"))

        # L'API key è fondamentale per l'univocità - usa hash per sicurezza
        api_key = config.get("api_key")
        if api_key:
            # Gestisce sia stringhe che SecretStr e crea hash
            key_str = str(api_key.get_secret_value()) if hasattr(api_key, 'get_secret_value') else str(api_key)
            key_parts.append(_hash_api_key(key_str))

        # Parametri specifici che alterano l'oggetto creato
        if config.get("provider") == "huggingface":
            try:
                import torch
                default_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                default_device = "cpu"
            key_parts.append(config.get("device", default_device))
            key_parts.append(config.get("normalize", True))

        if config.get("base_url"):
            key_parts.append(config.get("base_url"))

        return tuple(key_parts)

    async def create_embedding_model_cache(self, config: Dict[str, Any]) -> Tuple[Embeddings, int]:
        try:
            cache_key = self._get_cache_key(config)

            async def _async_creator() -> Tuple[Embeddings, int]:
                self.logger.info(f"Cache miss for key {cache_key}. Creating new embedding object.")
                if config.get("legacy_mode", False):
                    emb, dim = await self._create_legacy(config)

                    # legacy presumibilmente openai -> wrappa
                    def builder():
                        from langchain_openai import OpenAIEmbeddings
                        model_name = config.get("model_name", "text-embedding-ada-002")
                        return OpenAIEmbeddings(api_key=config["api_key"], model=model_name)

                    emb = ResilientEmbeddings(builder, seed=emb)
                    return emb, dim

                provider = config["provider"].lower()
                if provider not in self.provider_map:
                    raise ValueError(f"Provider non supportato: {provider}")

                emb, dim = await self.provider_map[provider](config)

                if provider in {"openai", "vllm", "ollama", "google", "cohere", "voyage"}:
                    # builder sincrono per ricreare l’oggetto al volo
                    def builder():
                        # Clona la creazione “sincrona” del provider per ricostruire velocemente.
                        # Queste classi (OpenAIEmbeddings, CohereEmbeddings, ...) hanno costruttori sincroni leggeri.
                        if provider == "openai" or provider == "vllm":
                            from langchain_openai import OpenAIEmbeddings
                            return OpenAIEmbeddings(
                                model=config["model_name"],
                                base_url=config.get("base_url"),
                                api_key=config["api_key"],
                            )
                        if provider == "ollama":
                            from langchain_ollama import OllamaEmbeddings
                            return OllamaEmbeddings(
                                model=config["model_name"],
                                base_url=config.get("base_url", "http://localhost:11434"),
                            )
                        if provider == "google":
                            from langchain_google_genai import GoogleGenerativeAIEmbeddings
                            return GoogleGenerativeAIEmbeddings(
                                model=config["model_name"],
                                google_api_key=config["api_key"],
                            )
                        if provider == "cohere":
                            from langchain_cohere import CohereEmbeddings
                            return CohereEmbeddings(
                                model=config["model_name"],
                                cohere_api_key=config["api_key"],
                            )
                        if provider == "voyage":
                            from langchain_voyageai import VoyageAIEmbeddings
                            return VoyageAIEmbeddings(
                                model=config["model_name"],
                                voyage_api_key=config["api_key"],
                            )
                        # fallback: restituisci l'oggetto esistente se mai capitasse
                        return emb

                    emb = ResilientEmbeddings(builder, seed=emb)

                return emb, dim

            embedding_obj, dimension = await TimedCache.async_get(
                object_type="embedding",
                key=cache_key,
                constructor=_async_creator
            )
            return embedding_obj, dimension

        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}", exc_info=True)
            raise EmbeddingCreationError("Errore nella creazione degli embedding", e)

    async def create(self, config: Dict[str, Any]) -> Tuple[Embeddings, int]:
        return await self.create_embedding_model_cache(config)

    @staticmethod
    def _generate_model_id(config: Dict[str, Any]) -> str:
        """Genera un ID univoco per il caching del modello."""
        if config.get("legacy_mode", False):
            return f"legacy_{config.get('model_name')}"

        provider = config.get("provider", "unknown").lower()
        model_name = config.get("model_name", "default")

        # Per HuggingFace, includi il device e normalize_embeddings nell'ID della cache
        # se diverse combinazioni di questi parametri richiedono istanze diverse.
        if provider == "huggingface":
            try:
                import torch
                default_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                default_device = "cpu"
            device = config.get("device", default_device)
            normalize = config.get("normalize", True)
            return f"{provider}_{model_name}_{default_device}_{normalize}"

        return f"{provider}_{model_name}"

    @staticmethod
    def _get_default_dimension(provider: str, model_name: str) -> int:
        """Restituisce la dimensione di default per un dato provider/modello."""
        dimensions = {
            "openai": {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            },
            "huggingface": {
                "BAAI/bge-m3": 1024,
            },
            "ollama": 4096,
            "google": 768,
            "cohere": 1024,
            "vllm": 3072,
            "voyage": 1024,
        }

        if provider in dimensions and isinstance(dimensions[provider], dict):
            return dimensions[provider].get(model_name, 768)
        elif provider in dimensions:
            return dimensions[provider]
        return 768

    @staticmethod
    async def _create_openai(config: Dict) -> Tuple[Embeddings, int]:
        """Crea embedding OpenAI in modo asincrono"""
        from langchain_openai import OpenAIEmbeddings

        def _create():
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

        # Esegui la creazione in un thread separato per non bloccare
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    async def _create_huggingface(self, config: Dict) -> Tuple[Embeddings, int]:
        """Crea embedding HuggingFace in modo asincrono"""

        def _create():
            try:
                import torch
                default_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                default_device = "cpu"
            device = config.get("device", default_device)
            return HuggingFaceEmbeddings(
                model_name=config["model_name"],
                model_kwargs={"device": device, "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": config.get("normalize", True)}
            ), config.get("dimension", 768)

        # La preparazione del modello può essere pesante, quindi la rendiamo asincrona
        if config.get("prepare_model", True):
            await self._prepare_huggingface_model_async(config["model_name"])

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    @staticmethod
    async def _create_ollama(config: Dict) -> Tuple[Embeddings, int]:
        """Crea embedding Ollama in modo asincrono"""
        from langchain_ollama import OllamaEmbeddings

        def _create():
            return OllamaEmbeddings(
                model=config["model_name"],
                base_url=config.get("base_url", "http://localhost:11434")
            ), config.get("dimension", 4096)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    @staticmethod
    async def _create_vllm(config: Dict) -> Tuple[Embeddings, int]:
        """Crea embedding VLLM in modo asincrono"""
        from langchain_openai import OpenAIEmbeddings

        def _create():
            api_key = config.get("api_key")
            headers = config.get("custom_headers")
            return OpenAIEmbeddings(
                model=config["model_name"],
                base_url=config.get("base_url", "http://localhost:8001"),
                api_key=api_key,
                default_headers=headers
            ), config.get("dimension", 3072)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    @staticmethod
    async def _create_google(config: Dict) -> Tuple[Embeddings, int]:
        """Crea embedding Google in modo asincrono"""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        def _create():
            return GoogleGenerativeAIEmbeddings(
                model=config["model_name"],
                google_api_key=config["api_key"]
            ), config.get("dimension", 768)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    @staticmethod
    async def _create_cohere(config: Dict) -> Tuple[Embeddings, int]:
        """Crea embedding Cohere in modo asincrono"""
        from langchain_cohere import CohereEmbeddings

        def _create():
            return CohereEmbeddings(
                model=config["model_name"],
                cohere_api_key=config["api_key"]
            ), config.get("dimension", 1024)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    @staticmethod
    async def _create_voyage(config: Dict) -> Tuple[Embeddings, int]:
        """Crea embedding Voyage in modo asincrono"""
        from langchain_voyageai import VoyageAIEmbeddings

        def _create():
            return VoyageAIEmbeddings(
                model=config["model_name"],
                voyage_api_key=config["api_key"]
            ), config.get("dimension", 1024)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    @staticmethod
    async def _create_tei(config: Dict) -> Tuple[Embeddings, int]:
        """Crea embedding TEI in modo asincrono"""
        def _create():
            return TEIEmbeddings(
                base_url=config.get("base_url"),
                model=config.get("model_name"),
                api_key=config.get("api_key"),
                headers=config.get("custom_headers")
            ), config.get("dimension", 1024)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    async def _create_legacy(self, config: Dict) -> Tuple[Embeddings, int]:
        """Crea embedding in modalità legacy in modo asincrono"""
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
            embedding_obj, _ = await self._create_openai(config)
            return embedding_obj, 1536

        creator, default_dim = legacy_models[model_name]
        embedding_obj, creator_dim = await creator(config)
        return embedding_obj, default_dim

    @staticmethod
    async def _prepare_huggingface_model_async(model_name: str):

        """Prepara il modello HuggingFace in modo asincrono"""

        def _download():
            return snapshot_download(repo_id=model_name,
                                     #local_dir=f"./models/{model_name.replace('/', '_')}"
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _download)



def inject_embedding_async(factory: Optional[AsyncEmbeddingFactory] = None):
    """Decoratore asincrono per l'iniezione degli embedding con supporto Union"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, item, *args, **kwargs):
            nonlocal factory
            factory = factory or AsyncEmbeddingFactory()

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
                        "api_key": item.embedding.api_key.get_secret_value() if item.embedding.api_key else None,
                        "dimension": item.embedding.dimension,
                        "base_url": item.embedding.url,
                        "custom_headers": item.embedding.custom_headers
                    }
                else:
                    raise TypeError(f"Tipo non supportato per embedding: {type(item.embedding)}")

                # Crea gli embedding in modo asincrono
                embedding_obj, dimension = await factory.create(config)

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


def inject_embedding_qa_async(factory: Optional[AsyncEmbeddingFactory] = None):
    """Decoratore asincrono per l'iniezione degli embedding per QA"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, question, *args, **kwargs):
            nonlocal factory
            factory = factory or AsyncEmbeddingFactory()

            try:
                if isinstance(question.embedding, LlmEmbeddingModel): #EmbeddingModel):
                    provider = question.embedding.provider
                    key = question.embedding.api_key
                    model_name = question.embedding.name
                    base_url = question.embedding.url
                    dimension = question.embedding.dimension
                    custom_headers = question.embedding.custom_headers
                    config = {
                        "provider": provider,
                        "model_name": model_name,
                        "api_key": key.get_secret_value() if key else None,
                        "dimension": dimension,
                        "base_url": base_url,
                        "custom_headers": custom_headers
                    }
                else:
                    config = {
                        "legacy_mode": True,
                        "model_name": question.embedding,
                        "api_key": question.gptkey
                    }

                # Crea gli embedding in modo asincrono
                embedding_obj, dimension = await factory.create(config)

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


# Esempio di uso della funzione asincrona
async def create_embedding_instance(question):
    """
    Funzione di esempio per creare un'istanza di embedding in modo asincrono
    """
    factory = AsyncEmbeddingFactory()

    # Configura in base al tipo di question
    if hasattr(question, 'embedding') and isinstance(question.embedding, LlmEmbeddingModel):#EmbeddingModel):
        embedding_config = {
            "provider": question.embedding.provider,
            "model_name": question.embedding.name,
            "api_key": question.embedding.api_key,
            "dimension": question.embedding.dimension,
            "base_url": question.embedding.url
        }
    else:
        # Modalità legacy o configurazione semplice
        embedding_config = {
            "legacy_mode": True,
            "model_name": getattr(question, 'embedding', 'text-embedding-ada-002'),
            "api_key": getattr(question, 'gptkey', None)
        }

    # Chiamata asincrona
    embedding_obj, embedding_dimension = await factory.create(embedding_config)

    return embedding_obj, embedding_dimension

