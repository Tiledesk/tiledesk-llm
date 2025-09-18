import asyncio
import time
import logging
from typing import Optional, Dict, Any, Tuple
from functools import wraps
from langchain.embeddings.base import Embeddings
import torch
from pydantic import ValidationError

# Importa la tua factory esistente
from tilellm.shared.timed_cache import TimedCache
from tilellm.models import EmbeddingModel, LlmEmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingSessionManager:
    """Manager per gestire sessioni HTTP persistenti per gli embeddings"""

    _instances: Dict[str, 'EmbeddingSessionManager'] = {}
    _lock = asyncio.Lock()

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._embedding_client = None
        self._ref_count = 0
        self._client_lock = asyncio.Lock()
        self._last_use = time.time()
        self._session_timeout = 300  # 5 minuti

    @classmethod
    async def get_instance(cls, config: Dict[str, Any]) -> 'EmbeddingSessionManager':
        """Ottiene l'istanza singleton per la configurazione"""
        config_key = cls._create_config_key(config)

        async with cls._lock:
            if config_key not in cls._instances:
                cls._instances[config_key] = cls(config)

            instance = cls._instances[config_key]
            instance._last_use = time.time()
            return instance

    @staticmethod
    def _create_config_key(config: Dict[str, Any]) -> str:
        """Crea chiave di configurazione compatibile con la tua factory"""
        key_parts = []
        key_parts.append(config.get("provider", ""))
        key_parts.append(config.get("model_name", ""))

        api_key = config.get("api_key")
        if api_key:
            key_parts.append(str(api_key)[:20])  # Solo primi 20 caratteri

        if config.get("provider") == "huggingface":
            key_parts.append(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            key_parts.append(str(config.get("normalize", True)))

        if config.get("base_url"):
            key_parts.append(config.get("base_url"))

        return '|'.join(key_parts)

    async def get_embedding_client(self):
        """Ottiene il client embedding con sessione persistente"""
        async with self._client_lock:
            # Se il client esiste ed è fresco, riutilizzalo
            if (self._embedding_client is not None and
                    time.time() - self._last_use < self._session_timeout):
                self._ref_count += 1
                self._last_use = time.time()
                return self._embedding_client

            # Altrimenti, ricrea il client
            if self._embedding_client is not None:
                await self._close_client()

            self._embedding_client = await self._create_embedding_client()
            self._ref_count += 1
            self._last_use = time.time()
            logger.debug(f"Created new embedding client: {self.config.get('provider', 'unknown')}")

            return self._embedding_client

    async def _create_embedding_client(self):
        """Crea il client embedding con configurazione per sessioni persistenti"""
        provider = self.config.get("provider", "").lower()

        if provider == "openai":
            return await self._create_openai_with_session()
        elif provider == "text-embedding-ada-002":
            return await self._create_openai_with_session()
        elif provider == "huggingface":
            return await self._create_huggingface_with_session()
        elif provider == "google":
            return await self._create_google_with_session()
        elif provider == "cohere":
            return await self._create_cohere_with_session()
        elif provider == "ollama":
            return await self._create_ollama_with_session()
        elif provider == "vllm":
            return await self._create_vllm_with_session()
        elif provider == "voyage":
            return await self._create_voyage_with_session()
        else:
            raise ValueError(f"Provider non supportato: {provider}")

    async def _create_openai_with_session(self):
        """Crea OpenAI embeddings con sessione HTTP persistente"""
        from langchain_openai import OpenAIEmbeddings
        import httpx

        # Crea un client HTTP persistente
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

        model_name = self.config.get("model_name", "text-embedding-ada-002")

        embedding_client = OpenAIEmbeddings(
            api_key=self.config["api_key"],
            model=model_name,
            http_async_client=http_client,  # Client HTTP persistente
            max_retries=3,
        )

        # Aggiungi riferimento al client HTTP per cleanup successivo
        embedding_client._http_client = http_client

        return embedding_client

    async def _create_huggingface_with_session(self):
        """Crea HuggingFace embeddings (non necessita gestione HTTP)"""
        from langchain_huggingface import HuggingFaceEmbeddings

        device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        return HuggingFaceEmbeddings(
            model_name=self.config["model_name"],
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": self.config.get("normalize", True)}
        )

    async def _create_google_with_session(self):
        """Crea Google embeddings con sessione persistente"""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        import httpx

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

        embedding_client = GoogleGenerativeAIEmbeddings(
            model=self.config["model_name"],
            google_api_key=self.config["api_key"],
        )

        # Alcuni client Google potrebbero supportare client HTTP personalizzati
        if hasattr(embedding_client, '_client'):
            embedding_client._http_client = http_client

        return embedding_client

    async def _create_cohere_with_session(self):
        """Crea Cohere embeddings"""
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(
            model=self.config["model_name"],
            cohere_api_key=self.config["api_key"]
        )

    async def _create_ollama_with_session(self):
        """Crea Ollama embeddings"""
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=self.config["model_name"],
            base_url=self.config.get("base_url", "http://localhost:11434")
        )

    async def _create_vllm_with_session(self):
        """Crea VLLM embeddings con sessione persistente"""
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr
        import httpx

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

        embedding_client = OpenAIEmbeddings(
            model=self.config["model_name"],
            base_url=self.config.get("base_url", "http://localhost:8001"),
            api_key=SecretStr("a"),
            http_async_client=http_client
        )

        embedding_client._http_client = http_client
        return embedding_client

    async def _create_voyage_with_session(self):
        """Crea Voyage embeddings"""
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(
            model=self.config["model_name"],
            voyage_api_key=self.config["api_key"]
        )

    async def release_client(self):
        """Rilascia un riferimento al client"""
        async with self._client_lock:
            self._ref_count = max(0, self._ref_count - 1)

            # Chiudi il client se non ci sono riferimenti e è scaduto
            if (self._ref_count == 0 and
                    time.time() - self._last_use > self._session_timeout):
                await self._close_client()

    async def _close_client(self):
        """Chiude il client e le sue sessioni HTTP"""
        if self._embedding_client is not None:
            try:
                # Chiudi sessione HTTP se presente
                if hasattr(self._embedding_client, '_http_client'):
                    await self._embedding_client._http_client.aclose()
                elif hasattr(self._embedding_client, 'client') and hasattr(self._embedding_client.client, 'close'):
                    await self._embedding_client.client.close()

                logger.debug("Closed embedding client HTTP session")
            except Exception as e:
                logger.warning(f"Error closing embedding client: {e}")
            finally:
                self._embedding_client = None

    async def force_close(self):
        """Forza la chiusura del client"""
        async with self._client_lock:
            await self._close_client()
            self._ref_count = 0


class CachedAsyncEmbeddingFactory:
    """Versione ottimizzata della tua AsyncEmbeddingFactory con gestione sessioni"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("CachedAsyncEmbeddingFactory initialized")

    def _get_cache_key(self, config: Dict[str, Any]) -> Tuple:
        """Usa la stessa logica della tua factory originale"""
        key_parts = []
        key_parts.append(config.get("provider"))
        key_parts.append(config.get("model_name"))

        api_key = config.get("api_key")
        if api_key:
            key_parts.append(str(api_key))

        if config.get("provider") == "huggingface":
            key_parts.append(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            key_parts.append(config.get("normalize", True))

        if config.get("base_url"):
            key_parts.append(config.get("base_url"))

        return tuple(key_parts)

    async def create(self, config: Dict[str, Any]) -> Tuple[Embeddings, int]:
        """
        Metodo principale ottimizzato per creare embeddings con sessioni persistenti
        """
        try:
            cache_key = self._get_cache_key(config)

            async def _cached_creator() -> Tuple[Embeddings, int]:
                self.logger.info(f"Creating new cached embedding for {config.get('provider', 'unknown')}")

                # Ottieni il session manager
                session_manager = await EmbeddingSessionManager.get_instance(config)

                # Ottieni il client con sessione persistente
                embedding_client = await session_manager.get_embedding_client()

                # Calcola dimensione
                dimension = self._get_dimension_for_config(config)

                # Wrappa il client in un oggetto che gestisce il session manager
                wrapped_client = EmbeddingWrapper(embedding_client, session_manager, dimension)

                return wrapped_client, dimension

            # Usa la cache con TimedCache
            embedding_obj, dimension = await TimedCache.async_get(
                object_type="embedding",
                key=cache_key,
                constructor=_cached_creator
            )

            return embedding_obj, dimension

        except Exception as e:
            self.logger.error(f"Error creating cached embedding: {str(e)}", exc_info=True)
            raise EmbeddingCreationError("Errore nella creazione degli embedding cached", e)

    def _get_dimension_for_config(self, config: Dict[str, Any]) -> int:
        """Calcola la dimensione basandosi sulla configurazione"""
        # Se specificata esplicitamente
        if config.get("dimension"):
            return config["dimension"]

        # Dimensioni per provider/modello
        provider = config.get("provider", "").lower()
        model_name = config.get("model_name", "")

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

        return 768  # Default fallback


class EmbeddingWrapper(Embeddings):
    """Wrapper che mantiene il riferimento al session manager"""

    def __init__(self, embedding_client: Embeddings, session_manager: EmbeddingSessionManager, dimension: int):
        self.embedding_client = embedding_client
        self.session_manager = session_manager
        self.dimension = dimension
        self._last_use = time.time()

    async def aembed_documents(self, texts):
        """Wrapper per embed_documents asincrono"""
        self._last_use = time.time()
        try:
            if hasattr(self.embedding_client, 'aembed_documents'):
                return await self.embedding_client.aembed_documents(texts)
            else:
                # Fallback per client che non supportano async
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.embedding_client.embed_documents, texts)
        except Exception as e:
            logger.warning(f"Error in aembed_documents, attempting reconnection: {e}")
            # Ricrea il client e riprova
            await self.session_manager.force_close()
            self.embedding_client = await self.session_manager.get_embedding_client()

            if hasattr(self.embedding_client, 'aembed_documents'):
                return await self.embedding_client.aembed_documents(texts)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.embedding_client.embed_documents, texts)

    async def aembed_query(self, text):
        """Wrapper per embed_query asincrono"""
        self._last_use = time.time()
        try:
            if hasattr(self.embedding_client, 'aembed_query'):
                return await self.embedding_client.aembed_query(text)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.embedding_client.embed_query, text)
        except Exception as e:
            logger.warning(f"Error in aembed_query, attempting reconnection: {e}")
            # Ricrea il client e riprova
            await self.session_manager.force_close()
            self.embedding_client = await self.session_manager.get_embedding_client()

            if hasattr(self.embedding_client, 'aembed_query'):
                return await self.embedding_client.aembed_query(text)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.embedding_client.embed_query, text)

    def embed_documents(self, texts):
        """Metodo sincrono"""
        self._last_use = time.time()
        return self.embedding_client.embed_documents(texts)

    def embed_query(self, text):
        """Metodo sincrono"""
        self._last_use = time.time()
        return self.embedding_client.embed_query(text)

    async def close(self):
        """Rilascia il session manager"""
        await self.session_manager.release_client()

    def is_healthy(self) -> bool:
        """Check se l'embedding è ancora valido"""
        return time.time() - self._last_use < 300  # 5 minuti


class EmbeddingCreationError(Exception):
    """Eccezione per errori nella creazione degli embedding"""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(f"{message} - Original error: {str(original_error)}" if original_error else message)


# Decoratori ottimizzati che usano la nuova factory
def inject_embedding_async_optimized(factory: Optional[CachedAsyncEmbeddingFactory] = None):
    """Decoratore ottimizzato per l'iniezione degli embedding"""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, item, *args, **kwargs):
            nonlocal factory
            factory = factory or CachedAsyncEmbeddingFactory()

            try:
                # Usa la stessa logica della tua factory originale
                if isinstance(item.embedding, str):
                    config = {
                        "legacy_mode": True,
                        "provider": "openai",  # Default per legacy
                        "model_name": item.embedding,
                        "api_key": item.gptkey
                    }
                elif isinstance(item.embedding, LlmEmbeddingModel):
                    config = {
                        "provider": item.embedding.provider,
                        "model_name": item.embedding.name,
                        "api_key": item.embedding.api_key,
                        "dimension": item.embedding.dimension,
                        "base_url": item.embedding.url
                    }
                else:
                    raise TypeError(f"Tipo non supportato per embedding: {type(item.embedding)}")

                # Crea embeddings con sessioni ottimizzate
                embedding_obj, dimension = await factory.create(config)

                kwargs["embedding_obj"] = embedding_obj
                kwargs["embedding_dimension"] = dimension

                return await func(self, item, *args, **kwargs)

            except ValidationError as ve:
                raise EmbeddingCreationError("Validazione fallita", ve)
            except Exception as e:
                raise EmbeddingCreationError("Errore generico nella creazione embedding", e)

        return wrapper

    return decorator


def inject_embedding_qa_async_optimized(factory: Optional[CachedAsyncEmbeddingFactory] = None):
    """Decoratore ottimizzato per QA"""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, question, *args, **kwargs):
            nonlocal factory
            factory = factory or CachedAsyncEmbeddingFactory()

            try:
                if isinstance(question.embedding, EmbeddingModel):
                    config = {
                        "provider": question.embedding.embedding_provider,
                        "model_name": question.embedding.embedding_model,
                        "api_key": question.embedding.embedding_key,
                        "dimension": question.embedding.embedding_dimension,
                        "base_url": question.embedding.embedding_host
                    }
                else:
                    config = {
                        "legacy_mode": True,
                        "provider": "openai",
                        "model_name": question.embedding,
                        "api_key": question.gptkey
                    }

                embedding_obj, dimension = await factory.create(config)

                kwargs["embedding_obj"] = embedding_obj
                kwargs["embedding_dimension"] = dimension

                return await func(self, question, *args, **kwargs)

            except ValidationError as ve:
                raise EmbeddingCreationError("Validazione fallita", ve)
            except Exception as e:
                raise EmbeddingCreationError("Errore generico", e)

        return wrapper

    return decorator


# Esempio di utilizzo integrato
async def create_optimized_embedding_instance(question):
    """Funzione helper per creare embeddings ottimizzati"""
    factory = CachedAsyncEmbeddingFactory()

    if hasattr(question, 'embedding') and isinstance(question.embedding, EmbeddingModel):
        config = {
            "provider": question.embedding.embedding_provider,
            "model_name": question.embedding.embedding_model,
            "api_key": question.embedding.embedding_key,
            "dimension": question.embedding.embedding_dimension,
            "base_url": question.embedding.embedding_host
        }
    else:
        config = {
            "legacy_mode": True,
            "provider": "openai",
            "model_name": getattr(question, 'embedding', 'text-embedding-ada-002'),
            "api_key": getattr(question, 'gptkey', None)
        }

    return await factory.create(config)