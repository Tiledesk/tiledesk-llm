import asyncio
import json
import os

from functools import wraps
import hashlib

import logging
from typing import Dict, Any, Callable, Tuple, Optional

from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from tilellm.models import LlmEmbeddingModel  # EmbeddingModel



from tilellm.shared.embedding_factory import EmbeddingFactory, AsyncEmbeddingFactory
from tilellm.shared.tiledesk_chatmodel_info import TiledeskAICallbackHandler
from tilellm.shared.timed_cache import TimedCache
from tilellm.shared.llm_config import get_llm_params

logger = logging.getLogger(__name__)

def _str_to_bool(value: str) -> bool:
    """
    Convert a string to boolean.
    Accepts: 'true', '1', 'yes', 'on' (case-insensitive) -> True
    All other values -> False
    """
    return str(value).lower() in ('true', '1', 'yes', 'on')


def get_service_config():
    """
    Loads service configuration from environment variables.
    Returns:
        A dictionary with the service configuration.
    """
    config = {}

    # Profile-based service configuration
    tilellm_profile = os.environ.get("TILELLM_PROFILE", "").lower()

    # Define service profiles
    service_profiles = {
        "app-base": {
            "task_executor": True,
            "graphrag": False,
            "graphrag_falkor": False,
            "pdf_ocr": False,
            "conversion": True,
            "tools_registry": True
        },
        "app-graph": {
            "task_executor": True,
            "graphrag": False,
            "graphrag_falkor": True,
            "pdf_ocr": False,
            "conversion": True,
            "tools_registry": True
        },
        "app-ocr": {
            "task_executor": True,
            "graphrag": False,
            "graphrag_falkor": True,
            "pdf_ocr": True,
            "conversion": True,
            "tools_registry": True
        },
        "app-all": {
            "task_executor": True,
            "graphrag": False,
            "graphrag_falkor": True,
            "pdf_ocr": True,
            "conversion": True,
            "tools_registry": True
        }
    }

    if tilellm_profile in service_profiles:
        config["services"] = service_profiles[tilellm_profile]
        logger.info(f"Using TILELLM_PROFILE '{tilellm_profile}': {config['services']}")
    else:
        config["services"] = {
            "task_executor": _str_to_bool(os.environ.get("ENABLE_TASKIQ", "true")),
            "graphrag": _str_to_bool(os.environ.get("ENABLE_GRAPHRAG", "false")), # Disable Neo4j by default
            "graphrag_falkor": _str_to_bool(os.environ.get("ENABLE_GRAPHRAG_FALKOR", "false")),
            "pdf_ocr": _str_to_bool(os.environ.get("ENABLE_PDF_OCR", "false")),
            "conversion": _str_to_bool(os.environ.get("ENABLE_CONVERSION", "true")),
            "tools_registry": _str_to_bool(os.environ.get("ENABLE_TOOLS_REGISTRY", "true")),
            "api_v2": _str_to_bool(os.environ.get("ENABLE_API_V2", "true"))
        }

    # TEI Configuration
    config["tei"] = {
        "embedding": {
            "url": os.environ.get("TEI_EMBEDDING_URL", "http://localhost:7580"),
            "api_key": os.environ.get("TEI_EMBEDDING_API_KEY", ""),
            "model": os.environ.get("TEI_EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
        },
        "sparse_encoder": {
            "url": os.environ.get("TEI_SPARSE_ENCODER_URL", "http://localhost:7380"),
            "api_key": os.environ.get("TEI_SPARSE_ENCODER_API_KEY", ""),
            "model": os.environ.get("TEI_SPARSE_ENCODER_MODEL", "naver/efficient-splade-VI-BT-large-query")
        },
        "reranker": {
            "url": os.environ.get("TEI_RERANKER_URL", "http://localhost:7480"),
            "api_key": os.environ.get("TEI_RERANKER_API_KEY", ""),
            "model": os.environ.get("TEI_RERANKER_MODEL", "BAAI/bge-reranker-large")
        }
    }

    # MinIO Configuration
    config["minio"] = {
        "endpoint": os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        "access_key": os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        "secret_key": os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        "secure": _str_to_bool(os.environ.get("MINIO_SECURE", "False")),
        "bucket_name": os.environ.get("MINIO_GRAPHRAG_BUCKET", "graphrag"),
        "bucket_tables": os.environ.get("MINIO_TABLES_BUCKET", "document-tables"),
        "bucket_images": os.environ.get("MINIO_IMAGES_BUCKET", "document-images"),
        "bucket_pdfs": os.environ.get("MINIO_PDFS_BUCKET", "ocr-pdfs")
    }

    # Neo4j Configuration
    config["neo4j"] = {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USERNAME", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "password"),
        "database": os.environ.get("NEO4J_DATABASE", "neo4j")
    }

    # FalkorDB Configuration
    falkordb_uri = os.environ.get("FALKORDB_URI", "redis://localhost:6380/0")
    # Parse URI for host, port, db, username, password
    original_uri = falkordb_uri
    
    # Helper to clean empty strings to None
    def _clean_auth(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value if value else None
    
    try:
        # Remove redis://
        uri_to_parse = falkordb_uri
        if uri_to_parse.startswith("redis://"):
            uri_to_parse = uri_to_parse[8:]
        # Split off auth if present
        if "@" in uri_to_parse:
            auth_part, host_part = uri_to_parse.split("@", 1)
            if ":" in auth_part:
                falkordb_username, falkordb_password = auth_part.split(":", 1)
            else:
                falkordb_username = auth_part
                falkordb_password = None
        else:
            host_part = uri_to_parse
            falkordb_username = None
            falkordb_password = None
        
        # Clean username and password (empty strings -> None)
        falkordb_username = _clean_auth(falkordb_username)
        falkordb_password = _clean_auth(falkordb_password)
        
        # Split host:port/db
        if ":" in host_part:
            host_port_part, *db_part = host_part.split("/", 1)
            host, port = host_port_part.split(":", 1)
            port = int(port)
            db = int(db_part[0]) if db_part else 0
        else:
            host = host_part.split("/")[0]
            port = 6379
            db = int(host_part.split("/")[1]) if "/" in host_part else 0
    except Exception:
        host = "localhost"
        port = 6379
        db = 0
        falkordb_username = None
        falkordb_password = None
    
    config["falkordb"] = {
        "uri": original_uri,
        "host": host,
        "port": port,
        "db": db,
        "username": falkordb_username,
        "password": falkordb_password,
        "max_connections": int(os.environ.get("FALKORDB_MAX_CONNECTIONS", "50")),
        "socket_timeout": float(os.environ.get("FALKORDB_SOCKET_TIMEOUT", "30.0")),
        "socket_connect_timeout": float(os.environ.get("FALKORDB_SOCKET_CONNECT_TIMEOUT", "10.0")),
        "socket_keepalive": _str_to_bool(os.environ.get("FALKORDB_SOCKET_KEEPALIVE", "True")),
        "retry_on_timeout": _str_to_bool(os.environ.get("FALKORDB_RETRY_ON_TIMEOUT", "True"))
    }

    # Redis Configuration
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        parsed = redis_url.replace("redis://", "").split(":")
        redis_host = parsed[0].split("/")[0] if "/" in parsed[0] else parsed[0]
        redis_port = int(parsed[1].split("/")[0]) if len(parsed) > 1 else 6379
        redis_db = int(parsed[2]) if len(parsed) > 2 else 0
    except Exception:
        redis_host = "localhost"
        redis_port = 6379
        redis_db = 0

    config["redis"] = {
        "host": redis_host,
        "port": redis_port,
        "db": redis_db,
        "queue_name": os.environ.get("REDIS_QUEUE_NAME", "tiledesk_ocr_queue")
    }

    # JWT Configuration
    config["jwt"] = {
        "secret_key": os.environ.get("JWT_SECRET_KEY")
    }

    return config


def _hash_api_key(api_key: str) -> str:
    """
    Crea un hash SHA256 della chiave API per utilizzarlo nella cache
    senza esporre la chiave completa nei log.
    """
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()#[:16]

async def _get_llm_config_for_client(question, llm_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to consolidate LLM client configuration based on the question object.
    Returns a dictionary of parameters suitable for various LangChain Chat models.
    """
    api_key_param = None
    custom_headers_to_use = None
    model_name_param = None
    base_url_param = None
    #provider_param = None
    
    # Determine API key, model name, base URL, and custom headers
    if isinstance(question.model, LlmEmbeddingModel):
        api_key_param = question.model.api_key
        custom_headers_to_use = question.model.custom_headers
        model_name_param = question.model.name
        base_url_param = question.model.url
        #provider_param = question.model.provider
    else:
        # Fallback for when question.model is a string or other object
        # The key can be llm_key or gptkey depending on the decorator
        if hasattr(question, 'llm_key'):
            api_key_param = question.llm_key
        elif hasattr(question, 'gptkey'):
            api_key_param = question.gptkey
            
        model_name_param = question.model if isinstance(question.model, str) else (question.model.name if hasattr(question.model, 'name') else None)
        base_url_param = None if isinstance(question.model, str) else (question.model.url if hasattr(question.model, 'url') else None)
        #provider_param = question.llm if isinstance(question.llm, str) else (question.model.provider if hasattr(question.model, 'provider') else None)

    # Default vLLM base_url if not specified
    if question.llm == "vllm" and not base_url_param:
        base_url_param = "http://localhost:8001"

    # Consolidate all parameters for the client
    client_config = {
        #"provider" : provider_param,
        "api_key": api_key_param,
        "model": model_name_param,
        "base_url": base_url_param,
        "default_headers": custom_headers_to_use,
        **llm_params # Include generic LLM parameters
    }
    
    # Filter out None values for default_headers if it's None to avoid passing default_headers=None explicitly
    # Some LangChain clients might not like default_headers=None if they expect Dict[str, str]
    if client_config.get("default_headers") is None:
        del client_config["default_headers"]
    
    return client_config

class LLMInjectionError(Exception):
    """Eccezione personalizzata per errori di injection LLM"""
    pass


ADA_AND_3_MODELS = {
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large"
}


class LocalEmbeddingModelCache:
    """
    Cache specifica per modelli HuggingFace.
    Ottimizzata per evitare ricaricamento ripetuto dello stesso modello.
    """
    _cache: Dict[str, Any] = {}
    _logger = logging.getLogger("LocalEmbeddingModelCache")

    @classmethod
    def get_model(
            cls,
            model_name: str,
            normalize_embeddings: bool = True
    ) -> Any:
        import torch
        # Genera chiave univoca basata su nome modello e normalizzazione
        cache_key = f"hf_{model_name}_{normalize_embeddings}"

        if cache_key in cls._cache:
            cls._logger.info(f"Modello HuggingFace '{model_name}' già in cache. Riutilizzo.")
            return cls._cache[cache_key]

        cls._logger.info(f"Caricamento modello HuggingFace '{model_name}'...")
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_obj = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
        )

        cls._cache[cache_key] = model_obj
        return model_obj

    @classmethod
    def clear_cache(cls):
        """Svuota la cache e libera memoria GPU"""
        import torch
        cls._logger.warning("Svuotamento cache modelli HuggingFace...")
        cls._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def inject_repo(func):
    """
    Annotation for injecting the correct Vector Store Repository (Pinecone or Qdrant).
    If question.engine.name is 'pinecone', it injects PineconeRepository (Pod or Serverless).
    If question.engine.name is 'qdrant', it injects QdrantRepository.
    :param func: The function to wrap.
    :return: The wrapped function.
    """

    @wraps(func)
    def wrapper(question, *args, **kwargs):
        engine_name = question.engine.name
        repo = None # Inizializza repo a None

        logger.info(f"Engine name: {engine_name}")

        repo_type = question.engine.type if engine_name == 'pinecone' else None
        # Costruisci chiave cache
        cache_key_parts = [engine_name]
        if repo_type:
            cache_key_parts.append(repo_type)

        if hasattr(question.engine, 'host') and question.engine.host:
            cache_key_parts.append(question.engine.host)
        elif hasattr(question.engine, 'endpoint') and question.engine.endpoint:
            cache_key_parts.append(question.engine.endpoint)

        cache_key = tuple(cache_key_parts)

        def _creator():
            logger.info(f"Creazione nuovo oggetto Repository: {cache_key}")
            if engine_name == 'pinecone':
                # repo_type = question.engine.type
                logger.info(f"Pinecone type: {repo_type}")

                if repo_type == 'pod':
                    from tilellm.store.pinecone.pinecone_repository_pod import PineconeRepositoryPod
                    return PineconeRepositoryPod()
                elif repo_type == 'serverless':
                    from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
                    return PineconeRepositoryServerless()
                else:
                    raise ValueError(f"Unknown Pinecone repository type: {repo_type}")
            elif engine_name == 'qdrant':
                logger.info("Injecting QdrantRepository")
                from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository
                return QdrantRepository()
            elif engine_name == 'milvus':
                logger.info("Injecting MilvusRepository")
                from tilellm.store.milvus.milvus_repository import MilvusRepository
                return MilvusRepository()
            else:
                raise ValueError(f"Unknown engine name: {engine_name}")

        # 3. Ottieni il repository dalla cache
        repo = TimedCache.get(object_type="repository",
                              key=cache_key,
                              constructor=_creator
                              )


        kwargs['repo'] = repo
        return func(question, *args, **kwargs)

    return wrapper


def inject_repo_async(func: Callable) -> Callable:
    """
    Versione async del decoratore inject_repo.
    Utilizza TimedCache.async_get per gestire repository in contesti async.

    :param func: The async function to wrap.
    :return: The wrapped async function.
    """

    @wraps(func)
    async def async_wrapper(question, *args, **kwargs):
        engine_name = question.engine.name
        repo = None
        logger.info(f"Engine name: {engine_name}")

        repo_type = question.engine.type if engine_name == 'pinecone' else None

        cache_key_tuple = await _build_repository_cache_key(question)  # type: ignore

        cache_key = cache_key_tuple

        async def _async_creator() -> Any:
            """Factory function async per creare nuove istanze del repository"""
            logger.info(f"Creazione async nuovo oggetto Repository: {cache_key}")

            try:
                if engine_name == 'pinecone':
                    logger.info(f"Pinecone type: {repo_type}")
                    if repo_type == 'pod':
                        from tilellm.store.pinecone.pinecone_repository_pod import PineconeRepositoryPod
                        repo_instance = PineconeRepositoryPod()
                        logger.info(f"Creato async PineconeRepositoryPod con chiave: {cache_key}")
                        return repo_instance
                    elif repo_type == 'serverless':
                        from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
                        repo_instance = PineconeRepositoryServerless()
                        logger.info(f"Creato async PineconeRepositoryServerless con chiave: {cache_key}")
                        return repo_instance
                    else:
                        raise ValueError(f"Unknown Pinecone repository type: {repo_type}")

                elif engine_name == 'qdrant':
                    logger.info("Injecting QdrantRepository (async)")
                    from tilellm.store.qdrant.qdrant_repository_local import QdrantRepository
                    repo_instance = QdrantRepository()
                    logger.info(f"Create async QdrantRepository with key: {cache_key}")
                    return repo_instance

                elif engine_name == 'milvus':
                    logger.info("Injecting MilvusRepository (async)")
                    from tilellm.store.milvus.milvus_repository import MilvusRepository
                    repo_instance = MilvusRepository()
                    logger.info(f"Create async MilvusRepository with key: {cache_key}")
                    return repo_instance

                else:
                    raise ValueError(f"Unknown engine name: {engine_name}")

            except Exception as e:
                logger.error(f"Error in the asynchronous creation of the repository {cache_key}: {e}")
                raise

        try:
            # Ottieni il repository dalla cache (versione async)
            repo = await TimedCache.async_get(  # type: ignore
                object_type="repository",
                key=cache_key,
                constructor=_async_creator
            )

            if repo is None:
                logger.error(f"Repository None obtained from async cache for key: {cache_key}")
                raise RuntimeError(f"Failed to get repository for key: {cache_key}")

            logger.debug(f"Repository async {type(repo).__name__} get/create with key: {cache_key}")

            kwargs['repo'] = repo

            # Esegui la funzione async originale
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Error in decorator async inject_repo_async for {func.__name__}: {e}")
            raise

    return async_wrapper


def inject_llm(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        try:
            logger.debug(question)
            
            # The cache key now needs to include custom_headers if present to ensure uniqueness
            # The _get_llm_config_for_client will return the actual headers to use
            # But for the cache key, we need to hash the headers or at least their presence
            
            # Temporarily call helper to get config for cache key, discard direct config values
            # The _creator will re-evaluate for the actual instance.
            llm_params_for_cache_key = get_llm_params(
                provider=question.llm,
                temperature=question.temperature,
                top_p=question.top_p,
                max_tokens=question.max_tokens
            )
            temp_client_base_config = asyncio.run(_get_llm_config_for_client(question, llm_params_for_cache_key)) # Use asyncio.run for sync context
            
            cache_key_parts = [
                question.llm,
                temp_client_base_config.get("model") if temp_client_base_config.get("model") else (question.model if isinstance(question.model, str) else question.model.name if hasattr(question.model, 'name') else None),
                _hash_api_key(str(temp_client_base_config.get("api_key").get_secret_value())) if temp_client_base_config.get("api_key") else "no_key"  # type: ignore  # type: ignore
            ]
            if temp_client_base_config.get("base_url"):
                cache_key_parts.append(temp_client_base_config.get("base_url"))
            if temp_client_base_config.get("default_headers"):
                # Hash of headers to ensure cache key uniqueness
                headers_hash = hashlib.sha256(json.dumps(temp_client_base_config["default_headers"], sort_keys=True).encode('utf-8')).hexdigest()
                cache_key_parts.append(headers_hash)
            
            cache_key = tuple(cache_key_parts)

            def _creator():
                logger.info(f"Creazione nuovo oggetto Chat in cache con chiave: {cache_key}")
                
                # Re-evaluate llm_params and client config within _creator
                inner_llm_params = get_llm_params(
                    provider=question.llm,
                    temperature=question.temperature,
                    top_p=question.top_p,
                    max_tokens=question.max_tokens
                )
                inner_client_base_config = asyncio.run(_get_llm_config_for_client(question, inner_llm_params)) # Call async helper in sync context

                inner_client_config = {**inner_llm_params}
                if inner_client_base_config.get("api_key"):
                    inner_client_config["api_key"] = inner_client_base_config["api_key"]
                if inner_client_base_config.get("model"):
                    inner_client_config["model"] = inner_client_base_config["model"]
                if inner_client_base_config.get("base_url"):
                    inner_client_config["base_url"] = inner_client_base_config["base_url"]
                if inner_client_base_config.get("default_headers"):
                    inner_client_config["default_headers"] = inner_client_base_config["default_headers"]

                if question.llm == "openai" or question.llm == "vllm":
                    from langchain_openai import ChatOpenAI
                    return ChatOpenAI(**inner_client_config)

                elif question.llm == "anthropic":
                    from langchain_anthropic import ChatAnthropic
                    inner_client_config["anthropic_api_key"] = inner_client_config.pop("api_key", None)
                    return ChatAnthropic(**inner_client_config)

                elif question.llm == "cohere":
                    from langchain_cohere import ChatCohere
                    inner_client_config["cohere_api_key"] = inner_client_config.pop("api_key", None)
                    return ChatCohere(**inner_client_config)

                elif question.llm == "google":
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    inner_client_config["google_api_key"] = inner_client_config.pop("api_key", None)
                    return ChatGoogleGenerativeAI(**inner_client_config)

                elif question.llm == "ollama":
                    from langchain_community.chat_models import ChatOllama
                    inner_client_config["num_predict"] = inner_client_config.pop("max_tokens", None)
                    return ChatOllama(**inner_client_config)

                elif question.llm == "groq":
                    from langchain_groq import ChatGroq
                    return ChatGroq(**inner_client_config)

                elif question.llm == "deepseek":
                    from langchain_deepseek import ChatDeepSeek
                    return ChatDeepSeek(**inner_client_config)

                elif question.llm == "mistralai":
                    from langchain_mistralai import ChatMistralAI
                    inner_client_config["model_name"] = inner_client_config.pop("model", None)
                    return ChatMistralAI(**inner_client_config)

                else:
                    logger.warning(f"Unknown LLM provider '{question.llm}', falling back to OpenAI")
                    from langchain_openai import ChatOpenAI
                    return ChatOpenAI(**inner_client_config)

            chat_model = TimedCache.get(
                object_type="chat",
                key=cache_key,
                constructor=_creator
            )

            # Add chat_model agli kwargs
            kwargs['chat_model'] = chat_model

            # Chiama la funzione originale con i nuovi kwargs
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"LLM initialization error: {e}", exc_info=True)
            raise

    return wrapper


def inject_llm_async(func: Callable) -> Callable:
    """
    Decorator asincrono per iniettare un LLM standard utilizzando TimedCache.
    Ottimizzato per riutilizzare istanze LLM con le stesse configurazioni.

    Args:
        func: Funzione da decorare che riceverà 'chat_model' negli kwargs
    """

    @wraps(func)
    async def async_wrapper(question, *args, **kwargs):
        try:
            logger.debug(f"Processing LLM injection for {func.__name__}")

            # Costruisce la chiave di cache univoca per il modello LLM
            cache_key = await _build_standard_llm_cache_key(question)

            async def _llm_creator():
                """Factory asincrona per creare nuove istanze LLM standard"""
                logger.info(f"Creazione nuovo oggetto LLM standard con chiave: {cache_key}")
                return await _create_standard_llm_instance(question)

            # Ottieni l'LLM dalla cache (versione asincrona)
            chat_model = await TimedCache.async_get(
                object_type="chat",
                key=cache_key,
                constructor=_llm_creator
            )

            if chat_model is None:
                logger.error(f"LLM None ottenuto dalla cache per chiave: {cache_key}")
                raise LLMInjectionError(f"Failed to get LLM for key: {cache_key}")

            logger.debug(f"LLM {type(chat_model).__name__} iniettato con successo")

            # Inietta il chat_model negli kwargs
            kwargs['chat_model'] = chat_model

            # Esegui la funzione originale
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Errore nel decorator inject_llm_async per {func.__name__}: {e}", exc_info=True)
            raise LLMInjectionError(f"LLM injection failed: {str(e)}")

    return async_wrapper


async def _build_standard_llm_cache_key(question) -> Tuple:
    """Costruisce la chiave di cache per il modello LLM standard"""
    cache_key_parts = [
        "standard",  # Distingue dai reasoning LLM
        question.llm,
        question.model if isinstance(question.model, str) else question.model.name,
        _hash_api_key(str(question.llm_key.get_secret_value())),  # Hash della chiave per sicurezza
    ]

    # Aggiungi URL per modelli self-hosted
    if question.llm in ["vllm", "ollama"] and hasattr(question.model, 'url'):
        cache_key_parts.append(question.model.url)  # type: ignore

    return tuple(cache_key_parts)


def inject_llm_chat(func):
    """
    Decorator che inietta modelli LLM e Embedding usando la TimedCache.
    I CallbackHandler sono sempre nuovi per ogni richiesta per garantire il corretto tracciamento.
    """

    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
        try:
            # --- 1. Gestione Cache per il Modello LLM (Chat Model) ---

            # Chiave di cache univoca per il modello LLM. Include tutti i parametri che lo definiscono.
            # Temporarily call helper to get config for cache key, discard direct config values
            # The _llm_creator will re-evaluate for the actual instance.
            llm_params_for_cache_key = get_llm_params(
                provider=question.llm,
                temperature=question.temperature,
                top_p=question.top_p,
                max_tokens=question.max_tokens
            )
            temp_client_base_config = asyncio.run(_get_llm_config_for_client(question, llm_params_for_cache_key)) # Use asyncio.run for sync context
            
            llm_cache_key_parts = [
                question.llm,
                temp_client_base_config.get("model") if temp_client_base_config.get("model") else (question.model if isinstance(question.model, str) else (question.model.name if hasattr(question.model, 'name') else None)),
                _hash_api_key(str(temp_client_base_config.get("api_key").get_secret_value())) if temp_client_base_config.get("api_key") else "no_key"  # type: ignore
            ]
            if temp_client_base_config.get("base_url"):
                llm_cache_key_parts.append(temp_client_base_config.get("base_url"))
            if temp_client_base_config.get("default_headers"):
                # Hash of headers to ensure cache key uniqueness
                headers_hash = hashlib.sha256(json.dumps(temp_client_base_config["default_headers"], sort_keys=True).encode('utf-8')).hexdigest()
                llm_cache_key_parts.append(headers_hash)
            
            llm_cache_key = tuple(llm_cache_key_parts)


            # Costruttore per il modello LLM (eseguito solo se l'oggetto non è in cache)
            def _llm_creator():
                logger.info(f"Creazione nuovo oggetto Chat in cache con chiave: {llm_cache_key}")
                
                # Ottieni parametri filtrati per il provider specifico
                inner_llm_params = get_llm_params(
                    provider=question.llm,
                    temperature=question.temperature,
                    top_p=question.top_p,
                    max_tokens=question.max_tokens
                )
                inner_client_base_config = asyncio.run(_get_llm_config_for_client(question, inner_llm_params))

                inner_client_config = {**inner_llm_params}
                if inner_client_base_config.get("api_key"):
                    inner_client_config["api_key"] = inner_client_base_config["api_key"]
                if inner_client_base_config.get("model"):
                    inner_client_config["model"] = inner_client_base_config["model"]
                if inner_client_base_config.get("base_url"):
                    inner_client_config["base_url"] = inner_client_base_config["base_url"]
                if inner_client_base_config.get("default_headers"):
                    inner_client_config["default_headers"] = inner_client_base_config["default_headers"]


                if question.llm == "openai" or question.llm == "vllm":
                    from langchain_openai import ChatOpenAI
                    return ChatOpenAI(**inner_client_config)

                elif question.llm == "anthropic":
                    from langchain_anthropic import ChatAnthropic
                    inner_client_config["anthropic_api_key"] = inner_client_config.pop("api_key", None)
                    return ChatAnthropic(**inner_client_config)

                elif question.llm == "cohere":
                    from langchain_cohere import ChatCohere
                    inner_client_config["cohere_api_key"] = inner_client_config.pop("api_key", None)
                    return ChatCohere(**inner_client_config)

                elif question.llm == "google":
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    inner_client_config["google_api_key"] = inner_client_config.pop("api_key", None)
                    return ChatGoogleGenerativeAI(**inner_client_config)

                elif question.llm == "mistralai":
                    from langchain_mistralai import ChatMistralAI
                    inner_client_config["model_name"] = inner_client_config.pop("model", None)
                    return ChatMistralAI(**inner_client_config)

                elif question.llm == "groq":
                    from langchain_groq import ChatGroq
                    return ChatGroq(**inner_client_config)

                elif question.llm == "deepseek":
                    from langchain_deepseek import ChatDeepSeek
                    return ChatDeepSeek(**inner_client_config)

                elif question.llm == "ollama":
                    from langchain_community.chat_models import ChatOllama
                    inner_client_config["num_predict"] = inner_client_config.pop("max_tokens", None)
                    return ChatOllama(**inner_client_config)

                else:  # Fallback a OpenAI
                    from langchain_openai import ChatOpenAI
                    logger.warning(f"Unknown LLM provider '{question.llm}', falling back to OpenAI")
                    return ChatOpenAI(**inner_client_config)

            # Recupera o crea il modello LLM dalla cache
            llm = TimedCache.get(object_type="chat", key=llm_cache_key, constructor=_llm_creator)

            # --- 2. Gestione Cache per il Modello di Embedding ---

            # Usa la tua EmbeddingFactory già refattorizzata (soluzione più pulita) o la logica qui sotto.
            # Qui replichiamo la logica per chiarezza.
            embedding_config = {}
            if isinstance(question.embedding, LlmEmbeddingModel):#EmbeddingModel):
                embedding_config = {
                    "provider": question.embedding.provider,
                    "model_name": question.embedding.name,
                    "api_key": question.embedding.api_key,
                    "base_url": question.embedding.url
                }
            else:  # Modalità legacy con stringa
                embedding_config = {
                    "provider": question.embedding,  # es. "openai" o "huggingface"
                    "model_name": question.embedding,
                    "api_key": question.gptkey,
                    "legacy_mode": True
                }


            embedding_cache_key = tuple(sorted(embedding_config.items()))

            def _embedding_creator():
                logger.info(f"Creazione nuovo oggetto Embedding in cache con chiave: {embedding_cache_key}")
                # ... Inserire qui la logica di creazione degli embedding che era nel decorator originale ...
                factory = EmbeddingFactory()
                embedding_obj, _ = factory.create(embedding_config)
                return embedding_obj

            llm_embeddings = _embedding_creator()

            # --- 3. Callback Handler creation (Always New) ---
            callback_handler = None
            if question.debug:

                if question.llm == "openai":
                    callback_handler = OpenAICallbackHandler()
                else:
                    callback_handler = TiledeskAICallbackHandler()

                # Assegna il callback handler al modello LLM *prima* di passarlo
                # NOTA: il modo migliore è passarlo al momento della chiamata, es: llm.invoke(..., callbacks=[...])
                # Se la classe non lo permette, l'assegnazione diretta può funzionare.
                llm.callbacks = [callback_handler]

            # --- 4. Iniezione e Chiamata ---

            kwargs['llm'] = llm
            kwargs['callback_handler'] = callback_handler
            kwargs['llm_embeddings'] = llm_embeddings

            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Errore durante l'iniezione del LLM: {e}", exc_info=True)
            raise

    return wrapper


def inject_llm_chat_async(func: Callable) -> Callable:
    """
    Decoratore async che inietta modelli LLM e Embedding usando la TimedCache.
    I CallbackHandler sono sempre nuovi per ogni richiesta per garantire il corretto tracciamento.
    Ottimizzazione: gli embedding vengono creati/recuperati una sola volta dalla cache.
    """

    @wraps(func)
    async def async_wrapper(question, *args, **kwargs):
        logger.debug(f"Processing question with LLM: {question.llm}")

        try:
            # --- 1. Gestione Cache per il Modello LLM (Chat Model) ---
            llm_cache_key = await _build_llm_cache_key(question)

            async def _llm_creator():
                logger.debug(f"Creazione nuovo oggetto Chat in cache con chiave: {llm_cache_key}")
                return await _create_llm_instance(question)

            # Recupera o crea il modello LLM dalla cache (async)
            llm = await TimedCache.async_get(
                object_type="chat",
                key=llm_cache_key,
                constructor=_llm_creator
            )

            # --- 2. Gestione Cache per il Modello di Embedding (OTTIMIZZATA) ---
            embedding_cache_key = await _build_embedding_cache_key(question)

            async def _embedding_creator():
                logger.debug(f"Creazione nuovo oggetto Embedding in cache con chiave: {embedding_cache_key}")
                return await _create_embedding_instance(question)

            # Recupera o crea gli embedding dalla cache (async) - UNA SOLA VOLTA
            llm_embeddings = await TimedCache.async_get(
                object_type="embedding",
                key=embedding_cache_key,
                constructor=_embedding_creator
            )

            # --- 3. Callback Handler creation (Always New) ---
            callback_handler = _create_callback_handler(question, llm)

            # --- 4. Iniezione e Chiamata ---
            kwargs['llm'] = llm
            kwargs['callback_handler'] = callback_handler
            kwargs['llm_embeddings'] = llm_embeddings
            kwargs['embedding_config_key'] = embedding_cache_key

            logger.debug(f"LLM e Embedding iniettati con successo per {func.__name__}")
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Errore durante l'iniezione async del LLM per {func.__name__}: {e}", exc_info=True)
            raise

    return async_wrapper


async def _build_repository_cache_key(question: Any) -> tuple:
    """
    Costruisce la chiave di cache basata sui parametri specifici dell'engine,
    inclusa la versione hashata della chiave API.
    """
    # Assicurati che question.engine esista
    if not hasattr(question, 'engine') or not question.engine:
        # Restituisce una tupla chiara in caso di configurazione mancante
        return ("engine_config_missing",)

    engine_config = question.engine

    # Semplificazione: gestisce sia oggetti con attributi che dizionari
    def get_value(key: str, default: Any = None) -> Any:
        if isinstance(engine_config, dict):
            return engine_config.get(key, default)
        return getattr(engine_config, key, default)

    engine_name = get_value('name')

    # --- 1. Estrarre e Hashare l'API Key ---

    raw_api_key_object = get_value("api_key")

    hashed_api_key: str

    if raw_api_key_object:
        try:
            # Assumiamo Pydantic SecretStr: estrai il valore segreto
            api_key_str = raw_api_key_object.get_secret_value()
            hashed_api_key = _hash_api_key(api_key_str)
        except AttributeError:
            # Non è un SecretStr, trattalo come stringa o valore diretto
            api_key_str = str(raw_api_key_object)
            hashed_api_key = _hash_api_key(api_key_str)
    else:
        hashed_api_key = "no_key"

    # --- 2. Costruire le Parti della Tupla ---

    # Repo Type (solo per Pinecone)
    repo_type = get_value('type') if engine_name == 'pinecone' else None

    # Host/Endpoint (gestione fallback)
    host = get_value('host') or get_value('endpoint')

    # Index Name
    index_name = get_value('index_name')

    # --- 3. Composizione Finale della Tupla ---

    # La tupla può contenere diversi tipi, ma qui usiamo str/None per semplicità
    cache_key_parts = (
        engine_name or "unknown_engine",
        repo_type,  # 'pod' | 'serverless' | None
        hashed_api_key,  # L'elemento cruciale per l'univocità delle credenziali
        host,  # Host del servizio
        index_name,  # Nome dell'indice
        # Aggiungi qui altre parti se necessario (es. vector_size)
    )

    # Filtriamo gli elementi None per rendere la tupla più pulita e corta
    # e castiamo a stringa per consistenza se usiamo un hash finale
    final_key_tuple = tuple(str(p) for p in cache_key_parts if p is not None)

    return final_key_tuple

async def _build_llm_cache_key_old(question) -> tuple:
    """Costruisce la chiave di cache per il modello LLM"""

    cache_key_parts = [
        "chat",
        question.llm,
        question.model if isinstance(question.model, str) else question.model.name,
        _hash_api_key(str(question.gptkey.get_secret_value()))
    ]

    # Aggiungi URL per modelli self-hosted
    if question.llm in ["vllm", "ollama"] and hasattr(question.model, 'url'):
        cache_key_parts.append(question.model.url)  # type: ignore

    return tuple(cache_key_parts)

async def _build_llm_cache_key(question) -> tuple:
    """Costruisce la chiave di cache per il modello LLM"""

    cache_key_parts_dic = {}

    if isinstance(question.model, LlmEmbeddingModel):  # EmbeddingModel):
        cache_key_parts_dic = {
            "model_type": "chat_object",
            "provider": question.model.provider,
            "model_name": question.model.name,
            "api_key": _hash_api_key(str(question.model.api_key.get_secret_value())),  # Hash della chiave  # type: ignore
            #"base_url": question.model.url
        }
        if question.model.url is not None:
            cache_key_parts_dic["base_url"] = question.model.url

    else:  # Modalità legacy con stringa
        cache_key_parts_dic = {
            "model_type": "chat_legacy",
            "provider": question.llm,
            "model_name": question.model,
            "api_key": _hash_api_key(str(question.gptkey.get_secret_value())),
            "legacy_mode": True
        }

    sorted_params = tuple(
        (k, str(v)) for k, v in sorted(cache_key_parts_dic.items())
        if v is not None
    )

    # Chiave finale: prefisso + parametri

    cache_key =("chat",) + sorted_params
    logger.info(cache_key)


    #return tuple(cache_key_parts)
    return cache_key




async def _build_embedding_cache_key(question) -> tuple:
    """Costruisce la chiave di cache per gli embedding"""
    embedding_config = {}

    if isinstance(question.embedding, LlmEmbeddingModel):#EmbeddingModel):
        embedding_config = {
            "embedding_type": "object",
            "provider": question.embedding.provider,
            "model_name": question.embedding.name,
            "api_key": _hash_api_key(str(question.embedding.api_key.get_secret_value())),  # Hash della chiave  # type: ignore
            "base_url": question.embedding.url
        }
    else:  # Modalità legacy con stringa
        embedding_config = {
            "embedding_type": "legacy",
            "provider": question.embedding,
            "model_name": question.embedding,
            "api_key": _hash_api_key(str(question.gptkey.get_secret_value())),
            "legacy_mode": True
        }

    # Ordina per garantire consistenza della chiave
    return tuple(sorted(embedding_config.items()))


async def _create_llm_instance(question):
    """Crea una nuova istanza del modello LLM usando configurazione centralizzata"""

    try:

        provider_param = question.model.provider.value if hasattr(question.model, 'provider') else question.llm

        llm_params = get_llm_params(
            provider=provider_param,  # type: ignore
            temperature=question.temperature,
            top_p=question.top_p,
            max_tokens=question.max_tokens
        )

        client_base_config = await _get_llm_config_for_client(question, llm_params)

        client_config = {**llm_params}

        if client_base_config.get("api_key"):
            client_config["api_key"] = client_base_config["api_key"]
        if client_base_config.get("model"):
            client_config["model"] = client_base_config["model"]
        if client_base_config.get("base_url"):
            client_config["base_url"] = client_base_config["base_url"]
        if client_base_config.get("default_headers"):
            client_config["default_headers"] = client_base_config["default_headers"]


        if provider_param == "openai" or provider_param == "vllm":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(**client_config)

        elif provider_param == "anthropic":
            from langchain_anthropic import ChatAnthropic
            client_config["anthropic_api_key"] = client_config.pop("api_key", None)
            return ChatAnthropic(**client_config)

        elif provider_param == "cohere":
            from langchain_cohere import ChatCohere
            client_config["cohere_api_key"] = client_config.pop("api_key", None)
            return ChatCohere(**client_config)

        elif provider_param == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            client_config["google_api_key"] = client_config.pop("api_key", None)
            return ChatGoogleGenerativeAI(**client_config)

        elif provider_param == "mistralai":
            from langchain_mistralai import ChatMistralAI
            client_config["model_name"] = client_config.pop("model", None)
            return ChatMistralAI(**client_config)

        elif provider_param == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(**client_config)

        elif provider_param == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(**client_config)

        elif provider_param == "ollama":
            from langchain_community.chat_models import ChatOllama
            client_config["num_predict"] = client_config.pop("max_tokens", None)
            return ChatOllama(**client_config)

        else:  # Fallback a OpenAI
            from langchain_openai import ChatOpenAI
            logger.warning(f"Unknown LLM provider '{question.llm}', falling back to OpenAI")
            return ChatOpenAI(**client_config)

    except Exception as e:
        logger.error(f"Errore nella creazione del modello LLM {question.llm}: {e}", exc_info=True)
        raise

async def _create_standard_llm_instance(question) -> Any:
    """Crea una nuova istanza del modello LLM standard usando configurazione centralizzata"""
    try:
        llm_params = get_llm_params(
            provider=question.llm,
            temperature=question.temperature,
            top_p=question.top_p,
            max_tokens=question.max_tokens
        )
        
        # Use the centralized helper to get client configuration base
        client_base_config = await _get_llm_config_for_client(question, llm_params)
        
        # Start with llm_params and add base config (model is handled per provider)
        client_config = {**llm_params}
        if client_base_config.get("api_key"):
            client_config["api_key"] = client_base_config["api_key"]
        if client_base_config.get("model"):
            client_config["model"] = client_base_config["model"]
        if client_base_config.get("base_url"):
            client_config["base_url"] = client_base_config["base_url"]
        if client_base_config.get("default_headers"):
            client_config["default_headers"] = client_base_config["default_headers"]

        # Now, adapt client_config for each specific provider
        if question.llm == "openai" or question.llm == "vllm":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(**client_config)

        elif question.llm == "anthropic":
            from langchain_anthropic import ChatAnthropic
            client_config["anthropic_api_key"] = client_config.pop("api_key", None)
            return ChatAnthropic(**client_config)

        elif question.llm == "cohere":
            from langchain_cohere import ChatCohere
            client_config["cohere_api_key"] = client_config.pop("api_key", None)
            return ChatCohere(**client_config)

        elif question.llm == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            client_config["google_api_key"] = client_config.pop("api_key", None)
            return ChatGoogleGenerativeAI(**client_config)

        elif question.llm == "ollama":
            from langchain_community.chat_models import ChatOllama
            client_config["num_predict"] = client_config.pop("max_tokens", None)
            return ChatOllama(**client_config)

        elif question.llm == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(**client_config)

        elif question.llm == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(**client_config)

        elif question.llm == "mistralai":
            from langchain_mistralai import ChatMistralAI
            client_config["model_name"] = client_config.pop("model", None)
            return ChatMistralAI(**client_config)

        else:
            logger.warning(f"Unknown LLM provider '{question.llm}', falling back to OpenAI")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(**client_config)

    except Exception as e:
        logger.error(f"Errore nella creazione del modello LLM standard {question.llm}: {e}", exc_info=True)
        raise

async def _create_embedding_instance(question):
    """Crea una nuova istanza del modello di embedding utilizzando EmbeddingFactory"""
    try:
        # Prepara la configurazione per EmbeddingFactory
        if isinstance(question.embedding, LlmEmbeddingModel):#EmbeddingModel):
            embedding_config = {
                "provider": question.embedding.provider,
                "model_name": question.embedding.name,
                "api_key": question.embedding.api_key,
                "base_url": question.embedding.url
            }
        else:  # Modalità legacy
            embedding_config = {
                "provider": question.embedding,
                "model_name": question.embedding,
                "api_key": question.gptkey,
                "legacy_mode": True
            }

        # Usa EmbeddingFactory per creare l'istanza
        #factory = EmbeddingFactory()
        #embedding_obj, embedding_dimension = factory.create(embedding_config)
        factory = AsyncEmbeddingFactory()
        embedding_obj, embedding_dimension = await factory.create(embedding_config)

        # Potresti voler salvare anche embedding_dimension se necessario
        # Per ora restituiamo solo l'oggetto embedding
        return embedding_obj

    except Exception as e:
        logger.error(f"Errore nella creazione degli embedding: {e}")
        raise


def _create_callback_handler(question, llm):
    """Crea un nuovo callback handler (sempre nuovo per ogni richiesta)"""
    callback_handler = None

    if question.debug:
        try:
            if question.llm == "openai":
                from langchain_community.callbacks import OpenAICallbackHandler
                callback_handler = OpenAICallbackHandler()
            else:
                # Assumo che tu abbia questo callback personalizzato
                callback_handler = TiledeskAICallbackHandler()

            # Assegna il callback handler al modello LLM
            if callback_handler:
                llm.callbacks = [callback_handler]
                logger.debug(f"Callback handler {type(callback_handler).__name__} assegnato")

        except Exception as e:
            logger.warning(f"Errore nella creazione del callback handler: {e}")
            callback_handler = None

    return callback_handler



def inject_reason_llm(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        try:
            logger.debug(question)

            llm_params_for_cache_key = get_llm_params(
                provider=question.llm,
                temperature=question.temperature,
                top_p=question.top_p,
                max_tokens=question.max_tokens
            )
            temp_client_base_config = asyncio.run(_get_llm_config_for_client(question, llm_params_for_cache_key))
            
            cache_key_parts = [
                "reasoning", # Add type to distinguish from standard LLM
                question.llm,
                temp_client_base_config.get("model") if temp_client_base_config.get("model") else (question.model if isinstance(question.model, str) else (question.model.name if hasattr(question.model, 'name') else None)),
                _hash_api_key(str(temp_client_base_config.get("api_key").get_secret_value())) if temp_client_base_config.get("api_key") else "no_key"  # type: ignore
            ]
            if temp_client_base_config.get("base_url"):
                cache_key_parts.append(temp_client_base_config.get("base_url"))
            if temp_client_base_config.get("default_headers"):
                headers_hash = hashlib.sha256(json.dumps(temp_client_base_config["default_headers"], sort_keys=True).encode('utf-8')).hexdigest()
                cache_key_parts.append(headers_hash)
            
            cache_key = tuple(cache_key_parts)

            def _creator():
                logger.info(f"Creazione nuovo oggetto Chat (reasoning) in cache con chiave: {cache_key}")
                
                inner_llm_params = get_llm_params(
                    provider=question.llm,
                    temperature=question.temperature,
                    top_p=question.top_p,
                    max_tokens=question.max_tokens
                )
                inner_client_base_config = asyncio.run(_get_llm_config_for_client(question, inner_llm_params))

                client_config = {**inner_llm_params}
                if inner_client_base_config.get("api_key"):
                    client_config["api_key"] = inner_client_base_config["api_key"]
                if inner_client_base_config.get("model"):
                    client_config["model"] = inner_client_base_config["model"]
                if inner_client_base_config.get("base_url"):
                    client_config["base_url"] = inner_client_base_config["base_url"]
                if inner_client_base_config.get("default_headers"):
                    client_config["default_headers"] = inner_client_base_config["default_headers"]

                if question.llm == "openai":
                    from langchain_openai import ChatOpenAI
                    # Use max_completion_tokens for reasoning context
                    client_config["max_completion_tokens"] = client_config.pop("max_tokens", None)
                    return ChatOpenAI(**client_config)
                elif question.llm == "anthropic":
                    from langchain_anthropic import ChatAnthropic
                    client_config["anthropic_api_key"] = client_config.pop("api_key", None)
                    client_config["max_tokens"] = client_config.pop("max_tokens", None) # Anthropic uses max_tokens, not max_completion_tokens
                    if hasattr(question, 'thinking') and question.thinking is not None:
                        client_config["thinking"] = question.thinking
                    return ChatAnthropic(**client_config)
                elif question.llm == "deepseek":
                    from langchain_deepseek import ChatDeepSeek
                    # Deepseek uses max_tokens. Langchain DeepSeek model uses max_tokens not max_completion_tokens
                    return ChatDeepSeek(**client_config)
                elif question.llm == "vllm":
                    from langchain_openai import ChatOpenAI # vLLM uses OpenAI compatible API
                    client_config["max_completion_tokens"] = client_config.pop("max_tokens", None)
                    return ChatOpenAI(**client_config)
                else:
                    logger.warning(f"Unknown LLM provider '{question.llm}' for reasoning, falling back to OpenAI")
                    from langchain_openai import ChatOpenAI
                    client_config["max_completion_tokens"] = client_config.pop("max_tokens", None)
                    return ChatOpenAI(**client_config)

            chat_model = TimedCache.get(
                object_type="reasoning",
                key=cache_key,
                constructor=_creator
            )

            # Add chat_model agli kwargs
            kwargs['chat_model'] = chat_model

            # Chiama la funzione originale con i nuovi kwargs
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"LLM initialization error: {e}", exc_info=True)
            raise

    return wrapper


def inject_reason_llm_async(func: Callable) -> Callable:
    """
    Decorator asincrono per iniettare un LLM per reasoning utilizzando TimedCache.
    Supporta modelli specializzati per il reasoning come Claude con thinking mode.

    Args:
        func: Funzione da decorare che riceverà 'chat_model' negli kwargs
    """

    @wraps(func)
    async def async_wrapper(question, *args, **kwargs):
        try:
            logger.debug(f"Processing reasoning LLM injection for {func.__name__}")

            # Costruisce la chiave di cache univoca per il modello LLM di reasoning
            cache_key = await _build_reasoning_llm_cache_key(question)

            async def _reasoning_llm_creator():
                """Factory asincrona per creare nuove istanze LLM di reasoning"""
                logger.info(f"Creazione nuovo oggetto LLM reasoning con chiave: {cache_key}")
                return await _create_reasoning_llm_instance(question)

            # Ottieni l'LLM di reasoning dalla cache (versione asincrona)
            chat_model = await TimedCache.async_get(
                object_type="reasoning",
                key=cache_key,
                constructor=_reasoning_llm_creator
            )

            if chat_model is None:
                logger.error(f"Reasoning LLM None ottenuto dalla cache per chiave: {cache_key}")
                raise LLMInjectionError(f"Failed to get reasoning LLM for key: {cache_key}")

            logger.debug(f"Reasoning LLM {type(chat_model).__name__} iniettato con successo")

            # Inietta il chat_model negli kwargs
            kwargs['chat_model'] = chat_model

            # Esegui la funzione originale
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Errore nel decorator inject_reason_llm per {func.__name__}: {e}", exc_info=True)
            raise LLMInjectionError(f"Reasoning LLM injection failed: {e}")

    return async_wrapper


async def _build_reasoning_llm_cache_key(question) -> Tuple:
    """Costruisce la chiave di cache per il modello LLM di reasoning"""
    cache_key_parts = [
        "reasoning",  # Distingue dagli LLM standard
        question.llm,
        question.model if isinstance(question.model, str) else question.model.name,
        _hash_api_key(str(question.llm_key.get_secret_value())),
    ]

    # Aggiungi parametri specifici per reasoning
    if hasattr(question, 'thinking') and question.thinking is not None:
        cache_key_parts.append(f"thinking_{question.thinking}")

    # Aggiungi URL per modelli self-hosted se necessario
    if question.llm in ["vllm", "ollama"] and hasattr(question.model, 'url'):
        cache_key_parts.append(question.model.url)  # type: ignore

    return tuple(cache_key_parts)


async def _create_reasoning_llm_instance(question) -> Any:
    """Crea una nuova istanza del modello LLM specializzato per reasoning"""
    try:
        llm_params = get_llm_params(
            provider=question.llm,
            temperature=question.temperature,
            top_p=question.top_p,
            max_tokens=question.max_tokens
        )

        client_base_config = await _get_llm_config_for_client(question, llm_params)

        client_config = {**llm_params}
        if client_base_config.get("api_key"):
            client_config["api_key"] = client_base_config["api_key"]
        if client_base_config.get("model"):
            client_config["model"] = client_base_config["model"]
        if client_base_config.get("base_url"):
            client_config["base_url"] = client_base_config["base_url"]
        if client_base_config.get("default_headers"):
            client_config["default_headers"] = client_base_config["default_headers"]

        # Estrai la configurazione reasoning se presente
        thinking_config = question.thinking if hasattr(question, 'thinking') and question.thinking else None

        if question.llm == "openai":
            from langchain_openai import ChatOpenAI
            client_config["max_completion_tokens"] = client_config.pop("max_tokens", None)

            # Aggiungi parametri specifici per GPT-5 reasoning
            if thinking_config:
                # Costruisci il dizionario reasoning se ci sono parametri
                reasoning_dict = {}
                if thinking_config.reasoning_effort:
                    reasoning_dict["effort"] = thinking_config.reasoning_effort
                if thinking_config.reasoning_summary:
                    reasoning_dict["summary"] = thinking_config.reasoning_summary

                # Passa il dizionario reasoning solo se ha contenuti
                if reasoning_dict:
                    client_config["reasoning"] = reasoning_dict
                    # Usa il nuovo formato response per reasoning models
                    client_config["output_version"] = "responses/v1"

            return ChatOpenAI(**client_config)

        elif question.llm == "anthropic":
            from langchain_anthropic import ChatAnthropic
            client_config["anthropic_api_key"] = client_config.pop("api_key", None)

            # Aggiungi parametri specifici per Claude thinking
            if thinking_config:
                thinking_dict = {}
                if thinking_config.type:
                    thinking_dict["type"] = thinking_config.type
                if thinking_config.budget_tokens:
                    thinking_dict["budget_tokens"] = thinking_config.budget_tokens
                if thinking_dict:
                    client_config["thinking"] = thinking_dict

            return ChatAnthropic(**client_config)

        elif question.llm == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            client_config["google_api_key"] = client_config.pop("api_key", None)

            # Aggiungi parametri specifici per Gemini thinking
            if thinking_config:
                if thinking_config.thinkingBudget is not None:
                    # Gemini 2.5 Pro usa thinkingBudget
                    client_config["thinking_budget"] = thinking_config.thinkingBudget
                elif thinking_config.thinkingLevel:
                    # Gemini 3.0 Pro usa thinkingLevel
                    client_config["thinking_level"] = thinking_config.thinkingLevel

            return ChatGoogleGenerativeAI(**client_config)

        elif question.llm == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            # DeepSeek non ha parametri specifici per reasoning, è automatico
            return ChatDeepSeek(**client_config)

        elif question.llm == "vllm":
            from langchain_openai import ChatOpenAI # vLLM uses OpenAI compatible API
            client_config["max_completion_tokens"] = client_config.pop("max_tokens", None)
            return ChatOpenAI(**client_config)

        else:
            logger.warning(f"LLM provider '{question.llm}' not optimized for reasoning, falling back to OpenAI")
            from langchain_openai import ChatOpenAI
            client_config["max_completion_tokens"] = client_config.pop("max_tokens", None)
            return ChatOpenAI(**client_config)

    except Exception as e:
        logger.error(f"Errore nella creazione del modello LLM reasoning {question.llm}: {e}", exc_info=True)
        raise

def decode_jwt(token:str):
    import jwt
    config = get_service_config()
    jwt_secret_key = config["jwt"]["secret_key"]
    if not jwt_secret_key:
        raise ValueError("JWT_SECRET_KEY not configured")
    return jwt.decode(jwt=token, key=jwt_secret_key, algorithms=['HS256'])

