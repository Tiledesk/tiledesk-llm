import torch
from functools import wraps
import hashlib

import logging
from typing import Dict, Any, Callable, Tuple

from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_deepseek import ChatDeepSeek
from langchain_mistralai import ChatMistralAI

from langchain_ollama import ChatOllama
from pydantic import SecretStr

from tilellm.models import EmbeddingModel
from tilellm.shared import const

from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_groq import ChatGroq

from tilellm.shared.embedding_factory import EmbeddingFactory, AsyncEmbeddingFactory
from tilellm.shared.tiledesk_chatmodel_info import TiledeskAICallbackHandler
from tilellm.shared.timed_cache import TimedCache
from tilellm.shared.llm_config import get_llm_params

logger = logging.getLogger(__name__)

def _hash_api_key(api_key: str) -> str:
    """
    Crea un hash SHA256 della chiave API per utilizzarlo nella cache
    senza esporre la chiave completa nei log.
    """
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:16]

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

        # Costruisci chiave cache
        cache_key_parts = [engine_name]
        if repo_type:
            cache_key_parts.append(repo_type)

        if hasattr(question.engine, 'host') and question.engine.host:
            cache_key_parts.append(question.engine.host)
        elif hasattr(question.engine, 'endpoint') and question.engine.endpoint:
            cache_key_parts.append(question.engine.endpoint)

        cache_key = tuple(cache_key_parts)

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
                    logger.info(f"Creato async QdrantRepository con chiave: {cache_key}")
                    return repo_instance

                else:
                    raise ValueError(f"Unknown engine name: {engine_name}")

            except Exception as e:
                logger.error(f"Errore nella creazione async del repository {cache_key}: {e}")
                raise

        try:
            # Ottieni il repository dalla cache (versione async)
            repo = await TimedCache.async_get(
                object_type="repository",
                key=cache_key,
                constructor=_async_creator
            )

            if repo is None:
                logger.error(f"Repository None ottenuto dalla cache async per chiave: {cache_key}")
                raise RuntimeError(f"Failed to get repository for key: {cache_key}")

            logger.debug(f"Repository async {type(repo).__name__} ottenuto/creato per chiave: {cache_key}")

            kwargs['repo'] = repo

            # Esegui la funzione async originale
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Errore nel decoratore async inject_repo per {func.__name__}: {e}")
            raise

    return async_wrapper



def inject_llm(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        try:
            logger.debug(question)
            # 1. Crea una chiave di cache univoca per il modello LLM
            cache_key = (
                question.llm,
                question.model,
                _hash_api_key(str(question.llm_key.get_secret_value()))  # Hash della chiave per sicurezza
            )
            def _creator():
                # Ottieni parametri filtrati per il provider specifico
                llm_params = get_llm_params(
                    provider=question.llm,
                    temperature=question.temperature,
                    top_p=question.top_p,
                    max_tokens=question.max_tokens
                )

                if question.llm == "openai":
                    # OLD: return ChatOpenAI(api_key=question.llm_key, model=question.model,
                    #                        temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
                    return ChatOpenAI(api_key=question.llm_key, model=question.model, **llm_params)

                elif question.llm == "anthropic":
                    # OLD: return ChatAnthropic(anthropic_api_key=question.llm_key, model=question.model,
                    #                           temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
                    return ChatAnthropic(anthropic_api_key=question.llm_key, model=question.model, **llm_params)

                elif question.llm == "cohere":
                    # OLD: return ChatCohere(cohere_api_key=question.llm_key, model=question.model,
                    #                        temperature=question.temperature, max_tokens=question.max_tokens)
                    # p è compreso tra 0.00 e 0.99
                    return ChatCohere(cohere_api_key=question.llm_key, model=question.model, **llm_params)

                elif question.llm == "google":
                    # OLD: return ChatGoogleGenerativeAI(google_api_key=question.llm_key, model=question.model,
                    #                                     temperature=question.temperature, max_tokens=question.max_tokens,
                    #                                     top_p=question.top_p, convert_system_message_to_human=True)
                    # ai_msg.usage_metadata per controllare i token
                    # convert_system_message_to_human è deprecato, LangChain gestisce automaticamente i system message
                    return ChatGoogleGenerativeAI(google_api_key=question.llm_key, model=question.model, **llm_params)

                elif question.llm == "ollama":
                    # OLD: return ChatOllama(model=question.model.name, temperature=question.temperature,
                    #                        num_predict=question.max_tokens, top_p=question.max_tokens, base_url=question.model.url)
                    return ChatOllama(model=question.model.name, base_url=question.model.url, **llm_params)

                elif question.llm == "vllm":
                    # OLD: return ChatOpenAI(api_key=SecretStr(question.llm_key), model=question.model.name, base_url=question.model.url,
                    #                        temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
                    return ChatOpenAI(api_key=SecretStr(question.llm_key), model=question.model.name,
                                     base_url=question.model.url, **llm_params)

                elif question.llm == "groq":
                    # OLD: return ChatGroq(api_key=question.llm_key, model=question.model,
                    #                      temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
                    return ChatGroq(api_key=question.llm_key, model=question.model, **llm_params)

                elif question.llm == "deepseek":
                    # OLD: return ChatDeepSeek(api_key=question.llm_key, model=question.model,
                    #                          temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
                    return ChatDeepSeek(api_key=question.llm_key, model=question.model, **llm_params)

                elif question.llm == "mistralai":
                    # OLD: return ChatMistralAI(api_key=question.llm_key, model_name=question.model,
                    #                           temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
                    return ChatMistralAI(api_key=question.llm_key, model_name=question.model, **llm_params)

                else:
                    # OLD: return ChatOpenAI(api_key=question.llm_key, model=question.model,
                    #                        temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
                    return ChatOpenAI(api_key=question.llm_key, model=question.model, **llm_params)

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
            logger.error(f"Errore nel decorator inject_llm per {func.__name__}: {e}", exc_info=True)
            raise LLMInjectionError(f"LLM injection failed: {e}") from e

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
        cache_key_parts.append(question.model.url)

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
            llm_cache_key_parts = [
                question.llm,
                question.model if isinstance(question.model, str) else question.model.name,
                _hash_api_key(str(question.gptkey.get_secret_value()))
            ]
            if question.llm in ["vllm", "ollama"]:
                llm_cache_key_parts.append(question.model.url)

            llm_cache_key = tuple(llm_cache_key_parts)

            # Costruttore per il modello LLM (eseguito solo se l'oggetto non è in cache)
            def _llm_creator():
                logger.info(f"Creazione nuovo oggetto Chat in cache con chiave: {llm_cache_key}")
                if question.llm == "openai":
                    return ChatOpenAI(api_key=question.gptkey, model=question.model, temperature=question.temperature,
                                      max_tokens=question.max_tokens, top_p=question.top_p)
                elif question.llm == "anthropic":
                    return ChatAnthropic(anthropic_api_key=question.gptkey, model=question.model,
                                         temperature=question.temperature, max_tokens=question.max_tokens,
                                         top_p=question.top_p)
                elif question.llm == "cohere":
                    return ChatCohere(cohere_api_key=question.gptkey, model=question.model,
                                      temperature=question.temperature, max_tokens=question.max_tokens)
                elif question.llm == "google":
                    # convert_system_message_to_human è deprecato, LangChain gestisce automaticamente i system message
                    return ChatGoogleGenerativeAI(google_api_key=question.gptkey, model=question.model,
                                                  temperature=question.temperature, max_tokens=question.max_tokens,
                                                  top_p=question.top_p)
                elif question.llm == "mistralai":
                    return ChatMistralAI(api_key=question.gptkey,model_name=question.model,
                                         temperature=question.temperature,max_tokens=question.max_tokens,
                                         top_p=question.top_p)

                elif question.llm == "vllm":
                    return ChatOpenAI(api_key=SecretStr(question.gptkey), model=question.model.name,
                                      base_url=question.model.url, temperature=question.temperature,
                                      max_tokens=question.max_tokens, top_p=question.top_p)
                elif question.llm == "groq":
                    return ChatGroq(api_key=question.gptkey, model=question.model, temperature=question.temperature,
                                    max_tokens=question.max_tokens, top_p=question.top_p)
                elif question.llm == "deepseek":
                    return ChatDeepSeek(api_key=question.gptkey, model=question.model, temperature=question.temperature,
                                        max_tokens=question.max_tokens, top_p=question.top_p)
                elif question.llm == "ollama":
                    return ChatOllama(model=question.model.name, temperature=question.temperature,
                                      num_predict=question.max_tokens, base_url=question.model.url)
                else:  # Fallback a OpenAI
                    return ChatOpenAI(api_key=question.gptkey, model=question.model, temperature=question.temperature,
                                      max_tokens=question.max_tokens, top_p=question.top_p)

            # Recupera o crea il modello LLM dalla cache
            llm = TimedCache.get(object_type="chat", key=llm_cache_key, constructor=_llm_creator)

            # --- 2. Gestione Cache per il Modello di Embedding ---

            # Usa la tua EmbeddingFactory già refattorizzata (soluzione più pulita) o la logica qui sotto.
            # Qui replichiamo la logica per chiarezza.
            embedding_config = {}
            if isinstance(question.embedding, EmbeddingModel):
                embedding_config = {
                    "provider": question.embedding.embedding_provider,
                    "model_name": question.embedding.embedding_model,
                    "api_key": question.embedding.embedding_key,
                    "base_url": question.embedding.embedding_host
                }
            else:  # Modalità legacy con stringa
                embedding_config = {
                    "provider": question.embedding,  # es. "openai" o "huggingface"
                    "model_name": question.embedding,
                    "api_key": question.gptkey,
                    "legacy_mode": True
                }

            # Qui usiamo la EmbeddingFactory che a sua volta usa TimedCache.
            # factory = EmbeddingFactory()
            # llm_embeddings, _ = factory.create(embedding_config)
            # Per adesso, la lasciamo esplicita:

            embedding_cache_key = tuple(sorted(embedding_config.items()))

            def _embedding_creator():
                logger.info(f"Creazione nuovo oggetto Embedding in cache con chiave: {embedding_cache_key}")
                # ... Inserire qui la logica di creazione degli embedding che era nel decorator originale ...
                # Questa parte diventa molto complessa, per questo usare EmbeddingFactory è meglio.
                # Per semplicità, ipotizziamo di avere la factory:
                factory = EmbeddingFactory()
                embedding_obj, _ = factory.create(embedding_config)
                return embedding_obj

            llm_embeddings = _embedding_creator()#TimedCache.get(object_type="embedding", key=embedding_cache_key, constructor=_embedding_creator)

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
                logger.info(f"Creazione nuovo oggetto Chat in cache con chiave: {llm_cache_key}")
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
                logger.info(f"Creazione nuovo oggetto Embedding in cache con chiave: {embedding_cache_key}")
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

            logger.debug(f"LLM e Embedding iniettati con successo per {func.__name__}")
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Errore durante l'iniezione async del LLM per {func.__name__}: {e}", exc_info=True)
            raise

    return async_wrapper




async def _build_llm_cache_key(question) -> tuple:
    """Costruisce la chiave di cache per il modello LLM"""

    cache_key_parts = [
        "chat",
        question.llm,
        question.model if isinstance(question.model, str) else question.model.name,
        _hash_api_key(str(question.gptkey.get_secret_value()))
    ]

    # Aggiungi URL per modelli self-hosted
    if question.llm in ["vllm", "ollama"] and hasattr(question.model, 'url'):
        cache_key_parts.append(question.model.url)

    return tuple(cache_key_parts)


async def _build_embedding_cache_key(question) -> tuple:
    """Costruisce la chiave di cache per gli embedding"""
    embedding_config = {}

    if isinstance(question.embedding, EmbeddingModel):
        embedding_config = {
            "provider": question.embedding.embedding_provider,
            "model_name": question.embedding.embedding_model,
            "api_key": _hash_api_key(str(question.embedding.embedding_key.get_secret_value())),  # Hash della chiave
            "base_url": question.embedding.embedding_host
        }
    else:  # Modalità legacy con stringa
        embedding_config = {
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
        # Ottieni parametri filtrati per il provider specifico
        llm_params = get_llm_params(
            provider=question.llm,
            temperature=question.temperature,
            top_p=question.top_p,
            max_tokens=question.max_tokens
        )

        if question.llm == "openai":
            from langchain_openai import ChatOpenAI
            # OLD: return ChatOpenAI(api_key=question.gptkey, model=question.model,
            #                        temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatOpenAI(api_key=question.gptkey, model=question.model, **llm_params)

        elif question.llm == "anthropic":
            from langchain_anthropic import ChatAnthropic
            # OLD: return ChatAnthropic(anthropic_api_key=question.gptkey, model=question.model,
            #                           temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatAnthropic(anthropic_api_key=question.gptkey, model=question.model, **llm_params)

        elif question.llm == "cohere":
            from langchain_cohere import ChatCohere
            # OLD: return ChatCohere(cohere_api_key=question.gptkey, model=question.model,
            #                        temperature=question.temperature, max_tokens=question.max_tokens)
            return ChatCohere(cohere_api_key=question.gptkey, model=question.model, **llm_params)

        elif question.llm == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            # OLD: return ChatGoogleGenerativeAI(google_api_key=question.gptkey, model=question.model,
            #                                     temperature=question.temperature, max_tokens=question.max_tokens,
            #                                     top_p=question.top_p, convert_system_message_to_human=True)
            # convert_system_message_to_human è deprecato, LangChain gestisce automaticamente i system message
            return ChatGoogleGenerativeAI(google_api_key=question.gptkey, model=question.model, **llm_params)

        elif question.llm == "mistralai":
            from langchain_mistralai import ChatMistralAI
            # OLD: return ChatMistralAI(api_key=question.gptkey, model_name=question.model,
            #                           temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatMistralAI(api_key=question.gptkey, model_name=question.model, **llm_params)

        elif question.llm == "vllm":
            from langchain_openai import ChatOpenAI
            # OLD: return ChatOpenAI(api_key=SecretStr(question.gptkey), model=question.model.name, base_url=question.model.url,
            #                        temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatOpenAI(api_key=SecretStr(question.gptkey), model=question.model.name,
                             base_url=question.model.url, **llm_params)

        elif question.llm == "groq":
            from langchain_groq import ChatGroq
            # OLD: return ChatGroq(api_key=question.gptkey, model=question.model,
            #                      temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatGroq(api_key=question.gptkey, model=question.model, **llm_params)

        elif question.llm == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            # OLD: return ChatDeepSeek(api_key=question.gptkey, model=question.model,
            #                          temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatDeepSeek(api_key=question.gptkey, model=question.model, **llm_params)

        elif question.llm == "ollama":
            from langchain_community.chat_models import ChatOllama
            # OLD: return ChatOllama(model=question.model.name, temperature=question.temperature,
            #                        num_predict=question.max_tokens, base_url=question.model.url)
            return ChatOllama(model=question.model.name, base_url=question.model.url, **llm_params)

        else:  # Fallback a OpenAI
            from langchain_openai import ChatOpenAI
            logger.warning(f"Unknown LLM provider '{question.llm}', falling back to OpenAI")
            # OLD: return ChatOpenAI(api_key=question.gptkey, model=question.model,
            #                        temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatOpenAI(api_key=question.gptkey, model=question.model, **llm_params)

    except Exception as e:
        logger.error(f"Errore nella creazione del modello LLM {question.llm}: {e}")
        raise


async def _create_standard_llm_instance(question) -> Any:
    """Crea una nuova istanza del modello LLM standard usando configurazione centralizzata"""
    try:
        # Ottieni parametri filtrati per il provider specifico
        llm_params = get_llm_params(
            provider=question.llm,
            temperature=question.temperature,
            top_p=question.top_p,
            max_tokens=question.max_tokens
        )

        if question.llm == "openai":
            # OLD: return ChatOpenAI(api_key=question.llm_key, model=question.model,
            #                        temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatOpenAI(api_key=question.llm_key, model=question.model, **llm_params)

        elif question.llm == "anthropic":
            # OLD: return ChatAnthropic(anthropic_api_key=question.llm_key, model=question.model,
            #                           temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatAnthropic(anthropic_api_key=question.llm_key, model=question.model, **llm_params)

        elif question.llm == "cohere":
            # OLD: return ChatCohere(cohere_api_key=question.llm_key, model=question.model,
            #                        temperature=question.temperature, max_tokens=question.max_tokens)
            return ChatCohere(cohere_api_key=question.llm_key, model=question.model, **llm_params)

        elif question.llm == "google":
            # OLD: return ChatGoogleGenerativeAI(google_api_key=question.llm_key, model=question.model,
            #                                     temperature=question.temperature, max_tokens=question.max_tokens,
            #                                     top_p=question.top_p, convert_system_message_to_human=True)
            # convert_system_message_to_human è deprecato, LangChain gestisce automaticamente i system message
            return ChatGoogleGenerativeAI(google_api_key=question.llm_key, model=question.model, **llm_params)

        elif question.llm == "ollama":
            # OLD: return ChatOllama(model=question.model.name, temperature=question.temperature,
            #                        num_predict=question.max_tokens, top_p=question.top_p, base_url=question.model.url)
            return ChatOllama(model=question.model.name, base_url=question.model.url, **llm_params)

        elif question.llm == "vllm":
            # OLD: return ChatOpenAI(api_key=SecretStr(question.llm_key), model=question.model.name, base_url=question.model.url,
            #                        temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatOpenAI(api_key=SecretStr(question.llm_key), model=question.model.name,
                             base_url=question.model.url, **llm_params)

        elif question.llm == "groq":
            # OLD: return ChatGroq(api_key=question.llm_key, model=question.model,
            #                      temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatGroq(api_key=question.llm_key, model=question.model, **llm_params)

        elif question.llm == "deepseek":
            # OLD: return ChatDeepSeek(api_key=question.llm_key, model=question.model,
            #                          temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatDeepSeek(api_key=question.llm_key, model=question.model, **llm_params)

        elif question.llm == "mistralai":
            # OLD: return ChatMistralAI(api_key=question.llm_key, model_name=question.model,
            #                           temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatMistralAI(api_key=question.llm_key, model_name=question.model, **llm_params)

        else:
            # Fallback a OpenAI
            logger.warning(f"Unknown LLM provider '{question.llm}', falling back to OpenAI")
            # OLD: return ChatOpenAI(api_key=question.llm_key, model=question.model,
            #                        temperature=question.temperature, max_tokens=question.max_tokens, top_p=question.top_p)
            return ChatOpenAI(api_key=question.llm_key, model=question.model, **llm_params)

    except Exception as e:
        logger.error(f"Errore nella creazione del modello LLM standard {question.llm}: {e}")
        raise

async def _create_embedding_instance(question):
    """Crea una nuova istanza del modello di embedding utilizzando EmbeddingFactory"""
    try:
        # Prepara la configurazione per EmbeddingFactory
        if isinstance(question.embedding, EmbeddingModel):
            embedding_config = {
                "provider": question.embedding.embedding_provider,
                "model_name": question.embedding.embedding_model,
                "api_key": question.embedding.embedding_key,
                "base_url": question.embedding.embedding_host
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

            # 1. Crea una chiave di cache univoca per il modello LLM
            cache_key = (
                question.llm,
                question.model,
                _hash_api_key(str(question.llm_key.get_secret_value()))  # Hash della chiave per sicurezza
            )

            def _creator():
                if question.llm == "openai":
                    return ChatOpenAI(api_key=question.llm_key,
                                      model=question.model,
                                      temperature=question.temperature,
                                      max_completion_tokens=question.max_tokens)
                elif question.llm == "anthropic":
                    return ChatAnthropic(anthropic_api_key=question.llm_key,
                                         model=question.model,
                                         temperature=question.temperature,
                                         thinking=question.thinking,
                                         max_tokens=question.max_tokens)
                elif question.llm == "deepseek":
                    return ChatDeepSeek(api_key=question.llm_key,
                                        model=question.model,
                                        temperature=question.temperature,
                                        )
                else:
                    # Comportamento di default come OpenAI
                    return ChatOpenAI(api_key=question.llm_key,
                                      model=question.model,
                                      temperature=question.temperature,
                                      max_completion_tokens=question.max_tokens)

            chat_model = TimedCache.get(
                object_type="reasoning",  # Assicurati che questo corrisponda al tipo di oggetto gestito dalla tua cache
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
            raise LLMInjectionError(f"Reasoning LLM injection failed: {e}") from e

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
        cache_key_parts.append(question.model.url)

    return tuple(cache_key_parts)


async def _create_reasoning_llm_instance(question) -> Any:
    """Crea una nuova istanza del modello LLM specializzato per reasoning"""
    try:
        if question.llm == "openai":
            # Per OpenAI, usa max_completion_tokens invece di max_tokens per reasoning
            return ChatOpenAI(
                api_key=question.llm_key,
                model=question.model,
                temperature=question.temperature,
                max_completion_tokens=question.max_tokens
            )

        elif question.llm == "anthropic":
            # Anthropic supporta il thinking mode per reasoning
            kwargs = {
                "anthropic_api_key": question.llm_key,
                "model": question.model,
                "temperature": question.temperature,
                "max_tokens": question.max_tokens
            }

            # Aggiungi thinking mode se disponibile
            if hasattr(question, 'thinking') and question.thinking is not None:
                kwargs["thinking"] = question.thinking

            return ChatAnthropic(**kwargs)

        elif question.llm == "deepseek":
            # DeepSeek supporta bene il reasoning
            return ChatDeepSeek(
                api_key=question.llm_key,
                model=question.model,
                temperature=question.temperature,
                max_tokens=question.max_tokens
            )

        else:
            # Fallback a OpenAI con max_completion_tokens
            logger.warning(f"LLM provider '{question.llm}' not optimized for reasoning, falling back to OpenAI")
            return ChatOpenAI(
                api_key=question.llm_key,
                model=question.model,
                temperature=question.temperature,
                max_completion_tokens=question.max_tokens
            )

    except Exception as e:
        logger.error(f"Errore nella creazione del modello LLM reasoning {question.llm}: {e}")
        raise


def inject_reason_llm_old(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
        if question.llm == "openai":
            chat_model = ChatOpenAI(api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_completion_tokens=question.max_tokens)
        elif question.llm == "anthropic":
            chat_model = ChatAnthropic(anthropic_api_key=question.llm_key,
                                       model=question.model,
                                       temperature=question.temperature,
                                       thinking=question.thinking,
                                       max_tokens=question.max_tokens)
        elif question.llm == "deepseek":
            chat_model = ChatDeepSeek(api_key=question.llm_key,
                                      model=question.model,
                                      temperature=question.temperature,
                                      #max_tokens=question.max_tokens
                                      )

        else:
            chat_model = ChatOpenAI(api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_completion_tokens=question.max_tokens)



        # Add chat_model agli kwargs
        kwargs['chat_model'] = chat_model

        # Chiama la funzione originale con i nuovi kwargs
        return await func(question, *args, **kwargs)

    return wrapper

def decode_jwt(token:str):
    import jwt
    jwt_secret_key = const.JWT_SECRET_KEY
    return jwt.decode(jwt=token, key=jwt_secret_key, algorithms=['HS256'])


