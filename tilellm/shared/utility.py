import torch
from functools import wraps

import logging
from typing import Dict, Any

from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.embeddings import CohereEmbeddings  # , GooglePalmEmbeddings
from langchain_deepseek import ChatDeepSeek

from langchain_ollama import ChatOllama
from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from tilellm.models.item_model import EmbeddingModel
from tilellm.shared import const

from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_groq import ChatGroq

from tilellm.shared.embedding_factory import EmbeddingFactory
from tilellm.shared.tiledesk_chatmodel_info import TiledeskAICallbackHandler
from tilellm.shared.timed_cache import TimedCache

logger = logging.getLogger(__name__)







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
        cache_key = (engine_name, repo_type)

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


# DA RIMUOVERE
def inject_embedding():
    def decorator(func):
        @wraps(func)
        async def wrapper(self, item, *args, **kwargs):

            if item.model and item.model.name.lower() == "huggingface":
                embedding_obj = LocalEmbeddingModelCache.get_model(
                    model_name=item.model.name,
                    normalize_embeddings=True
                )
                dimension = item.model.dimension or 384

                # Caso retrocompatibile con stringa
            elif item.embedding == "huggingface":
                embedding_obj = LocalEmbeddingModelCache.get_model(
                    model_name="BAAI/bge-m3",
                    normalize_embeddings=True
                )
                dimension = 1024

            elif item.model:  # Caso nuovo con configurazione avanzata
                if item.model.name.lower() == "ollama":
                    from langchain_ollama.embeddings import OllamaEmbeddings
                    embedding_obj = OllamaEmbeddings(
                        model=item.model.name,
                        base_url=item.model.url or "http://localhost:11434"
                    )
                    dimension = item.model.dimension or 4096  # Default per Ollama

                #elif item.model.name.lower() == "huggingface":
                #    import torch
                #    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
                #    device = 'cuda' if torch.cuda.is_available() else 'cpu'

                #    embedding_obj = HuggingFaceEmbeddings(
                #        model_name=item.model.name,  # Es: "sentence-transformers/all-MiniLM-L6-v2"
                #        model_kwargs={'device': device},
                #        encode_kwargs={'normalize_embeddings': True}
                #    )
                #    dimension = item.model.dimension or 384  # Dimensione tipica per MiniLM-L6

                elif item.model.name.lower() in ["google", "cohere", "voyage"]:
                    # Gestione provider con API key esterna
                    provider_map = {
                        "google": GoogleGenerativeAIEmbeddings,
                        "cohere": CohereEmbeddings,
                        "voyage": VoyageAIEmbeddings
                    }
                    embedding_obj = provider_map[item.model.name.lower()](
                        model=item.model.name,
                        **{f"{item.model.name.lower()}_api_key": item.gptkey}
                    )
                    dimension = item.model.dimension or 1024

                else:  # Default per modelli OpenAI personalizzati
                    embedding_obj = OpenAIEmbeddings(
                        api_key=item.gptkey,
                        model=item.model.name
                    )
                    dimension = item.model.dimension or 1536

            else:  # Retrocompatibilità con campo embedding stringa
                from langchain_huggingface.embeddings import HuggingFaceEmbeddings
                from langchain_ollama.embeddings import OllamaEmbeddings
                embedding_map = {
                    "text-embedding-ada-002": (OpenAIEmbeddings, 1536),
                    "text-embedding-3-small": (OpenAIEmbeddings, 1536),
                    "text-embedding-3-large": (OpenAIEmbeddings, 3072),
                    "voyage-multilingual-2": (VoyageAIEmbeddings, 1024),
                    #"huggingface": (HuggingFaceEmbeddings, 1024),
                    "ollama": (OllamaEmbeddings, 4096),
                    "google": (GoogleGenerativeAIEmbeddings, 768),
                    "cohere": (CohereEmbeddings, 1024)
                }

                if item.embedding in embedding_map:
                    import torch
                    cls, dim = embedding_map[item.embedding]
                    embedding_obj = cls(**{
                        "api_key": item.gptkey,
                        "model": item.embedding
                    }) if cls != HuggingFaceEmbeddings else HuggingFaceEmbeddings(
                        model_name="BAAI/bge-m3",
                        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    dimension = dim
                else:  # Fallback a OpenAI
                    embedding_obj = OpenAIEmbeddings(
                        api_key=item.gptkey,
                        model=item.embedding
                    )
                    dimension = 1536

            # Aggiungi var1 e var2 agli kwargs
            kwargs['embedding_obj'] = embedding_obj
            kwargs['embedding_dimension'] = dimension

            # Chiama la funzione originale con i nuovi kwargs
            return await func(self, item, *args, **kwargs)

        return wrapper

    return decorator


def inject_llm(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        try:
            logger.debug(question)
            # 1. Crea una chiave di cache univoca per il modello LLM
            cache_key = (
                question.llm,
                question.model,
                str(question.llm_key)  # L'API Key è cruciale
            )
            def _creator():
                if question.llm == "openai":
                    return ChatOpenAI(api_key=question.llm_key,
                                            model=question.model,
                                            temperature=question.temperature,
                                            max_tokens=question.max_tokens,
                                            top_p=question.top_p)

                elif question.llm == "anthropic":
                   return ChatAnthropic(anthropic_api_key=question.llm_key,
                                               model=question.model,
                                               temperature=question.temperature,
                                               max_tokens=question.max_tokens,
                                               top_p=question.top_p)

                elif question.llm == "cohere":
                    return ChatCohere(cohere_api_key=question.llm_key,
                                            model=question.model,
                                            temperature=question.temperature,
                                            max_tokens=question.max_tokens
                                            )
                    #p è compreso tra 0.00 e 0.99

                elif question.llm == "google":
                    return ChatGoogleGenerativeAI(google_api_key=question.llm_key,
                                                        model=question.model,
                                                        temperature=question.temperature,
                                                        max_tokens=question.max_tokens,
                                                        top_p=question.top_p,
                                                        convert_system_message_to_human=True)

                    #ai_msg.usage_metadata per controllare i token

                elif question.llm == "ollama":
                    return ChatOllama(model = question.model.name,
                                            temperature=question.temperature,
                                            num_predict=question.max_tokens,
                                            top_p=question.max_tokens,
                                            base_url=question.model.url)
                elif question.llm == "vllm":
                    return ChatOpenAI(api_key=SecretStr(question.llm_key),
                                     model=question.model.name,
                                     base_url=question.model.url,
                                     temperature=question.temperature,
                                     max_tokens=question.max_tokens,
                                     top_p=question.top_p
                                     )

                elif question.llm == "groq":
                    return ChatGroq(api_key=question.llm_key,
                                          model=question.model,
                                          temperature=question.temperature,
                                          max_tokens=question.max_tokens,
                                          top_p=question.top_p
                                          )

                elif question.llm == "deepseek":
                    return ChatDeepSeek(api_key=question.llm_key,
                                          model=question.model,
                                          temperature=question.temperature,
                                          max_tokens=question.max_tokens,
                                          top_p=question.top_p
                                          )


                else:
                    return ChatOpenAI(api_key=question.llm_key,
                                            model=question.model,
                                            temperature=question.temperature,
                                            max_tokens=question.max_tokens,
                                            top_p=question.top_p
                                            )

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
                str(question.gptkey)
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
                    return ChatGoogleGenerativeAI(google_api_key=question.gptkey, model=question.model,
                                                  temperature=question.temperature, max_tokens=question.max_tokens,
                                                  top_p=question.top_p, convert_system_message_to_human=True)
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

            # Qui dovresti usare la EmbeddingFactory che a sua volta usa TimedCache.
            # factory = EmbeddingFactory()
            # llm_embeddings, _ = factory.create(embedding_config)
            # Per questo esempio, la lasciamo esplicita:

            embedding_cache_key = tuple(sorted(embedding_config.items()))

            def _embedding_creator():
                logger.info(f"Creazione nuovo oggetto Embedding in cache con chiave: {embedding_cache_key}")
                # ... Inserire qui la logica di creazione degli embedding che era nel decorator originale ...
                # Questa parte diventa molto complessa, per questo usare EmbeddingFactory è meglio.
                # Per semplicità, ipotizziamo di avere la factory:
                factory = EmbeddingFactory()
                embedding_obj, _ = factory.create(embedding_config)
                return embedding_obj

            llm_embeddings = TimedCache.get(object_type="embedding", key=embedding_cache_key,
                                            constructor=_embedding_creator)

            # --- 3. Creazione del Callback Handler (SEMPRE NUOVO) ---

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

def inject_llm_chat_old(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
        try:
            cache_key = (
                question.llm,
                question.model,
                str(question.gptkey)  # L'API Key è crucial
            )

            if question.llm == "openai":
                callback_handler = OpenAICallbackHandler()
                llm = ChatOpenAI(api_key=question.gptkey,
                                 model=question.model,
                                 temperature=question.temperature,
                                 max_tokens=question.max_tokens,
                                 top_p=question.top_p,
                                 callbacks=[callback_handler])

            elif question.llm == "anthropic":
                callback_handler = TiledeskAICallbackHandler()
                llm = ChatAnthropic(anthropic_api_key=question.gptkey,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_tokens=question.max_tokens,
                                    top_p=question.top_p,
                                    #thinking={"type": "enabled", "budget_tokens": 10_000},
                                    callbacks=[callback_handler])

            elif question.llm == "cohere":
                callback_handler = TiledeskAICallbackHandler()
                llm = ChatCohere(cohere_api_key=question.gptkey,
                                 model=question.model,
                                 temperature=question.temperature,
                                 max_tokens=question.max_tokens)

            elif question.llm == "google":
                callback_handler = TiledeskAICallbackHandler()
                llm = ChatGoogleGenerativeAI(google_api_key=question.gptkey,
                                             model=question.model,
                                             temperature=question.temperature,
                                             max_tokens=question.max_tokens,
                                             top_p=question.top_p,
                                             convert_system_message_to_human=True)

            elif question.llm == "vllm":
                callback_handler = TiledeskAICallbackHandler()
                llm = ChatOpenAI(api_key=SecretStr(question.gptkey),
                                 model=question.model.name,
                                 base_url=question.model.url,
                                 temperature=question.temperature,
                                 max_tokens=question.max_tokens,
                                 top_p=question.top_p,
                                 callbacks=[callback_handler]
                                 )

            elif question.llm == "groq":
                callback_handler = TiledeskAICallbackHandler()
                llm = ChatGroq(api_key=question.gptkey,
                               model=question.model,
                               temperature=question.temperature,
                               max_tokens=question.max_tokens,
                               top_p=question.top_p
                               )

            elif question.llm == "deepseek":
                callback_handler = TiledeskAICallbackHandler()
                llm = ChatDeepSeek(api_key=question.gptkey,
                                   model=question.model,
                                   temperature=question.temperature,
                                   max_tokens=question.max_tokens,
                                   top_p=question.top_p
                               )

            elif question.llm == "ollama":
                #disable_streaming = not question.stream
                callback_handler = TiledeskAICallbackHandler()
                llm = ChatOllama(model=question.model.name,
                                 temperature=question.temperature,
                                 num_predict=question.max_tokens,
                                 base_url=question.model.url,
                                 #format="json",
                                 disable_streaming=not question.stream,
                                 callback_handler=[callback_handler]
                                 )

            else:
                callback_handler = OpenAICallbackHandler()
                llm = ChatOpenAI(api_key=question.gptkey,
                                 model=question.model,
                                 temperature=question.temperature,
                                 max_tokens=question.max_tokens,
                                 top_p=question.top_p,
                                 callbacks=[callback_handler]
                                 )

            # Verifichiamo se è un'istanza di EmbeddingModel
            if isinstance(question.embedding, EmbeddingModel):
                ## aggiunte per cache
                if question.embedding.embedding_provider == "huggingface":
                    llm_embeddings = LocalEmbeddingModelCache.get_model(
                        model_name=question.embedding.embedding_model,
                        normalize_embeddings=True
                    )
                else:
                    provider = question.embedding.embedding_provider
                    key = question.embedding.embedding_key
                    model_name = question.embedding.embedding_model
                    base_url = question.embedding.embedding_host

                    if provider == "openai":
                        llm_embeddings = OpenAIEmbeddings(api_key=key, model=model_name, timeout=30)

                    #elif provider == "huggingface":
                    #    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
                    #    import torch
                    #    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    #    llm_embeddings = HuggingFaceEmbeddings(
                    #        model_name=model_name,
                    #        model_kwargs={'device': device},
                    #        encode_kwargs={'normalize_embeddings': False}
                    #    )
                    elif provider == "cohere":
                        llm_embeddings = CohereEmbeddings(model=model_name, cohere_api_key=key)
                    elif provider == "google":
                        llm_embeddings = GoogleGenerativeAIEmbeddings(google_api_key=key, model=model_name)
                    elif provider == "ollama":
                        from langchain_ollama.embeddings import OllamaEmbeddings
                        llm_embeddings = OllamaEmbeddings(model=model_name, base_url=key)             ### Importante se uso ollama la url viene passata nella key
                    elif provider == "vllm":
                        llm_embeddings = OpenAIEmbeddings(
                            model=model_name,
                            base_url=base_url,
                            api_key=SecretStr(key)
                        )

                    else:
                        raise ValueError(f"Provider non supportato: {provider}")


            else:
                embedding_str = question.embedding
                if embedding_str == "huggingface":
                    llm_embeddings = LocalEmbeddingModelCache.get_model(
                        model_name="BAAI/bge-m3",
                        normalize_embeddings=True  # Coerente con implementazione originale
                    )
                    #from langchain_huggingface.embeddings import HuggingFaceEmbeddings
                    #import torch
                    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    #llm_embeddings = HuggingFaceEmbeddings(
                    #    model_name="BAAI/bge-m3",
                    #    model_kwargs={'device': device},
                    #    encode_kwargs={'normalize_embeddings': False}
                    #)
                elif embedding_str == "cohere":
                    llm_embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=question.gptkey)
                elif embedding_str == "google":
                    llm_embeddings = GoogleGenerativeAIEmbeddings(google_api_key=question.gptkey,
                                                                  model="models/embedding-001")
                elif embedding_str == "ollama":
                    from langchain_ollama.embeddings import OllamaEmbeddings
                    llm_embeddings = OllamaEmbeddings(
                        model=question.model.name,
                        base_url=question.model.url
                    )
                else:
                    # Default OpenAI con nome modello dalla stringa
                    llm_embeddings = OpenAIEmbeddings(
                        api_key=question.gptkey,
                        model=embedding_str if embedding_str in ADA_AND_3_MODELS else "text-embedding-ada-002",
                        timeout=30,
                    )

            # Add chat_model agli kwargs
            kwargs['llm'] = llm
            kwargs['callback_handler'] = callback_handler
            kwargs['llm_embeddings'] = llm_embeddings

            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Errore durante l'iniezione del LLM: {e}", exc_info=True)
            raise

    return wrapper

def inject_reason_llm(func):
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


