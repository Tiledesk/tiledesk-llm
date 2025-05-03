from functools import wraps

import logging

from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.embeddings import CohereEmbeddings  # , GooglePalmEmbeddings
from langchain_deepseek import ChatDeepSeek

from langchain_ollama import ChatOllama
from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from tilellm.models.item_model import EmbeddingModel
from tilellm.shared import const

from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_groq import ChatGroq

from tilellm.shared.tiledesk_chatmodel_info import TiledeskAICallbackHandler

logger = logging.getLogger(__name__)







ADA_AND_3_MODELS = {
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large"
}

def inject_repo(func):
    """
    Annotation for inject PineconeRepository.
    If PINECONE_TYP is pod is injected PineconeRepositoryPod
    If PINECONE_TYP is serverless is injected PineconeRepositoryServerless
    :param func:
    :return:
    """

    @wraps(func)
    def wrapper(question, *args, **kwargs):
        engine_name = question.engine.name
        repo_type= question.engine.type
        #print(f"============== ENGINE {question.engine.model_dump()}")
        #repo_type = os.environ.get("PINECONE_TYPE")
        logger.info(f"pinecone type {repo_type}")
        if repo_type == 'pod':

            from tilellm.store.pinecone.pinecone_repository_pod import PineconeRepositoryPod
            repo = PineconeRepositoryPod()
        elif repo_type == 'serverless':
            from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
            repo = PineconeRepositoryServerless()
        else:
            raise ValueError("Unknown repository type")

        kwargs['repo'] = repo
        return func(question, *args, **kwargs)

    return wrapper


def inject_embedding():
    def decorator(func):
        @wraps(func)
        async def wrapper(self, item, *args, **kwargs):

            if item.model:  # Caso nuovo con configurazione avanzata
                if item.model.name.lower() == "ollama":
                    from langchain_ollama.embeddings import OllamaEmbeddings
                    embedding_obj = OllamaEmbeddings(
                        model=item.model.name,
                        base_url=item.model.url or "http://localhost:11434"
                    )
                    dimension = item.model.dimension or 4096  # Default per Ollama

                elif item.model.name.lower() == "huggingface":
                    import torch
                    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'

                    embedding_obj = HuggingFaceEmbeddings(
                        model_name=item.model.name,  # Es: "sentence-transformers/all-MiniLM-L6-v2"
                        model_kwargs={'device': device},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    dimension = item.model.dimension or 384  # Dimensione tipica per MiniLM-L6

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
                    "huggingface": (HuggingFaceEmbeddings, 1024),
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


def inject_embedding_old():
    def decorator(func):
        @wraps(func)
        async def wrapper(self, item, *args, **kwargs):

            # Logica per determinare var1 e var2 basandosi su 'item'
            if item.embedding == "text-embedding-ada-002":
                embedding_obj = OpenAIEmbeddings(api_key=item.gptkey, model=item.embedding)
                dimension = 1536
            elif item.embedding == "text-embedding-3-large":
                embedding_obj = OpenAIEmbeddings(api_key=item.gptkey, model=item.embedding)
                dimension = 3072
            elif item.embedding == "text-embedding-3-small":
                embedding_obj = OpenAIEmbeddings(api_key=item.gptkey, model=item.embedding)
                dimension = 1536
            elif item.embedding == "voyage-multilingual-2":
                embedding_obj = VoyageAIEmbeddings(voyage_api_key=const.VOYAGEAI_API_KEY, model="voyage-multilingual-2")
                # query_result = voyage.embed_query(text)
                dimension = 1024
            elif item.embedding == "huggingface":
                import torch
                from langchain_huggingface.embeddings import HuggingFaceEmbeddings
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model_name = "BAAI/bge-m3"
                model_kwargs = {'device': device}
                encode_kwargs = {'normalize_embeddings': True} #True per cosine similarity

                embedding_obj = HuggingFaceEmbeddings(model_name=model_name,
                                                      model_kwargs=model_kwargs,
                                                      encode_kwargs=encode_kwargs
                                                      )
                dimension = 1024
            elif item.embedding == "ollama":
                from langchain_ollama.embeddings import OllamaEmbeddings
                embedding_obj = OllamaEmbeddings(model=item.model.name,
                                                 base_url=item.model.url
                                                )
                dimension = item.model.dimension
                # dimension for llama3.2 3072
            elif item.embedding == "google":
                embedding_obj = GoogleGenerativeAIEmbeddings(google_api_key=item.gptkey,
                                                             model=item.model.name
                                                             )#"models/embedding-001"
                dimension = item.model.dimension

            elif item.embedding == "cohere":
                embedding_obj = CohereEmbeddings(model=item.model.name,
                                                 cohere_api_key=item.gptkey)  # embed-english-light-v3.0
                dimension = item.model.dimension

            else:
                embedding_obj = OpenAIEmbeddings(api_key=item.gptkey, model=item.embedding)
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
        logger.debug(question)
        if question.llm == "openai":
            chat_model = ChatOpenAI(api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_tokens=question.max_tokens,
                                    top_p=question.top_p)

        elif question.llm == "anthropic":
            chat_model = ChatAnthropic(anthropic_api_key=question.llm_key,
                                       model=question.model,
                                       temperature=question.temperature,
                                       max_tokens=question.max_tokens,
                                       top_p=question.top_p)

        elif question.llm == "cohere":
            chat_model = ChatCohere(cohere_api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_tokens=question.max_tokens
                                    )
            #p è compreso tra 0.00 e 0.99

        elif question.llm == "google":
            chat_model = ChatGoogleGenerativeAI(google_api_key=question.llm_key,
                                                model=question.model,
                                                temperature=question.temperature,
                                                max_tokens=question.max_tokens,
                                                top_p=question.top_p,
                                                convert_system_message_to_human=True)

            #ai_msg.usage_metadata per controllare i token

        elif question.llm == "ollama":
            chat_model = ChatOllama(model = question.model.name,
                                    temperature=question.temperature,
                                    num_predict=question.max_tokens,
                                    top_p=question.max_tokens,
                                    base_url=question.model.url)

        elif question.llm == "groq":
            chat_model = ChatGroq(api_key=question.llm_key,
                                  model=question.model,
                                  temperature=question.temperature,
                                  max_tokens=question.max_tokens,
                                  top_p=question.top_p
                                  )

        elif question.llm == "deepseek":
            chat_model = ChatDeepSeek(api_key=question.llm_key,
                                  model=question.model,
                                  temperature=question.temperature,
                                  max_tokens=question.max_tokens,
                                  top_p=question.top_p
                                  )


        else:
            chat_model = ChatOpenAI(api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_tokens=question.max_tokens,
                                    top_p=question.top_p
                                    )

        # Add chat_model agli kwargs
        kwargs['chat_model'] = chat_model

        # Chiama la funzione originale con i nuovi kwargs
        return await func(question, *args, **kwargs)

    return wrapper

def inject_llm_chat(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
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
            provider = question.embedding.embedding_provider
            key = question.embedding.embedding_key
            model_name = question.embedding.embedding_model

            if provider == "openai":
                llm_embeddings = OpenAIEmbeddings(api_key=key, model=model_name)
            elif provider == "huggingface":
                from langchain_huggingface.embeddings import HuggingFaceEmbeddings
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                llm_embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': False}
                )
            elif provider == "cohere":
                llm_embeddings = CohereEmbeddings(model=model_name, cohere_api_key=key)
            elif provider == "google":
                llm_embeddings = GoogleGenerativeAIEmbeddings(google_api_key=key, model=model_name)
            elif provider == "ollama":
                from langchain_ollama.embeddings import OllamaEmbeddings
                llm_embeddings = OllamaEmbeddings(model=model_name, base_url=key) ### Importante se uso ollama la url viene passata nella key
            else:
                raise ValueError(f"Provider non supportato: {provider}")


        else:
            embedding_str = question.embedding
            if embedding_str == "huggingface":
                from langchain_huggingface.embeddings import HuggingFaceEmbeddings
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                llm_embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3",
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': False}
                )
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
                    model=embedding_str if embedding_str in ADA_AND_3_MODELS else "text-embedding-ada-002"
                )

        """
        # FIXME va implementato il controllo se è embedding model o stringa
        if question.embedding == "openai":
            llm_embeddings = OpenAIEmbeddings(api_key=question.gptkey, model=question.embedding)
        elif isinstance(question.embedding, EmbeddingModel):
            llm_embeddings = OpenAIEmbeddings(api_key=question.embedding.embedding_key, model=question.embedding.embedding_model)
        elif question.embedding == "huggingface":
            import torch
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_name = "BAAI/bge-m3"
            model_kwargs = {'device': device}
            encode_kwargs = {'normalize_embeddings': False}
            llm_embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                                   model_kwargs=model_kwargs,
                                                   encode_kwargs=encode_kwargs
                                                   )
        elif question.embedding == "cohere":
            llm_embeddings = CohereEmbeddings(model=question.embedding, cohere_api_key=question.gptkey)#embed-english-light-v3.0
        elif question.embedding == "google":
            llm_embeddings = GoogleGenerativeAIEmbeddings(google_api_key=question.gptkey,model=question.embedding)#"models/embedding-001"
        elif question.embedding == "ollama":
            from langchain_ollama.embeddings import OllamaEmbeddings
            llm_embeddings = OllamaEmbeddings(model=question.model.name,
                                              base_url=question.model.url
                                              )
            dimension = question.model.dimension
        else:
            llm_embeddings = OpenAIEmbeddings(api_key=question.gptkey, model=question.embedding)
        
        """

        # Add chat_model agli kwargs
        kwargs['llm'] = llm
        kwargs['callback_handler'] = callback_handler
        kwargs['llm_embeddings'] = llm_embeddings

        return await func(question, *args, **kwargs)

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


