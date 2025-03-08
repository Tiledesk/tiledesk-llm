from functools import wraps

import logging
from gc import callbacks

import langchain_aws
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.embeddings import CohereEmbeddings  # , GooglePalmEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_ollama import ChatOllama
from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from tilellm.shared import const

from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_groq import ChatGroq
from langchain_aws.chat_models import ChatBedrockConverse, ChatBedrock

from tilellm.shared.tiledesk_chatmodel_info import TiledeskAICallbackHandler

logger = logging.getLogger(__name__)


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
        if question.model == "openai":
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

        if question.embedding == "openai":
            llm_embeddings = OpenAIEmbeddings(api_key=question.gptkey, model=question.embedding)
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


