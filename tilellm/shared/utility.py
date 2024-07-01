import os
from functools import wraps

import logging

from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from tilellm.shared import const

from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_aws.chat_models import ChatBedrockConverse, ChatBedrock


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
    def wrapper(*args, **kwargs):
        repo_type = os.environ.get("PINECONE_TYPE")
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
        return func(*args, **kwargs)

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
            elif item.embedding == "claude-3":
                embedding_obj = VoyageAIEmbeddings(voyage_api_key=const.VOYAGEAI_API_KEY, model="voyage-multilingual-2")
                # query_result = voyage.embed_query(text)
                dimension = 1024
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
        print(question)
        if question.llm == "openai":
            chat_model = ChatOpenAI(api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_tokens=question.max_tokens)

        elif question.llm == "anthropic":
            chat_model = ChatAnthropic(anthropic_api_key=question.llm_key,
                                       model=question.model,
                                       temperature=question.temperature,
                                       max_tokens=question.max_tokens)

        elif question.llm == "cohere":
            chat_model = ChatCohere(cohere_api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_tokens=question.max_tokens)

        elif question.llm == "google":
            chat_model = ChatGoogleGenerativeAI(google_api_key=question.llm_key,
                                                model=question.model,
                                                temperature=question.temperature,
                                                max_tokens=question.max_tokens,
                                                convert_system_message_to_human=True)

        elif question.llm == "groq":
            chat_model = ChatGroq(api_key=question.llm_key,
                                  model=question.model,
                                  temperature=question.temperature,
                                  max_tokens=question.max_tokens
                                  )

        else:
            chat_model = ChatOpenAI(api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_tokens=question.max_tokens)

        # Add chat_model agli kwargs
        kwargs['chat_model'] = chat_model

        # Chiama la funzione originale con i nuovi kwargs
        return await func(question, *args, **kwargs)

    return wrapper



