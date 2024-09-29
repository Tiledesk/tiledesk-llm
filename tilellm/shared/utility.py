import os
from functools import wraps

import logging
from gc import callbacks

import langchain_aws
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.embeddings import CohereEmbeddings #, GooglePalmEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import OpenAIEmbeddings


from tilellm.shared import const

from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
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
                from langchain_community.embeddings import HuggingFaceBgeEmbeddings
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model_name = "BAAI/bge-m3"
                model_kwargs = {'device': device}
                encode_kwargs = {'normalize_embeddings': True} #True per cosine similarity

                embedding_obj = HuggingFaceBgeEmbeddings(model_name=model_name,
                                                         model_kwargs=model_kwargs,
                                                         encode_kwargs=encode_kwargs

                                                      )
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
        logger.debug(question)
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

        elif question.llm == "aws":
            import os

            os.environ["AWS_SECRET_ACCESS_KEY"] = question.llm_key.aws_secret_access_key
            os.environ["AWS_ACCESS_KEY_ID"] = question.llm_key.aws_access_key_id

            # chat_model = ChatBedrock(model_id=question.model,
            #                         model_kwargs={"temperature": question.temperature,"max_tokens":question.max_tokens },
            #                         region_name="eu-central-1"
            #                         )

            # import boto3

            # client_br = boto3.client('bedrock-runtime',
            #                         aws_access_key_id=question.llm_key.aws_secret_access_key,
            #                         aws_secret_access_key=question.llm_key.aws_secret_access_key,
            #                         region_name=question.llm_key.region_name
            #                         )
            # session = boto3.Session(aws_access_key_id=question.llm_key.aws_secret_access_key,
            #                        aws_secret_access_key=question.llm_key.aws_secret_access_key,
            #                        region_name=question.llm_key.region_name
            #                        )
            # client_ss = session.client("bedrock-runtime")

            chat_model = ChatBedrockConverse(
                # client=client_br,
                model=question.model,
                temperature=question.temperature,
                max_tokens=question.max_tokens,
                region_name=question.llm_key.region_name

                #                                 base_url="http://bedroc-proxy-paacejvmzcgv-121947512.eu-central-1.elb.amazonaws.com/api/v1/",

            )  # model_kwargs={"temperature": 0.001},

            # print(chat_model.client._get_credentials().access_key)


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

def inject_llm_chat(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
        if question.model == "openai":
            callback_handler = OpenAICallbackHandler()
            llm_embeddings = OpenAIEmbeddings(api_key=question.gptkey, model=question.embedding)
            llm = ChatOpenAI(api_key=question.gptkey,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_tokens=question.max_tokens,
                                    callbacks=[callback_handler])

        elif question.llm == "anthropic":
            callback_handler = TiledeskAICallbackHandler()
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
            llm = ChatAnthropic(anthropic_api_key=question.gptkey,
                                       model=question.model,
                                       temperature=question.temperature,
                                       max_tokens=question.max_tokens,
                                       callbacks=[callback_handler])

        elif question.llm == "cohere":
            callback_handler = TiledeskAICallbackHandler()
            llm_embeddings = CohereEmbeddings(model=question.embedding, cohere_api_key=question.gptkey)
            llm = ChatCohere(cohere_api_key=question.gptkey,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_tokens=question.max_tokens)

        elif question.llm == "google":
            callback_handler = TiledeskAICallbackHandler()
            #"models/embedding-gecko-001"
            llm_embeddings = OpenAIEmbeddings(api_key=question.gptkey, model=question.embedding)
            # llm_embeddings = GooglePalmEmbeddings(google_api_key=question.gptkey) #, model_name=question.embedding)
            llm = ChatGoogleGenerativeAI(google_api_key=question.gptkey,
                                                model=question.model,
                                                temperature=question.temperature,
                                                max_tokens=question.max_tokens,
                                                convert_system_message_to_human=True)

        elif question.llm == "groq":
            callback_handler = TiledeskAICallbackHandler()
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
            llm = ChatGroq(api_key=question.gptkey,
                                  model=question.model,
                                  temperature=question.temperature,
                                  max_tokens=question.max_tokens
                                  )

        elif question.llm == "aws":
            import os

            os.environ["AWS_SECRET_ACCESS_KEY"] = question.llm_key.aws_secret_access_key
            os.environ["AWS_ACCESS_KEY_ID"] = question.llm_key.aws_access_key_id
            callback_handler = TiledeskAICallbackHandler()
            llm_embeddings = OpenAIEmbeddings(api_key=question.gptkey, model=question.embedding)

            # chat_model = ChatBedrock(model_id=question.model,
            #                         model_kwargs={"temperature": question.temperature,"max_tokens":question.max_tokens },
            #                         region_name="eu-central-1"
            #                         )

            # import boto3

            # client_br = boto3.client('bedrock-runtime',
            #                         aws_access_key_id=question.llm_key.aws_secret_access_key,
            #                         aws_secret_access_key=question.llm_key.aws_secret_access_key,
            #                         region_name=question.llm_key.region_name
            #                         )
            # session = boto3.Session(aws_access_key_id=question.llm_key.aws_secret_access_key,
            #                        aws_secret_access_key=question.llm_key.aws_secret_access_key,
            #                        region_name=question.llm_key.region_name
            #                        )
            # client_ss = session.client("bedrock-runtime")

            llm = ChatBedrockConverse(
                # client=client_br,
                model=question.model,
                temperature=question.temperature,
                max_tokens=question.max_tokens,
                region_name=question.llm_key.region_name

                #                                 base_url="http://bedroc-proxy-paacejvmzcgv-121947512.eu-central-1.elb.amazonaws.com/api/v1/",

            )  # model_kwargs={"temperature": 0.001},

            # print(chat_model.client._get_credentials().access_key)


        else:
            callback_handler = OpenAICallbackHandler()
            llm_embeddings = OpenAIEmbeddings(api_key=question.gptkey, model=question.embedding)
            llm = ChatOpenAI(api_key=question.gptkey,
                             model=question.model,
                             temperature=question.temperature,
                             max_tokens=question.max_tokens,
                             callbacks=[callback_handler]
                             )

        # Add chat_model agli kwargs
        kwargs['llm'] = llm
        kwargs['callback_handler'] = callback_handler
        kwargs['llm_embeddings'] = llm_embeddings

        return await func(question, *args, **kwargs)

    return wrapper

def inject_llm_o1(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
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


