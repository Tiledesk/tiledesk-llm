import uuid

import fastapi
from langchain.chains import ConversationalRetrievalChain, LLMChain # Deprecata
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
# from tilellm.store.pinecone_repository import add_pc_item as pinecone_add_item
# from tilellm.store.pinecone_repository import create_pc_index, get_embeddings_dimension
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from tilellm.models.item_model import RetrievalResult, ChatEntry
from tilellm.shared.utility import inject_repo
import tilellm.shared.const as const
# from tilellm.store.pinecone_repository_base import PineconeRepositoryBase

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

import logging

logger = logging.getLogger(__name__)


@inject_repo
async def ask_with_memory1(question_answer, repo=None):
    
    try:
        logger.info(question_answer)
        # question = str
        # namespace: str
        # gptkey: str
        # model: str =Field(default="gpt-3.5-turbo")
        # temperature: float = Field(default=0.0)
        # top_k: int = Field(default=5)
        # max_tokens: int = Field(default=128)
        # system_context: Optional[str]
        # chat_history_dict : Dict[str, ChatEntry]
        
        question_answer_list = []
        if question_answer.chat_history_dict is not None:
            for key, entry in question_answer.chat_history_dict.items():
                question_answer_list.append((entry.question, entry.answer))

        logger.info(question_answer_list)
        openai_callback_handler = OpenAICallbackHandler()
        
        llm = ChatOpenAI(model_name=question_answer.model, 
                         temperature=question_answer.temperature,
                         openai_api_key=question_answer.gptkey,
                         max_tokens=question_answer.max_tokens,
                         callbacks=[openai_callback_handler])

        emb_dimension = repo.get_embeddings_dimension(question_answer.embedding)
        oai_embeddings = OpenAIEmbeddings(api_key=question_answer.gptkey, model=question_answer.embedding) 
        
        vector_store = await repo.create_pc_index(oai_embeddings, emb_dimension)

        retriever = vector_store.as_retriever(search_type='similarity',
                                              search_kwargs={'k': question_answer.top_k,
                                                             'namespace': question_answer.namespace}
                                              )
        # Query on store for relevant document, returned by callback
        # mydocs = retriever.get_relevant_documents( question_answer.question)
        # from pprint import pprint
        # pprint(len(mydocs))

        if question_answer.system_context is not None and question_answer.system_context:
            print("blocco if")
            from langchain.chains import LLMChain

            # prompt_template = "Tell me a {adjective} joke"
            # prompt = PromptTemplate(
            #    input_variables=["adjective"], template=prompt_template
            # )
            # llm = LLMChain(llm=OpenAI(), prompt=prompt)
            sys_template = """{system_context}.
                              
                              {context}
                           """

            sys_prompt = PromptTemplate.from_template(sys_template)

            # llm_chain = LLMChain(llm=llm, prompt=prompt)
            crc = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": sys_prompt}
                )
            # from pprint import pprint
            # pprint(crc.combine_docs_chain.llm_chain.prompt.messages)
            # crc.combine_docs_chain.llm_chain.prompt.messages[0]=SystemMessagePromptTemplate.from_template(sys_prompt)
            
            result = crc.invoke({'question': question_answer.question,
                                 'system_context': question_answer.system_context,
                                 'chat_history': question_answer_list}
                                )
           
        else:
            print("blocco else")
            #PromptTemplate.from_template()
            crc = ConversationalRetrievalChain.from_llm(llm=llm,
                                                        retriever=retriever,
                                                        return_source_documents=True,
                                                        verbose=True)


            # 'Use the following pieces of context to answer the user\'s question. If you don\'t know the answer, just say that you don\'t know, don\'t try to make up an answer.',
            result = crc.invoke({'question': question_answer.question,
                                 'chat_history': question_answer_list}
                                )

        docs = result["source_documents"]
        from pprint import pprint
        pprint(result)

        ids = []
        sources = []
        for doc in docs:
            ids.append(doc.metadata['id'])
            sources.append(doc.metadata['source'])

        ids = list(set(ids))
        sources = list(set(sources))
        source = " ".join(sources)
        metadata_id = ids[0]

        logger.info(result)

        question_answer_list.append((result['question'], result['answer']))
        
        chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
        chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

        success = bool(openai_callback_handler.successful_requests)
        prompt_token_size = openai_callback_handler.total_tokens

        result_to_return = RetrievalResult(
            answer=result['answer'],
            namespace=question_answer.namespace,
            sources=sources,
            ids=ids,
            source=source,
            id=metadata_id,
            prompt_token_size=prompt_token_size,
            success=success,
            error_message=None,
            chat_history_dict=chat_history_dict
        )

        return result_to_return.dict()
    except Exception as e:
        import traceback 
        traceback.print_exc() 
        question_answer_list = []
        if question_answer.chat_history_dict is not None:
            for key, entry in question_answer.chat_history_dict.items():
                question_answer_list.append((entry.question, entry.answer))
        chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
        chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

        result_to_return = RetrievalResult(
            namespace=question_answer.namespace,
            error_message=repr(e),
            chat_history_dict=chat_history_dict
        )
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())


@inject_repo
async def ask_with_memory(question_answer, repo=None):
    try:
        logger.info(question_answer)
        # question = str
        # namespace: str
        # gptkey: str
        # model: str =Field(default="gpt-3.5-turbo")
        # temperature: float = Field(default=0.0)
        # top_k: int = Field(default=5)
        # max_tokens: int = Field(default=128)
        # system_context: Optional[str]
        # chat_history_dict : Dict[str, ChatEntry]

        question_answer_list = []
        if question_answer.chat_history_dict is not None:
            for key, entry in question_answer.chat_history_dict.items():
                question_answer_list.append((entry.question, entry.answer))

        logger.info(question_answer_list)
        openai_callback_handler = OpenAICallbackHandler()

        llm = ChatOpenAI(model_name=question_answer.model,
                         temperature=question_answer.temperature,
                         openai_api_key=question_answer.gptkey,
                         max_tokens=question_answer.max_tokens,
                         callbacks=[openai_callback_handler])

        emb_dimension = repo.get_embeddings_dimension(question_answer.embedding)
        oai_embeddings = OpenAIEmbeddings(api_key=question_answer.gptkey, model=question_answer.embedding)

        vector_store = await repo.create_pc_index(oai_embeddings, emb_dimension)

        retriever = vector_store.as_retriever(search_type='similarity',
                                              search_kwargs={'k': question_answer.top_k,
                                                             'namespace': question_answer.namespace}
                                              )

        if question_answer.system_context is not None and question_answer.system_context:

            # Contextualize question
            contextualize_q_system_prompt = const.contextualize_q_system_prompt
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

            # Answer question
            qa_system_prompt = question_answer.system_context
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            store = {}

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in store:
                    store[session_id] = ChatMessageHistory()
                return store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            result = conversational_rag_chain.invoke(
                {"input": question_answer.question, 'chat_history': question_answer_list},
                config={"configurable": {"session_id": uuid.uuid4().hex}
                        },  # constructs a key "abc123" in `store`.
            )

        else:
            # Contextualize question
            contextualize_q_system_prompt = const.contextualize_q_system_prompt
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

            # Answer question
            qa_system_prompt = const.qa_system_prompt
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            store = {}

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in store:
                    store[session_id] = ChatMessageHistory()
                return store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            result = conversational_rag_chain.invoke(
                {"input": question_answer.question, 'chat_history': question_answer_list},
                config={"configurable": {"session_id": uuid.uuid4().hex}
                        },  # constructs a key "abc123" in `store`.
                )

            # print(store)
            # print(question_answer_list)

        docs = result["context"]
        from pprint import pprint
        pprint(docs)

        ids = []
        sources = []
        for doc in docs:
            ids.append(doc.metadata['id'])
            sources.append(doc.metadata['source'])

        ids = list(set(ids))
        sources = list(set(sources))
        source = " ".join(sources)
        metadata_id = ids[0]

        logger.info(result)
        print(result['answer'])
        result['answer'], success = verify_answer(result['answer'])

        question_answer_list.append((result['input'], result['answer']))

        chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
        chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}



        # success = bool(openai_callback_handler.successful_requests)
        prompt_token_size = openai_callback_handler.total_tokens

        result_to_return = RetrievalResult(
            answer=result['answer'],
            namespace=question_answer.namespace,
            sources=sources,
            ids=ids,
            source=source,
            id=metadata_id,
            prompt_token_size=prompt_token_size,
            success=success,
            error_message=None,
            chat_history_dict=chat_history_dict
        )

        return result_to_return.dict()
    except Exception as e:
        import traceback
        traceback.print_exc()
        question_answer_list = []
        if question_answer.chat_history_dict is not None:
            for key, entry in question_answer.chat_history_dict.items():
                question_answer_list.append((entry.question, entry.answer))
        chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
        chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

        result_to_return = RetrievalResult(
            namespace=question_answer.namespace,
            error_message=repr(e),
            chat_history_dict=chat_history_dict
        )
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())

@inject_repo
async def ask_with_sequence(question_answer, repo=None):
    try:
        logger.info(question_answer)
        # question = str
        # namespace: str
        # gptkey: str
        # model: str =Field(default="gpt-3.5-turbo")
        # temperature: float = Field(default=0.0)
        # top_k: int = Field(default=5)
        # max_tokens: int = Field(default=128)
        # system_context: Optional[str]
        # chat_history_dict : Dict[str, ChatEntry]

        question_answer_list = []
        if question_answer.chat_history_dict is not None:
            for key, entry in question_answer.chat_history_dict.items():
                question_answer_list.append((entry.question, entry.answer))

        logger.info(question_answer_list)
        openai_callback_handler = OpenAICallbackHandler()

        llm = ChatOpenAI(model_name=question_answer.model,
                         temperature=question_answer.temperature,
                         openai_api_key=question_answer.gptkey,
                         max_tokens=question_answer.max_tokens,

                         callbacks=[openai_callback_handler])

        emb_dimension = repo.get_embeddings_dimension(question_answer.embedding)
        oai_embeddings = OpenAIEmbeddings(api_key=question_answer.gptkey, model=question_answer.embedding)

        vector_store = await repo.create_pc_index(oai_embeddings, emb_dimension)
        idllmchain = get_idproduct_chain(llm)
        res = idllmchain.invoke(question_answer.question)


        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': question_answer.top_k,
                                                                                       'namespace': question_answer.namespace})

        # mydocs = retriever.get_relevant_documents( question_answer.question)
        # from pprint import pprint
        # pprint(len(mydocs))

        if question_answer.system_context is not None and question_answer.system_context:
            from langchain.chains import LLMChain

            # prompt_template = "Tell me a {adjective} joke"
            # prompt = PromptTemplate(
            #    input_variables=["adjective"], template=prompt_template
            # )
            # llm = LLMChain(llm=OpenAI(), prompt=prompt)
            sys_template = """{system_context}.

                              {context}
                           """

            sys_prompt = PromptTemplate.from_template(sys_template)

            # llm_chain = LLMChain(llm=llm, prompt=prompt)
            crc = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": sys_prompt}
            )
            # from pprint import pprint
            # pprint(crc.combine_docs_chain.llm_chain.prompt.messages)
            # crc.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(sys_prompt)

            result = crc.invoke({'question': question_answer.question, 'system_context': question_answer.system_context,
                                 'chat_history': question_answer_list})

        else:
            crc = ConversationalRetrievalChain.from_llm(llm=llm,
                                                        retriever=retriever,
                                                        return_source_documents=True)

            result = crc.invoke({'question': res.get('text'), 'chat_history': question_answer_list})

        docs = result["source_documents"]

        ids = []
        sources = []
        for doc in docs:
            ids.append(doc.metadata['id'])
            sources.append(doc.metadata['source'])
            print(doc)

        ids = list(set(ids))
        sources = list(set(sources))
        source = " ".join(sources)
        id = ids[0]

        logger.info(result)

        question_answer_list.append((result['question'], result['answer']))

        chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
        chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

        success = bool(openai_callback_handler.successful_requests)
        prompt_token_size = openai_callback_handler.total_tokens

        result_to_return = RetrievalResult(
            answer=result['answer'],
            namespace=question_answer.namespace,
            sources=sources,
            ids=ids,
            source=source,
            id=id,
            prompt_token_size=prompt_token_size,
            success=success,
            error_message=None,
            chat_history_dict=chat_history_dict

        )

        return result_to_return.dict()
    except Exception as e:
        import traceback
        traceback.print_exc()
        question_answer_list = []
        if question_answer.chat_history_dict is not None:
            for key, entry in question_answer.chat_history_dict.items():
                question_answer_list.append((entry.question, entry.answer))
        chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
        chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

        result_to_return = RetrievalResult(
            namespace=question_answer.namespace,
            error_message=repr(e),
            chat_history_dict=chat_history_dict

        )
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())


@inject_repo
async def add_pc_item(item, repo=None):
    """
    Add items to namespace
    :type repo: PineconeRepositoryBase
    :param item:
    :param repo:
    :return:
    """
    return await repo.add_pc_item(item)


@inject_repo
async def delete_namespace(namespace: str, repo=None):
    """
    Delete Namespace from index
    :param namespace:
    :param repo:
    :return:
    """
    # from tilellm.store.pinecone_repository import delete_pc_namespace
    try:
        return await repo.delete_pc_namespace(namespace)
    except Exception as ex:
        raise ex


@inject_repo
async def delete_id_from_namespace(metadata_id: str, namespace: str, repo=None):
    """
    Delete items from namespace
    :param metadata_id:
    :param namespace:
    :param repo:
    :return:
    """
    # from tilellm.store.pinecone_repository import delete_pc_ids_namespace # , delete_pc_ids_namespace1
    try:
        return await repo.delete_pc_ids_namespace(metadata_id=metadata_id, namespace=namespace)
    except Exception as ex:
        logger.error(ex)
        raise ex


@inject_repo
async def get_list_namespace(repo=None):
    """
    Get list namespaces with namespace id and vector count
    :param repo:
    :return: list of all namespaces in index
    """
    # from tilellm.store.pinecone_repository import pinecone_list_namespaces
    try:
        return await repo.pinecone_list_namespaces()
    except Exception as ex:
        raise ex


@inject_repo
async def get_ids_namespace(metadata_id: str, namespace: str, repo=None):
    """
    Get all items from namespace given id
    :param metadata_id:
    :param namespace:
    :param repo:
    :return:
    """
    # from tilellm.store.pinecone_repository import get_pc_ids_namespace
    try:
        return await repo.get_pc_ids_namespace(metadata_id=metadata_id, namespace=namespace)
    except Exception as ex:
        raise ex


@inject_repo
async def get_listitems_namespace(namespace: str, repo=None):
    """
    Get all items from given namespace
    :param namespace: namespace_id
    :param repo:
    :return: list of al items
    """
    # from tilellm.store.pinecone_repository import get_pc_all_obj_namespace
    try:
        return await repo.get_pc_all_obj_namespace(namespace=namespace)
    except Exception as ex:
        raise ex


@inject_repo
async def get_sources_namespace(source: str, namespace: str, repo=None):
    """
    Get all item from namespace given source
    :param source:
    :param namespace:
    :param repo:
    :return:
    """
    # from tilellm.store.pinecone_repository import get_pc_sources_namespace
    try:
        return await repo.get_pc_sources_namespace(source=source, namespace=namespace)
    except Exception as ex:
        raise ex


def get_idproduct_chain(llm) -> LLMChain:
    summary_template = """
         I want the product Identifier from this phrase (remember, it's composed by 5 digit like 36400. Ignore the other informations). Give me only the number. {question}.
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["question"],
        template=summary_template,
    )

    return LLMChain(llm=llm, prompt=summary_prompt_template)


def verify_answer(s):
    if s.endswith("<NOANS>"):
        s = s[:-7]  # Rimuove <NOANS> dalla fine della stringa
        success = False
    else:
        success = True
    return s, success
