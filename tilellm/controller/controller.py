import uuid
from datetime import datetime
from typing import List

import fastapi
import asyncio
from fastapi.responses import JSONResponse

from langchain.chains import ConversationalRetrievalChain, LLMChain  # Deprecata

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate #, SystemMessagePromptTemplate

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from fastapi.responses import StreamingResponse
from langgraph.prebuilt import create_react_agent

from tilellm.controller.controller_utils import preprocess_chat_history, \
    fetch_question_vectors, retrieve_documents, create_chains, get_or_create_session_history, \
    generate_answer_with_history, format_result, handle_exception, initialize_retrievers, create_chains_deepseek, \
    _create_event, extract_conversation_flow, create_contextualize_query
from tilellm.models.schemas import (RetrievalResult,
                                    IndexingResult,
                                       RepositoryNamespaceResult,
                                       RepositoryDescNamespaceResult,
                                       RepositoryItems,
                                       SimpleAnswer,
                                       RepositoryItem,
                                       RepositoryNamespace,
                                       RepositoryEngine,ReasoningAnswer, RetrievalChunksResult)
from tilellm.models import (ChatEntry,
                            QuestionToAgent,
                            QuestionToLLM,
                            QuestionAnswer
                            )

from tilellm.shared.utility import inject_repo, inject_llm, inject_llm_chat, inject_reason_llm, inject_repo_async, \
    inject_llm_chat_async, inject_llm_async, inject_reason_llm_async

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from tilellm.agents.shopify_agent import lookup as shopify_lookup_agent

from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage

)

import logging

from tilellm.tools.reranker import RerankedRetriever, TileReranker

logger = logging.getLogger(__name__)

@inject_repo_async
@inject_llm_chat_async
async def ask_hybrid_with_memory(question_answer, repo=None, llm=None, callback_handler=None, llm_embeddings=None):
    try:
        logger.info(question_answer)

        # Preprocess chat history
        chat_history_list, question_answer_list = preprocess_chat_history(question_answer)
        # Initialize embeddings and encoders
        emb_dimension, sparse_encoder, index = await repo.initialize_embeddings_and_index(question_answer,
                                                                                          llm_embeddings)
        # Fetch vectors for the given question
        dense_vector, sparse_vector = await fetch_question_vectors(question_answer, sparse_encoder, llm_embeddings)
        ### Modifiche
        # Perform hybrid search - modifica per recuperare più documenti se necessario
        search_top_k = question_answer.top_k * question_answer.reranking_multiplier if question_answer.reranking else question_answer.top_k

        # Temporaneamente modifica top_k per la ricerca
        original_top_k = question_answer.top_k
        question_answer.top_k = search_top_k

        # Perform hybrid search
        results = await repo.perform_hybrid_search(question_answer, index, dense_vector, sparse_vector)

        # Ripristina il valore originale
        question_answer.top_k = original_top_k

        ### Fine modifiche ORIG: results = await repo.perform_hybrid_search(question_answer, index, dense_vector, sparse_vector)
        # Retrieve documents based on search results
        if question_answer.reranking:
            contextualize_query = await create_contextualize_query(llm,question_answer)
        else:
            contextualize_query= question_answer.question

        retriever = retrieve_documents(question_answer, results,contextualize_query)

        # Create chains for contextualization and Q&A
        history_aware_retriever, question_answer_chain, qa_prompt = await create_chains(llm, question_answer, retriever)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Load session history and prepare conversational chain
        store = {}
        get_session_history = lambda session_id: get_or_create_session_history(store, session_id,
                                                                                   question_answer.chat_history_dict)
        # Generate the final answer, with or without citations result, citations, success
        result_to_return = await generate_answer_with_history(llm=llm,
                                                              question_answer=question_answer,
                                                              rag_chain = rag_chain,
                                                              retriever = retriever,
                                                              get_session_history = get_session_history,
                                                              qa_prompt=qa_prompt,
                                                              callback_handler=callback_handler,
                                                              question_answer_list=question_answer_list
                                                              )

        #await index.close()


        return result_to_return
    except Exception as e:
        return handle_exception(e, question_answer)


@inject_reason_llm_async
async def ask_reason_llm(question, chat_model=None):
    try:
        logger.info(question)
        chat_history_list = []

        if question.chat_history_dict is not None:
            for key, entry in question.chat_history_dict.items():
                chat_history_list.append(HumanMessage(content=entry.question))  # ('human', entry.question))
                chat_history_list.append(AIMessage(content=entry.answer))

        qa_prompt = ChatPromptTemplate.from_messages(
            [   MessagesPlaceholder("chat_history", n_messages=question.n_messages),
                ("human", "{input}")
            ]
        )

        store = {}
        get_session_history = lambda session_id: get_or_create_session_history(store, session_id,
                                                                               question.chat_history_dict)

        runnable = qa_prompt | chat_model

        runnable_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"

        )

        if question.stream:
            # return runnable_with_history
            async def get_stream_llm():
                full_response = ""
                full_response_reasoning= ""
                message_id = str(uuid.uuid4())
                start_time = datetime.now()

                yield _create_event("metadata", {
                    "message_id": message_id,
                    "status": "started",
                    "timestamp": start_time.isoformat()
                })

                async for chunk in runnable_with_history.astream({"input": question.question},
                                                                 config={
                                                                     "configurable": {"session_id": uuid.uuid4().hex}}):

                    if hasattr(chunk, 'content'):

                        is_reasoning_content, content_text, re_content= get_reasoning_content(chunk, question.llm)
                        full_response += content_text
                        if is_reasoning_content:
                           full_response_reasoning += re_content  # chunk.additional_kwargs.reasoning_content
                           yield _create_event("chunk", {"reasoning_content": re_content,
                                                          "message_id": message_id})
                        else:
                           yield _create_event("chunk", {"content": content_text, "message_id": message_id})

                        await asyncio.sleep(0.02)  # Per un flusso più regolare


                end_time = datetime.now()

                if not question.chat_history_dict:
                    question.chat_history_dict = {}

                num_question = len(question.chat_history_dict.keys())
                question.chat_history_dict[str(num_question)] = {"question": question.question, "answer": full_response}

                simple_reasoning_answer = ReasoningAnswer(answer=full_response,
                                                reasoning_content= full_response_reasoning,
                                                chat_history_dict=question.chat_history_dict)
                yield _create_event("metadata", {
                    "message_id": message_id,
                    "status": "completed",
                    "timestamp": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "model_used": simple_reasoning_answer.model_dump()  # Sostituire con calcolo reale dei token
                })

            return StreamingResponse(
                get_stream_llm(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            logger.info(question)

            result = await runnable_with_history.ainvoke(
                {"input": question.question},  # 'chat_history_a': chat_history_list,
                config={"configurable": {"session_id": uuid.uuid4().hex}
                        },
            )
            # logger.info(result)

            if not question.chat_history_dict:
                question.chat_history_dict = {}


            _, content, reasoning_content=get_reasoning_content(result, question.llm)
            num = len(question.chat_history_dict.keys())
            question.chat_history_dict[str(num)] = {"question": question.question, "answer": content}
            return JSONResponse(
                content=ReasoningAnswer(answer=content,
                                        reasoning_content=reasoning_content,
                                        chat_history_dict=question.chat_history_dict).model_dump())


    except Exception as e:
        import traceback
        traceback.print_exc()

        result_to_return = SimpleAnswer(answer=repr(e),
                                        chat_history_dict={})
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())


@inject_llm_async
async def ask_to_llm(question: QuestionToLLM, chat_model=None) :
    try:
        logger.info(question)
        chat_history_list = []

        if question.chat_history_dict is not None:
            for key, entry in question.chat_history_dict.items():
                chat_history_list.append(HumanMessage(content=entry.question))  # ('human', entry.question))
                chat_history_list.append(AIMessage(content=entry.answer))



        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", question.system_context),
                MessagesPlaceholder("chat_history", n_messages=question.n_messages),
                ("human", "{input}"),
            ]
        )

        store = {}
        get_session_history = lambda session_id: get_or_create_session_history(store, session_id,
                                                                               question.chat_history_dict)



        runnable = qa_prompt | chat_model

        runnable_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"

        )

        if question.stream:

            #return runnable_with_history
           async def get_stream_llm():
               full_response = ""
               message_id = str(uuid.uuid4())
               start_time = datetime.now()

               yield _create_event("metadata", {
                   "message_id": message_id,
                   "status": "started",
                   "timestamp": start_time.isoformat()
               })

               async for chunk in runnable_with_history.astream({"input": question.question},
                                                                config={
                                                                    "configurable": {"session_id": uuid.uuid4().hex}}):
                   if hasattr(chunk, 'content'):
                       full_response += chunk.content
                       yield _create_event("chunk", {"content": chunk.content, "message_id": message_id})
                       await asyncio.sleep(0.02)  # Per un flusso più regolare

               end_time = datetime.now()

               if not question.chat_history_dict:
                   question.chat_history_dict = {}

               num_question = len(question.chat_history_dict.keys())
               question.chat_history_dict[str(num_question)] = {"question": question.question, "answer": full_response}

               simple_answer = SimpleAnswer(answer=full_response, chat_history_dict=question.chat_history_dict)
               yield _create_event("metadata", {
                   "message_id": message_id,
                   "status": "completed",
                   "timestamp": end_time.isoformat(),
                   "duration": (end_time - start_time).total_seconds(),
                   "full_response": full_response,
                   "model_used": simple_answer.model_dump() # Sostituire con calcolo reale dei token
               })

           return StreamingResponse(
               get_stream_llm(),
               media_type="text/event-stream",
               headers={"Cache-Control": "no-cache"}
           )


        else:
            result = await runnable_with_history.ainvoke(
                {"input": question.question}, # 'chat_history_a': chat_history_list,
                config={"configurable": {"session_id": uuid.uuid4().hex}
                        },
            )
            # logger.info(result)

            if not question.chat_history_dict:
                question.chat_history_dict = {}

            num = len(question.chat_history_dict.keys())
            question.chat_history_dict[str(num)] = {"question": question.question, "answer": result.content}


            return JSONResponse(content=SimpleAnswer(answer=result.content, chat_history_dict=question.chat_history_dict).model_dump())


    except Exception as e:
        import traceback
        traceback.print_exc()

        result_to_return = SimpleAnswer(answer=repr(e),
                                        chat_history_dict={})
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())


@inject_llm_async
async def ask_mcp_agent_llm(question: QuestionToLLM, chat_model=None):
    mcp_client = question.create_mcp_client()

    try:
        tools = await mcp_client.get_tools()

        agent = create_react_agent(chat_model, tools)
        response = await agent.ainvoke({"messages": question.question})
        logger.info(response)
        result = extract_conversation_flow(response['messages'])
        logger.info(result)
        return JSONResponse(
            content=SimpleAnswer(answer=result, chat_history_dict={}).model_dump())

    except Exception as e:
        return handle_exception(e, "Exception in MCP dialog")


@inject_repo_async
@inject_llm_chat_async
async def ask_with_memory(question_answer, repo=None, llm=None, callback_handler=None, llm_embeddings=None) -> RetrievalResult:
    """
    Ask to LLM your questions
    :param question_answer:
    :param repo:
    :param llm:
    :param callback_handler:
    :param llm_embeddings:
    :return: RetrievalResult
    """
    try:

        logger.info(question_answer)

        # Preprocess chat history
        chat_history_list, question_answer_list = preprocess_chat_history(question_answer)

        # Modifiche
        # Initialize embeddings and retrievers (con supporto per re-ranking)


        base_retriever = await initialize_retrievers(question_answer, repo, llm_embeddings)

        # Wrap con RerankedRetriever se il re-ranking è abilitato
        if question_answer.reranking:
            contextualize_query = await create_contextualize_query(llm,question_answer)

            reranker = TileReranker(model_name=question_answer.reranker_model)
            retriever = RerankedRetriever(base_retriever=base_retriever,
                                          reranker=reranker,
                                          top_k=question_answer.top_k,
                                          use_reranking=question_answer.reranking,
                                          contextualize_query=contextualize_query)

        else:
            retriever = base_retriever

        # Create chains for contextualization and Q&A
        history_aware_retriever, question_answer_chain, qa_prompt = await create_chains(llm, question_answer, retriever)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Load session history and prepare conversational chain
        store = {}
        get_session_history = lambda session_id: get_or_create_session_history(store, session_id,
                                                                               question_answer.chat_history_dict)

        # Generate the final answer, with or without citations
        result_to_return = await generate_answer_with_history(llm=llm,
                                                              question_answer=question_answer,
                                                              rag_chain=rag_chain,
                                                              retriever=retriever,
                                                              get_session_history=get_session_history,
                                                              qa_prompt=qa_prompt,
                                                              callback_handler=callback_handler,
                                                              question_answer_list=question_answer_list
                                                              )

        return result_to_return
    except Exception as e:
        return handle_exception(e, question_answer)

@inject_repo_async
async def ask_for_chunks(question_answer:QuestionAnswer, repo=None) -> RetrievalChunksResult:
    """
    Ask to LLM your questions
    :param question_answer:
    :param repo:
    :return: RetrievalResult
    """
    try:

        logger.info(question_answer)

        # Generate the final answer, with or without citations
        result_to_return = await repo.get_chunks_from_repo(question_answer)

        #llm_embeddings = None
        return result_to_return
    except Exception as e:
        return handle_exception(e, question_answer)

@inject_llm_async
async def ask_to_agent(question_to_agent: QuestionToAgent, chat_model=None):
    try:
        logger.info(question_to_agent)
        #chat_history_list = []
        #tools = load_tools(
        #    question_to_agent.tools,
        #    endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index",
        #    api_key="ciccio"
        #)
        #agent = initialize_agent(
        #    tools, chat_model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        #)
        #result = agent.invoke(question_to_agent.question)
        result_history = ""
        if question_to_agent.chat_history_dict is not None:
            #for key, entry in question_to_agent.chat_history_dict.items():
            #    chat_history_list.append(HumanMessage(content=entry.question))  # ('human', entry.question)
            #    chat_history_list.append(AIMessage(content=entry.answer))
            # "chat_history": "Human: My name is Bob\\nAI: Hello Bob!",

            for i in range(len(question_to_agent.chat_history_dict)):
                entry = question_to_agent.chat_history_dict[str(i)]
                result_history += f"Human: {entry.question}\n"
                result_history += f"AI: {entry.answer}\n"

            result_history = result_history.strip()  # Remove trailing newline
            print(result_history)


        #qa_prompt = ChatPromptTemplate.from_messages(
        #    [
        #        ("system", question_to_agent.system_context),
        #        MessagesPlaceholder("tools_result"),
        #        MessagesPlaceholder("chat_history", n_messages=question_to_agent.n_messages),
        #        ("human", "{input}"),
        #    ]
        #)

        #store = {}

        #def get_session_history(session_id: str) -> BaseChatMessageHistory:
        #    if session_id not in store:
        #        store[session_id] = load_session_history(question_to_agent.chat_history_dict)  # ChatMessageHistory()
        #    return store[session_id]

        result_shopify = shopify_lookup_agent(question_to_agent=question_to_agent, chat_model=chat_model, chat_history=result_history)
        # print(f"RESULT: {result_shopify.get('output')} type: {type(result_shopify.get('output'))}")


        if not question_to_agent.chat_history_dict:
            question_to_agent.chat_history_dict = {}

        num = len(question_to_agent.chat_history_dict.keys())
        question_to_agent.chat_history_dict[str(num)] = dict({"question": question_to_agent.question, "answer": result_shopify.get("output")})

        answer_to_agent = SimpleAnswer(answer=result_shopify.get("output"), chat_history_dict=question_to_agent.chat_history_dict)
        # print(answer_to_agent)
        return answer_to_agent

    except Exception as e:
        import traceback
        traceback.print_exc()

        result_to_return = SimpleAnswer(answer=repr(e),
                                        chat_history_dict={})
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())


@inject_repo_async
async def ask_with_sequence(question_answer, repo=None) -> RetrievalResult:
    try:
        logger.info(question_answer)

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

        vector_store = await repo.create_index(oai_embeddings, emb_dimension, )
        idllmchain = get_idproduct_chain(llm)
        res = idllmchain.invoke(question_answer.question)

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': question_answer.top_k,
                                                                                       'namespace': question_answer.namespace})

        # mydocs = retriever.get_relevant_documents( question_answer.question)
        # from pprint import pprint
        # pprint(len(mydocs))

        if question_answer.system_context is not None and question_answer.system_context:

            sys_template = """{system_context}.

                              {context}
                           """

            sys_prompt = PromptTemplate.from_template(sys_template)


            crc = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": sys_prompt}
            )

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
        meta_id = ids[0]

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
            id=meta_id,
            prompt_token_size=prompt_token_size,
            success=success,
            error_message=None,
            chat_history_dict=chat_history_dict

        )

        return result_to_return
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


@inject_repo_async
async def add_item(item, repo=None) -> IndexingResult:
    """
    Add items to namespace
    :type repo: PineconeRepositoryBase
    :param item:
    :param repo:
    :return: PineconeIndexingResult
    """

    return await repo.add_item(item)


@inject_repo_async
async def add_item_hybrid(item, repo=None) -> IndexingResult:
    """

    :return:
    """
    return await repo.add_item_hybrid(item)


@inject_repo_async
async def delete_namespace(namespace_to_delete: RepositoryNamespace, repo=None):
    """
    Delete Namespace from index
    :param namespace_to_delete:
    :param repo:
    :return:
    """

    try:
        return await repo.delete_namespace(namespace_to_delete)
    except Exception as ex:
        raise ex


@inject_repo_async
async def delete_id_from_namespace(item_to_delete: RepositoryItem, metadata_id: str, namespace: str, repo=None):
    """
    Delete items from namespace
    :param item_to_delete: RepositoryItemToDelete
    :param metadata_id:
    :param namespace:
    :param repo:
    :return:
    """

    try:
        return await repo.delete_ids_namespace(engine=item_to_delete.engine, metadata_id=metadata_id,
                                               namespace=namespace)
    except Exception as ex:
        logger.error(ex)
        raise ex

@inject_repo_async
async def delete_chunk_id_from_namespace(repository_engine: RepositoryEngine, chunk_id:str, namespace: str, repo=None):
    """
    Delete chunk by id from namespace
    :param repository_engine: RepositoryEngine,
    :param chunk_id:
    :param namespace:
    :param repo:
    :return:
    """
    try:
        return await repo.delete_chunk_id_namespace(engine=repository_engine.engine,
                                                    chunk_id=chunk_id,
                                                    namespace=namespace)
    except Exception as ex:
        logger.error(ex)
        raise ex


@inject_repo_async
async def get_list_namespace(repository_engine: RepositoryEngine, repo=None) -> RepositoryNamespaceResult:
    """
    Get list namespaces with namespace id and vector count
    :param repository_engine: RepositoryEngine
    :param repo:
    :return: list of all namespaces in index
    """
    # from tilellm.store.pinecone_repository import pinecone_list_namespaces
    try:
        return await repo.list_namespaces(engine=repository_engine.engine)
    except Exception as ex:
        raise ex


@inject_repo_async
async def get_ids_namespace(repository_engine: RepositoryEngine, metadata_id: str, namespace: str, repo=None) -> RepositoryItems:
    """
    Get all items from namespace given id
    :param repository_engine: RepositoryEngine
    :param metadata_id:
    :param namespace:
    :param repo:
    :return:
    """
    try:
        return await repo.get_ids_namespace(engine=repository_engine.engine, metadata_id=metadata_id,
                                            namespace=namespace)
    except Exception as ex:
        raise ex


@inject_repo_async
async def get_listitems_namespace(repository_engine: RepositoryEngine, namespace: str, repo=None) -> RepositoryItems:
    """
    Get all items from given namespace
    :param repository_engine: RepositoryEngine
    :param namespace: namespace_id
    :param repo:
    :return: list of al items PineconeItems
    """
    try:
        return await repo.get_all_obj_namespace(engine=repository_engine.engine,
                                                   namespace=namespace)
    except Exception as ex:
        raise ex


@inject_repo_async
async def get_desc_namespace(repository_engine: RepositoryEngine, namespace: str, repo=None) -> RepositoryDescNamespaceResult:
    """
    Desc of Namespace
    :param repository_engine:
    :param namespace:
    :param repo:
    :return:
    """
    try:
        return await repo.get_desc_namespace(engine=repository_engine.engine, namespace=namespace)
    except Exception as ex:
        raise ex


@inject_repo_async
async def get_sources_namespace(repository_engine: RepositoryEngine, source: str, namespace: str, repo=None) -> RepositoryItems:
    """
    Get all item from namespace given source
    :param repository_engine: RepositoryEngine,
    :param source:
    :param namespace:
    :param repo:
    :return:
    """

    try:
        return await repo.get_sources_namespace(engine=repository_engine.engine, source=source, namespace=namespace)
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


def load_session_history(history) -> BaseChatMessageHistory:
    chat_history = ChatMessageHistory()
    if history is not None:
        for key, entry in history.items():
            chat_history.add_message(HumanMessage(content=entry.question))  # ('human', entry.question))
            chat_history.add_message(AIMessage(content=entry.answer))
    return chat_history


def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Source: {doc.metadata['source']}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)

def get_reasoning_content(chunk, llm):
    """
    Verifica se la chiave 'reasoning_content' esiste nel dizionario annidato
    'additional_kwargs' di un chunk e restituisce il suo valore.

    Args:
        chunk (dict): Il dizionario contenente i dati.
        llm: llm usato

    Returns:
        str or None: Il valore di 'reasoning_content' se esiste, altrimenti None.
    """
    if llm=="openai":
        return False, chunk.content, ''
    elif llm=="deepseek":
        if 'reasoning_content' in chunk.additional_kwargs:
            return True, chunk.content, chunk.additional_kwargs['reasoning_content']
        else:
            return False, chunk.content, ''
    elif llm=="anthropic":
        full_thinking=""
        full_text=""
        is_thinking = False
        if not chunk.content:  # Controlla se la lista è vuota
            return False, full_text, full_thinking
        for text_element in chunk.content:
            if 'thinking' in text_element:
                full_thinking += text_element['thinking']
                is_thinking = True
            if 'text' in text_element:
                full_text += text_element['text']
                is_thinking = False

        return is_thinking, full_text, full_thinking

    else:
        return False, '', ''




