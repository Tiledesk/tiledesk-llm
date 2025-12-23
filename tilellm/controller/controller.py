import base64
import uuid
import json
from datetime import datetime
from typing import List, Any, Optional

import re

from langchain_core.messages import ToolMessage

import fastapi
import asyncio
from fastapi.responses import JSONResponse

from langchain_core.documents import Document

from fastapi.responses import StreamingResponse
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

from tilellm.controller.controller_utils import preprocess_chat_history, \
    fetch_question_vectors, retrieve_documents, create_chains, get_or_create_session_history, \
    generate_answer_with_history, handle_exception, initialize_retrievers, _create_event, extract_conversation_flow, \
    create_contextualize_query, get_all_filtered_tools
from tilellm.controller.helpers import _get_question_list
from tilellm.models.schemas import (
    RetrievalResult,
    IndexingResult,
       RepositoryNamespaceResult,
       RepositoryDescNamespaceResult,
       RepositoryItems,
       RepositoryItem,
       RepositoryNamespace,
       RepositoryEngine,ReasoningAnswer, RetrievalChunksResult)
from tilellm.models import (
    ChatEntry,
    QuestionToLLM,
    QuestionAnswer,
    SimpleAnswer,
    PromptTokenInfo
    )

from tilellm.shared.utility import inject_repo_async, \
    inject_llm_chat_async, inject_llm_async, inject_reason_llm_async

from tilellm.shared.mcp_prompt import MCP_BASE64_MANAGEMENT_TEMPLATE, \
    MCP_DOC_HEADER_TEMPLATE, MCP_DOC_INSTRUCTIONS_TEMPLATE, MCP_INTERNAL_TOOL_TEMPLATE

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_retrieval_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

import logging

from tilellm.tools.reranker import RerankedRetriever, TileReranker

logger = logging.getLogger(__name__)


def handle_agent_exception(e: Exception, context: str = "Agent execution") -> JSONResponse:
    """
    Handles exceptions in agent functions and returns user‑friendly messages.

    Args:
        e (Exception): The caught exception.
        context (Any): Context information about the error for logging purposes.

    Returns:
        JSONResponse: A response containing a clear, user‑friendly error message.
    """
    import traceback
    traceback.print_exc()
    logger.error(f"Error in {context}: {str(e)}")

    # Determina il tipo di errore e crea un messaggio user-friendly
    error_message = str(e)
    user_message = ""
    status_code = 500

    # Errori di file non trovato
    if "not found" in error_message.lower() or "does not exist" in error_message.lower():
        user_message = "The requested file was not found. Please verify that the file was uploaded correctly."
        status_code = 404

    # Errori di tool/MCP
    elif "tool" in error_message.lower() and ("error" in error_message.lower() or "failed" in error_message.lower()):
        user_message = f"Error running the tool: {error_message}"
        status_code = 400

    # Errori di formato base64
    elif "base64" in error_message.lower() or "invalid" in error_message.lower():
        user_message = "Error processing the multimedia content. Please verify that the files are in the correct format."
        status_code = 400

    # Errori di modello/API
    elif "model" in error_message.lower() or "api" in error_message.lower() or "rate" in error_message.lower():
        user_message = f"Communication Error with the AI model. {error_message}"
        status_code = 503

    # Errori di timeout
    elif "timeout" in error_message.lower():
        user_message = "The operation took too long. Please try again with a simpler request."
        status_code = 504

    # Errore generico
    else:
        user_message = f"An error has occurred. {error_message}"
        status_code = 500

    error_response = SimpleAnswer(
        answer=user_message,
        tools_log=[],
        chat_history_dict={},
        prompt_token_info=PromptTokenInfo(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0
        )
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump()
    )


@inject_repo_async
@inject_llm_chat_async
async def ask_hybrid_with_memory(question_answer, repo=None, llm=None, callback_handler=None, llm_embeddings=None, emb_dimension=None, embedding_config_key=None):
    """
    Hybrid search
    :param question_answer:
    :param repo:
    :param llm:
    :param callback_handler:
    :param llm_embeddings:
    :param emb_dimension:
    :param embedding_config_key:
    :return:
    """
    try:
        logger.info(question_answer)

        # Preprocess chat history
        chat_history_list, question_answer_list = preprocess_chat_history(question_answer)
        # Initialize embeddings and encoders
        emb_dimension, sparse_encoder, index = await repo.initialize_embeddings_and_index(question_answer,
                                                                                            llm_embeddings,
                                                                                            emb_dimension,
                                                                                            embedding_config_key)
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
        def get_session_history(session_id):
            return get_or_create_session_history(store,
                                                 session_id,
                                                 question_answer.chat_history_dict
                                                 )
        # get_session_history = lambda session_id: get_or_create_session_history(store, session_id,
        #                                                                           question_answer.chat_history_dict)

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
    """
    Manages requests to LLMs with support for reasoning content (e.g., DeepSeek, gtp-5.1, gemini-2.5-pro, claude-4.5).
    Uses centralized private methods for streaming and history.
    :param question: QuestionToLLM
    :param chat_model: Il modello LLM
    :return: ReasoningAnswer in streaming o JSON
    """
    try:
        logger.info(question)

        # Costruisce il prompt template con history
        qa_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history", n_messages=question.n_messages),
            ("human", "{input}")
        ])

        # Setup session history
        store = {}
        #get_session_history = lambda session_id: get_or_create_session_history(
        #    store, session_id, question.chat_history_dict
        #)
        def get_session_history(session_id):
            return get_or_create_session_history(store,
                                                 session_id,
                                                 question.chat_history_dict
                                                 )

        # Crea il runnable con history
        runnable = qa_prompt | chat_model
        runnable_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        # Configurazione per il runnable
        config = {"configurable": {"session_id": uuid.uuid4().hex}}
        input_data = {"input": question.question}

        # --- STREAMING ---
        if question.stream:
            return StreamingResponse(
                _stream_generic_response(
                    runnable_with_history,
                    input_data,
                    question,
                    config=config,
                    chunk_processor=_reasoning_chunk_processor,
                    response_class=ReasoningAnswer
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )

        # --- RISPOSTA SINCRONA ---
        if question.structured_output and question.output_schema:
            structured_llm = chat_model.with_structured_output(question.output_schema)
            runnable = qa_prompt | structured_llm
            runnable_with_history = RunnableWithMessageHistory(
                runnable,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            result = await runnable_with_history.ainvoke(input_data, config=config)

            if hasattr(result, 'model_dump'):
                answer_content = result.model_dump()
            else:
                answer_content = result

            updated_history = _update_history(
                question.chat_history_dict,
                question.question,
                json.dumps(answer_content)
            )

            return JSONResponse(
                content=ReasoningAnswer(
                    answer=answer_content,
                    reasoning_content="Structured output does not have reasoning content.",
                    chat_history_dict=updated_history
                ).model_dump()
            )
        else:
            result = await runnable_with_history.ainvoke(input_data, config=config)

            # Estrai content e reasoning content
            _, content, reasoning_content = get_reasoning_content(result, question.llm)

            # Converti content in stringa se è una lista (formato responses/v1 di OpenAI)
            if isinstance(content, list):
                # Estrai il testo dalle risposte
                content_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if 'text' in item:
                            content_parts.append(item['text'])
                        elif 'content' in item:
                            content_parts.append(item['content'])
                content = ''.join(content_parts) if content_parts else str(content)
            elif not isinstance(content, str):
                content = str(content)

            # Aggiorna history usando il metodo centralizzato
            updated_history = _update_history(
                question.chat_history_dict,
                question.question,
                content
            )

            return JSONResponse(
                content=ReasoningAnswer(
                    answer=content,
                    reasoning_content=reasoning_content,
                    chat_history_dict=updated_history
                ).model_dump()
            )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in ask_reason_llm: {str(e)}\n{error_traceback}")

        # Determina il tipo di errore e il messaggio appropriato
        error_message = str(e)
        error_type = type(e).__name__

        # Gestisci errori specifici con messaggi più chiari
        if "unexpected keyword argument" in error_message:
            error_message = f"Configuration error: {error_message}. Please check the 'thinking' parameters for your chosen model."
        elif "API key" in error_message or "authentication" in error_message.lower():
            error_message = f"Authentication error: {error_message}"
        elif "rate limit" in error_message.lower():
            error_message = f"Rate limit exceeded: {error_message}"
        elif "timeout" in error_message.lower():
            error_message = f"Request timeout: {error_message}"

        # Restituisci errore come ReasoningAnswer con status 400
        error_response = ReasoningAnswer(
            answer=f"Error ({error_type}): {error_message}",
            reasoning_content="",
            chat_history_dict=question.chat_history_dict if hasattr(question, 'chat_history_dict') else {}
        )

        return JSONResponse(
            status_code=400,
            content=error_response.model_dump()
        )


@inject_llm_async
async def ask_to_llm(question: QuestionToLLM, chat_model=None):
    """
    Manages requests to LLMs with multimodal support, history, and streaming.
    :param question: QuestionToLLM
    :param chat_model:
    :return: SimpleAnswer
    """
    try:
        # --- 1. System message ---
        messages = []
        if question.system_context:
            messages.append(SystemMessage(content=question.system_context))

        # --- 2. Gestione HISTORY ---
        if question.chat_history_dict:
            if question.contextualize_prompt:
                # Modalità A: History iniettata come TESTO nel system prompt
                history_text = _format_history_as_text(question.chat_history_dict)

                # Modifica il system message per includere la history
                system_with_history = f"""{question.system_context}

                                        Previous conversation history:
                                        {history_text}
                                        """
                messages[0] = SystemMessage(content=system_with_history)

            else:
                # Modalità B: History come MESSAGGI strutturati
                history_messages = await _process_history_messages(
                    question.chat_history_dict,
                    question.max_history_messages,
                    question.summarize_old_history,
                    chat_model
                )
                messages.extend(history_messages)

        # --- 3. Nuova domanda ---
        question_content = question.get_question_content()
        messages.append(HumanMessage(content=question_content))

        # --- 4. STREAMING ---
        if question.stream:
            return StreamingResponse(
                _stream_generic_response(chat_model, messages, question),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )

        # --- 5. RISPOSTA SINCRONA ---
        if question.structured_output and question.output_schema:
            structured_llm = chat_model.with_structured_output(question.output_schema)
            result_message = await structured_llm.ainvoke(messages)

            # The result is a Pydantic model, convert it to a dict
            if hasattr(result_message, 'model_dump'):
                answer_content = result_message.model_dump()
            else:
                answer_content = result_message

            updated_history = _update_history(
                question.chat_history_dict,
                question.question,
                json.dumps(answer_content)  # Store the structured answer as a JSON string
            )
            # Token info might not be available in the same way, handle gracefully
            prompt_token_info = _extract_token_info(result_message)

            return JSONResponse(content=SimpleAnswer(
                answer=answer_content,
                chat_history_dict=updated_history,
                prompt_token_info=prompt_token_info
            ).model_dump())
        else:
            result_message = await chat_model.ainvoke(messages)

            # Aggiorna history
            updated_history = _update_history(
                question.chat_history_dict,
                question.question,
                result_message.content
            )

            # Estrai token info
            prompt_token_info = _extract_token_info(result_message)

            return JSONResponse(content=SimpleAnswer(
                answer=result_message.content,
                chat_history_dict=updated_history,
                prompt_token_info=prompt_token_info
            ).model_dump())

    except Exception as e:
        import traceback
        traceback.print_exc()
        result_to_return = SimpleAnswer(
            answer=repr(e),
            chat_history_dict={},
            prompt_token_info=None
        )
        raise fastapi.exceptions.HTTPException(
            status_code=400,
            detail=result_to_return.model_dump()
        )


async def _process_history_messages(
        chat_history: dict,
        max_messages: Optional[int],
        summarize_old: bool,
        chat_model
) -> list:
    """
    Processes the history and returns a list of messages.
    Handles limitation and summarization.
    """
    if not chat_history:
        return []

    sorted_keys = sorted(chat_history.keys(), key=lambda x: int(x))

    # No limits: everything returns.
    if max_messages is None:
        return _build_message_list(chat_history, sorted_keys)

    total_turns = len(sorted_keys)

    # History stays within the limit.
    if total_turns <= max_messages:
        return _build_message_list(chat_history, sorted_keys)

    # Exceed the limit.
    recent_keys = sorted_keys[-max_messages:]

    if not summarize_old:
        # Limit only: take the last N turns.
        return _build_message_list(chat_history, recent_keys)

    # Summarization active: summarize the old part.
    old_keys = sorted_keys[:-max_messages]
    old_history_text = _format_history_as_text(
        {k: chat_history[k] for k in old_keys}
    )

    # Crea il riassunto della history vecchia
    summary = await _summarize_history(old_history_text, chat_model)

    # Build messages: [summary] + [recent messages]
    messages = [
        SystemMessage(content=f"Summary of earlier conversation:\n{summary}")
    ]
    messages.extend(_build_message_list(chat_history, recent_keys))

    return messages


def _build_message_list(chat_history: dict, keys: list) -> list:
    """
    Builds list of Human/AI messages from history
    :param chat_history:
    :param keys:
    :return:
    """
    messages = []

    for key in keys:
        entry = chat_history[key]

        # User Message (multimodal)
        user_content = (
            entry.get_question_content()
            if hasattr(entry, 'get_question_content')
            else entry.question
        )
        messages.append(HumanMessage(content=user_content))

        # Assistant message
        messages.append(AIMessage(content=entry.answer))

    return messages


async def _summarize_history(history_text: str, chat_model) -> str:
    """
    Use the LLM to create a concise summary of the old history.
    """
    summarization_prompt = f"""You are tasked with summarizing a conversation history.
                            Create a concise summary that captures:
                            - Main topics discussed
                            - Key information exchanged
                            - Important context for continuing the conversation
                            
                            Conversation history to summarize:
                            {history_text}
                            
                            Provide a brief summary (max 200 words):"""

    try:
        summary_msg = await chat_model.ainvoke([
            HumanMessage(content=summarization_prompt)
        ])
        return summary_msg.content.strip()
    except Exception as e:
        logger.exception(e)
        # Fallback: Returns a truncated version.
        return f"Earlier conversation summary (auto-generated): {history_text[:500]}..."


def _format_history_as_text(chat_history: dict) -> str:
    """
    Format history as readable text
    :param chat_history:
    :return:
    """
    if not chat_history:
        return "No previous conversation."

    sorted_keys = sorted(chat_history.keys(), key=lambda x: int(x))
    history_lines = []

    for key in sorted_keys:
        entry = chat_history[key]

        # Gestisci question multimodale
        if hasattr(entry, 'get_question_text'):
            q_text = entry.get_question_text()
        else:
            q_text = str(entry.question)

        history_lines.append(f"User: {q_text}")
        history_lines.append(f"Assistant: {entry.answer}")

    return "\n".join(history_lines)


def _update_history(current_history: dict, new_question, new_answer) -> dict:
    """Update the history with new question/answer"""
    if not current_history:
        current_history = {}

    next_key = str(len(current_history))
    current_history[next_key] = ChatEntry(
        question=new_question,
        answer=new_answer
    )

    return current_history


def _extract_token_info(message) -> PromptTokenInfo:
    """Extracts token information from the message"""
    usage_meta = getattr(message, 'usage_metadata', None) or {}

    return PromptTokenInfo(
        input_tokens=usage_meta.get("input_tokens", 0),
        output_tokens=usage_meta.get("output_tokens", 0),
        total_tokens=usage_meta.get("total_tokens", 0)
    )


def _reasoning_chunk_processor(chunk, question):
    """
    Per-chunk processor with reasoning content (e.g., DeepSeek, Claude, Gemini).

    Args:
        chunk: The chunk received from the stream
        question: QuestionToLLM object with the configuration

    Returns:
        dict with keys: 'content', 'reasoning_content', 'events'

    :param chunk:
    :param question:
    :return:
    """
    is_reasoning, content_text, reasoning_text = get_reasoning_content(chunk, question.llm)

    # Controlla se mostrare il thinking nello stream
    show_thinking = True  # Default
    if hasattr(question, 'thinking') and question.thinking is not None:
        show_thinking = question.thinking.show_thinking_stream

    events = []
    if is_reasoning:
        # Se show_thinking_stream è True, invia il reasoning nello stream
        if show_thinking:
            events.append({"reasoning_content": reasoning_text})
        # Altrimenti non inviare eventi, ma accumula comunque il reasoning_content
    else:
        # Invia sempre il content normale
        events.append({"content": content_text})

    return {
        'content': content_text,
        'reasoning_content': reasoning_text,
        'events': events
    }


async def _stream_generic_response(
    runnable,
    input_data,
    question: QuestionToLLM,
    config: dict = None,
    chunk_processor=None,
    response_class=None
):
    """
    Generic method to handle streaming for any runnable.

    Args:
        runnable: The runnable to execute (chat_model, RunnableWithMessageHistory, agent, etc.)
        input_data: Input data to pass to the runnable (dict or list of messages)
        question: The QuestionToLLM object with request parameters
        config: Optional configuration for the runnable (default: None)
        chunk_processor: Optional function that takes (chunk, question) and returns a dict with keys:
            'content': main content to accumulate
            'events': list of SSE events to emit
            If None, uses standard processing ('content' only)
        response_class: Class for the final response (default: SimpleAnswer)

    Yields:
        Formatted SSE events

    :param runnable:
    :param input_data:
    :param question:
    :param config:
    :param chunk_processor:
    :param response_class:
    :return:
    """
    if response_class is None:
        response_class = SimpleAnswer

    full_response = ""
    additional_data = {}  # Per dati extra come reasoning_content
    message_id = str(uuid.uuid4())
    start_time = datetime.now()

    # Metadati iniziali
    yield _create_event("metadata", {
        "message_id": message_id,
        "status": "started",
        "timestamp": start_time.isoformat()
    })

    # Stream dei chunk (con o senza config)
    if config is not None:
        stream = runnable.astream(input_data, config=config)
    else:
        stream = runnable.astream(input_data)

    async for chunk in stream:
        if hasattr(chunk, 'content'):
            if chunk_processor:
                # Usa il processor custom per casi speciali (es. reasoning)
                result = chunk_processor(chunk, question)
                full_response += result.get('content', '')

                # Accumula dati extra
                for key, value in result.items():
                    if key not in ['content', 'events']:
                        if key not in additional_data:
                            additional_data[key] = ''
                        additional_data[key] += value

                # Emetti gli eventi custom
                for event_data in result.get('events', []):
                    yield _create_event("chunk", {**event_data, "message_id": message_id})
            else:
                # Processing standard: solo content
                full_response += chunk.content
                yield _create_event("chunk", {
                    "content": chunk.content,
                    "message_id": message_id
                })

            await asyncio.sleep(0.01)

    # Aggiorna history usando il metodo centralizzato
    updated_history = _update_history(
        question.chat_history_dict,
        question.question,
        full_response
    )

    # Token info (può essere 0 nello streaming)
    prompt_token_info = PromptTokenInfo(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0
    )

    end_time = datetime.now()

    # Costruisci la risposta finale
    response_data = {
        'answer': full_response,
        'chat_history_dict': updated_history,
        'prompt_token_info': prompt_token_info
    }
    response_data.update(additional_data)

    # Metadati finali
    yield _create_event("metadata", {
        "message_id": message_id,
        "status": "completed",
        "timestamp": end_time.isoformat(),
        "duration": (end_time - start_time).total_seconds(),
        "full_response": full_response,
        "model_used": response_class(**response_data).model_dump()
    })


@inject_llm_async
async def ask_to_llm_1(question: QuestionToLLM, chat_model=None) :
    try:
        logger.info(question)
        chat_history_list = []

        if question.chat_history_dict is not None:
            for key, entry in question.chat_history_dict.items():
                human_content = entry.question if isinstance(entry.question, str) else entry.question
                chat_history_list.append(HumanMessage(content=human_content))  # ('human', entry.question))
                chat_history_list.append(AIMessage(content=entry.answer))

        # Prepara il contenuto della domanda
        question_content = question.get_question_content()

        # --- INIZIO MODIFICA: gestione condizionale di contextualize_prompt ---
        # Se contextualize_prompt è True, include la history nel prompt
        if question.contextualize_prompt:
            # CODICE ORIGINALE: include MessagesPlaceholder per la history
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", question.system_context),
                    MessagesPlaceholder("chat_history", n_messages=question.n_messages),
                    ("human", "{input}")
                ]
            )
        else:
            # NUOVO CODICE: prompt senza history
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", question.system_context),
                    ("human", "{input}")
                ]
            )
        # --- FINE MODIFICA ---

        store = {}
        #get_session_history = lambda session_id: get_or_create_session_history(store, session_id,
        #                                                                       question.chat_history_dict)
        def get_session_history(session_id):
            return get_or_create_session_history(store,
                                                 session_id,
                                                 question.chat_history_dict
                                                 )


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

               async for chunk in runnable_with_history.astream({"input": question_content},
                                                                config={
                                                                    "configurable": {"session_id": uuid.uuid4().hex}}
                                                                ):
                   if hasattr(chunk, 'content'):
                       full_response += chunk.content
                       yield _create_event("chunk", {"content": chunk.content, "message_id": message_id})
                       await asyncio.sleep(0.02)  # Per un flusso più regolare

               end_time = datetime.now()

               if not question.chat_history_dict:
                   question.chat_history_dict = {}

               num_question = len(question.chat_history_dict.keys())
               question.chat_history_dict[str(num_question)] =  ChatEntry(question=question.question, # Salva l'originale (str o List[MultimodalContent])
                                                                          answer=full_response
                                                                          )


                   #{"question": question.question, "answer": full_response}

               #prompt_token_info = PromptTokenInfo(input_tokens=result.usage_metadata.get("input_tokens", 0),
               #                                    output_tokens=result.usage_metadata.get("output_tokens", 0),
               #                                    total_tokens=result.usage_metadata.get("total_tokens", 0), )

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
                {"input": question_content},# 'chat_history_a': chat_history_list,
                config={"configurable": {"session_id": uuid.uuid4().hex}
                        }
            )
            # logger.info(result)

            if not question.chat_history_dict:
                question.chat_history_dict = {}

            num = len(question.chat_history_dict.keys())
            question.chat_history_dict[str(num)] =  ChatEntry(question=question.question,
                                                              answer=result.content
                                                              )

                #{"question": question.question, "answer": result.content}
            prompt_token_info = PromptTokenInfo(input_tokens=result.usage_metadata.get("input_tokens",0),
                                                output_tokens=result.usage_metadata.get("output_tokens",0),
                                                total_tokens=result.usage_metadata.get("total_tokens",0),)

            return JSONResponse(content=SimpleAnswer(answer=result.content,
                                                     chat_history_dict=question.chat_history_dict,
                                                     prompt_token_info=prompt_token_info).model_dump())


    except Exception as e:
        import traceback
        traceback.print_exc()

        result_to_return = SimpleAnswer(answer=repr(e),
                                        chat_history_dict={},
                                        prompt_token_info=None)
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())


@inject_llm_async
async def ask_mcp_agent_llm(question: QuestionToLLM, chat_model: Any):
    """
    Invokes an MCP agent by handling multimodal inputs (text, images, documents).
    Cleanup Strategy:
        1. Normalizes input into a list of content items
        2.Extracts and stores base64 documents/images in a separate storage
        3. Replaces base64 with text references in messages
        4. Middleware further cleans messages before each LLM call
        5. The internal multimodal tool accesses storage when necessary

    :param question:
    :param chat_model:
    :return:
    """
    mcp_client = question.create_mcp_client()

    try:
        #tools = await mcp_client.get_tools()

        tools = await get_all_filtered_tools(mcp_client, question.servers)
        logger.info(f"Available MCP tools: {[tool.name for tool in tools]}")

        # Storage condiviso per contenuti base64
        base64_storage = {}
        base64_counter = 0

        # --- STEP 1: Normalizzazione dell'Input ---
        # Converte question.question (str, json-str, list) in lista unificata
        question_list = _get_question_list(question.question)

        message_content = []

        # --- STEP 2: Conversione in Formato Normalizzato ---
        # Converte tutti i formati possibili in una lista unificata di dict
        for content in question_list:
            if isinstance(content, dict):
                content_type = content.get("type", "text")

                if content_type == "text":
                    message_content.append({"type": "text", "text": content.get("text", "")})

                elif content_type == "image_url":
                    # Converte data URI in formato base64 normalizzato
                    image_url = content.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        try:
                            parts = image_url.split(",", 1)
                            mime_part = parts[0].split(";")[0].replace("data:", "")
                            base64_data = parts[1]
                            message_content.append({
                                "type": "image",
                                "source": {"type": "base64", "media_type": mime_part, "data": base64_data}
                            })
                        except Exception as e:
                            logger.warning(f"Failed to parse image data URI: {e}")
                    else:
                        message_content.append({"type": "image_url", "image_url": {"url": image_url}})

                elif content_type == "document_url":
                    # Converte data URI in formato base64 normalizzato
                    doc_url = content.get("document_url", {}).get("url", "")
                    if doc_url.startswith("data:"):
                        try:
                            parts = doc_url.split(",", 1)
                            mime_part = parts[0].split(";")[0].replace("data:", "")
                            base64_data = parts[1]
                            message_content.append({
                                "type": "document",
                                "source": {"type": "base64", "media_type": mime_part, "data": base64_data}
                            })
                        except Exception as e:
                            logger.warning(f"Failed to parse document data URI: {e}")
                    else:
                        # URL http: aggiunto come riferimento testuale
                        message_content.append({"type": "text", "text": f"[Document URL: {doc_url}]"})

                elif content_type == "document":
                    # IMPORTANTE: Normalizza il formato del document
                    # source può essere: stringa URL, stringa base64, o dict
                    doc_source = content.get("source")
                    doc_mime = content.get("mime_type", "application/pdf")

                    # Normalizza sempre in formato standard
                    if isinstance(doc_source, str):
                        # La stringa può essere URL o base64, lo STEP 3 la distinguerà
                        message_content.append({
                            "type": "document",
                            "source": doc_source,  # Mantieni stringa così com'è
                            "mime_type": doc_mime
                        })
                    elif isinstance(doc_source, dict):
                        # Già in formato dict, mantieni
                        message_content.append({
                            "type": "document",
                            "source": doc_source,
                            "mime_type": doc_mime
                        })
                    else:
                        logger.warning(f"Unknown document source type: {type(doc_source)}")

                elif content_type == "image":
                    # IMPORTANTE: Normalizza il formato dell'image
                    # source può essere: stringa URL, stringa base64, o dict
                    img_source = content.get("source")
                    img_mime = content.get("mime_type", "image/png")

                    # Normalizza sempre in formato standard
                    if isinstance(img_source, str):
                        # La stringa può essere URL o base64
                        message_content.append({
                            "type": "image",
                            "source": img_source,
                            "mime_type": img_mime
                        })
                    elif isinstance(img_source, dict):
                        # Già in formato dict, mantieni
                        message_content.append({
                            "type": "image",
                            "source": img_source,
                            "mime_type": img_mime
                        })
                    else:
                        logger.warning(f"Unknown image source type: {type(img_source)}")

            elif hasattr(content, 'type'):
                # Oggetto Pydantic (TextContent, ImageContent, DocumentContent)
                if content.type in ["document", "image"]:
                    # Normalizza sempre source come stringa (URL o base64)
                    source_data = content.source if isinstance(content.source, str) else base64.b64encode(
                        content.source).decode('utf-8')
                    message_content.append({
                        "type": content.type,
                        "source": source_data,  # Stringa (URL o base64)
                        "mime_type": content.mime_type
                    })
                else:
                    message_content.append(content.to_langchain_format())



        logger.info(f"Content types BEFORE extraction: {[item.get('type') for item in message_content]}")

        # --- STEP 3: Estrazione Documenti e Immagini Base64 ---
        # Estrae documenti/immagini e li sostituisce con riferimenti testuali
        document_metadata = []
        cleaned_content = []

        for item in message_content:
            item_type = item.get("type")

            if item_type == "document":
                # Estrai documento e crea riferimento testuale
                base64_counter += 1
                doc_id = f"doc_{base64_counter}"

                doc_source = item.get("source", {})
                doc_mime = item.get("mime_type", "application/pdf")

                # IMPORTANTE: Distingui tra URL e base64
                is_url = False
                url = None
                base64_data = None

                # Caso 1: source è una stringa
                if isinstance(doc_source, str):

                    # Controlla se è un URL http/https
                    if doc_source.startswith(("http://", "https://")):
                        is_url = True
                        url = doc_source
                        logger.info(f"Document is URL: {url}")
                    else:
                        # È base64 diretto (stringa lunga)
                        base64_data = doc_source
                        logger.info(f"Document is base64 string: {len(base64_data)} bytes")

                # Caso 2: source è un dict con formato {type: "base64", data: "..."}
                elif isinstance(doc_source, dict) and doc_source.get("type") == "base64":
                    base64_data = doc_source.get("data", "")
                    doc_mime = doc_source.get("media_type", doc_mime)
                    logger.info(f"Document is base64 dict: {len(base64_data)} bytes")

                else:
                    logger.warning(f"Unknown document source format: {type(doc_source)}")
                    continue

                # Salva in storage con indicazione del tipo
                if is_url:
                    base64_storage[doc_id] = {
                        "url": url,
                        "type": "document",
                        "source_type": "url",  # ← NUOVO: indica che è una URL
                        "media_type": doc_mime
                    }
                    # Aggiungi riferimento testuale con URL
                    cleaned_content.append({
                        "type": "text",
                        "text": f"[DOCUMENT_{doc_id}: {doc_mime}, URL={url}]"
                    })
                    document_metadata.append({"id": doc_id, "mime_type": doc_mime, "source_type": "url", "url": url})
                    logger.info(f"Extracted document {doc_id}: {doc_mime}, URL={url}")
                else:
                    base64_storage[doc_id] = {
                        "data": base64_data,
                        "type": "document",
                        "source_type": "base64",  # ← NUOVO: indica che è base64
                        "media_type": doc_mime
                    }
                    # Aggiungi riferimento testuale con dimensione
                    cleaned_content.append({
                        "type": "text",
                        "text": f"[DOCUMENT_{doc_id}: {doc_mime}, {len(base64_data)} bytes]"
                    })
                    document_metadata.append({"id": doc_id, "mime_type": doc_mime, "source_type": "base64"})
                    logger.info(f"Extracted document {doc_id}: {doc_mime}, {len(base64_data)} bytes")

            elif item_type == "image":
                # Converti immagine base64 in image_url (formato compatibile con LLM)
                source = item.get("source", {})
                if isinstance(source, dict) and source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    base64_data = source.get("data", "")

                    # Crea data URI per l'LLM (alcuni LLM supportano questo formato)
                    data_uri = f"data:{media_type};base64,{base64_data}"
                    cleaned_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })
                    logger.info("Converted image to data URI format")
                else:
                    cleaned_content.append(item)

            elif item_type in ["text", "image_url"]:
                # Tipi supportati → mantieni
                cleaned_content.append(item)

            else:
                logger.warning(f"Unsupported content type '{item_type}'")
                cleaned_content.append({"type": "text", "text": f"[Unsupported: {item_type}]"})

        message_content = cleaned_content
        logger.info(f"Content types AFTER extraction: {[item.get('type') for item in message_content]}")
        logger.info(f"Documents extracted: {len(document_metadata)}")


        # --- STEP 4: Creazione System Prompt Dinamico ---
        system_prompt = question.system_context or "You are a helpful assistant."

        has_documents = len(document_metadata) > 0
        has_tools = len(tools) > 0

        # Aggiungi istruzioni SOLO se ci sono documenti E tool
        if has_documents and has_tools:
            doc_list = ""
            for doc_info in document_metadata:
                doc_id = doc_info["id"]
                mime_type = doc_info["mime_type"]
                source_type = doc_info.get("source_type", "base64")
                storage_entry = base64_storage.get(doc_id, {})

                if source_type == "url":
                    url = doc_info.get("url", storage_entry.get("url", ""))
                    doc_list += f"  • {doc_id}: {mime_type}\n"
                    doc_list += "    Type: URL\n"
                    doc_list += f"    URL: {url}\n"
                    doc_list += f"    Referenced in message as: [DOCUMENT_{doc_id}]\n"
                else:
                    size = len(storage_entry.get("data", ""))
                    doc_list += f"  • {doc_id}: {mime_type}\n"
                    doc_list += "    Type: BASE64\n"
                    doc_list += f"    Size: {size} bytes\n"
                    doc_list += f"    Referenced in message as: [DOCUMENT_{doc_id}]\n"

            system_prompt += "\n\n" + MCP_DOC_HEADER_TEMPLATE.format(
                doc_count=len(document_metadata),
                tool_count=len(tools),
                doc_list=doc_list
            )
            system_prompt += "\n" + MCP_DOC_INSTRUCTIONS_TEMPLATE.format()

        # --- STEP 5: Setup Tool Multimodale Interno + Registry Tools ---
        from tilellm.tools.multimodal_llm_tool import create_multimodal_llm_tool
        from tilellm.modules.tools_registry.services.tool_registry import resolve_tools

        # Crea sempre il tool multimodale con accesso allo storage base64
        multimodal_tool = create_multimodal_llm_tool(chat_model, base64_storage)

        # Risolvi i tool dal registry se specificati
        registry_tools = []
        if question.tools:
            logger.info(f"Resolving {len(question.tools)} tools from registry: {question.tools}")
            registry_tools = resolve_tools(
                question.tools,
                chat_model=chat_model,
                base64_storage=base64_storage
            )
            logger.info(f"Resolved {len(registry_tools)} tools from registry")

        # Combina tutti i tool: MCP + multimodal interno + registry
        all_tools = tools + [multimodal_tool] + registry_tools

        logger.info(f"Total tools available: {len(all_tools)} ({len(tools)} MCP + 1 internal multimodal + {len(registry_tools)} registry)")

        # Aggiungi istruzioni per il tool interno (solo se necessario)
        if has_documents:
            system_prompt += "\n\n" + MCP_INTERNAL_TOOL_TEMPLATE.format()
            # --- AGGIORNA IL SYSTEM PROMPT ---
            system_prompt += MCP_BASE64_MANAGEMENT_TEMPLATE.format(storage_count=len(base64_storage))

        # --- STEP 6: Funzioni di Pulizia Messaggi ---
        def clean_message_content(content: str) -> tuple[str, bool]:
            """Pulisce contenuto stringa estraendo base64 pesanti."""
            nonlocal base64_counter

            if len(content) < 1000:
                return content, False

            # Prova a parsare come JSON (risposta tipica dei tool MCP)
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    was_cleaned = False
                    for key in ['images_base64', 'documents_base64', 'file_content']:
                        if key in data:
                            # Gestisci liste
                            if isinstance(data[key], list):
                                cleaned_list = []
                                for item in data[key]:
                                    if isinstance(item, str) and len(item) > 1000:
                                        base64_counter += 1
                                        ref_id = f"base64_ref_{base64_counter}"
                                        base64_storage[ref_id] = {
                                            'data': re.sub(r'[\n\r]', '', item),
                                            'type': 'image' if key == 'images_base64' else 'document',
                                            'media_type': 'image/png'
                                        }
                                        cleaned_list.append(f"<{ref_id}>")
                                        was_cleaned = True
                                    else:
                                        cleaned_list.append(item)
                                data[key] = cleaned_list
                            # Gestisci stringhe singole
                            elif isinstance(data[key], str) and len(data[key]) > 1000:
                                base64_counter += 1
                                ref_id = f"base64_ref_{base64_counter}"
                                base64_storage[ref_id] = {
                                    'data': re.sub(r'[\n\r]', '', data[key]),
                                    'type': 'image' if key == 'images_base64' else 'document',
                                    'media_type': 'image/png'
                                }
                                data[key] = f"<{ref_id}>"
                                was_cleaned = True

                    if was_cleaned:
                        return json.dumps(data), True
            except json.JSONDecodeError:
                pass

            # Fallback: stringa non-JSON base64 raw
            sample = (content[:500] + content[-500:]).strip()
            if re.fullmatch(r'[A-Za-z0-9+/=]+', re.sub(r'\s+', '', sample)):
                base64_counter += 1
                ref_id = f"base64_ref_{base64_counter}"
                base64_storage[ref_id] = {
                    'data': re.sub(r'[\n\r]', '', content),
                    'type': 'binary',
                    'media_type': 'application/octet-stream'
                }
                return f"[BINARY_REF:{ref_id}:length={len(content)}]", True

            return content, False

        @wrap_model_call
        async def pre_model_cleaning_middleware(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            """Middleware called before each LLM invocation to clean messages."""
            messages = request.messages
            cleaned_messages = []

            for msg in messages:
                if not hasattr(msg, 'content'):
                    cleaned_messages.append(msg)
                    continue

                content = msg.content

                # Pulisci solo contenuto stringa
                if isinstance(content, str):
                    cleaned_content, was_cleaned = clean_message_content(content)

                    if was_cleaned:
                        # Ricostruisci messaggio con content pulito
                        if msg.type == 'tool':
                            cleaned_messages.append(ToolMessage(
                                content=cleaned_content,
                                tool_call_id=msg.tool_call_id,
                                name=getattr(msg, 'name', None),
                                id=getattr(msg, 'id', None)
                            ))
                        elif msg.type == 'ai':
                            cleaned_messages.append(AIMessage(
                                content=cleaned_content,
                                tool_calls=getattr(msg, 'tool_calls', None),
                                id=getattr(msg, 'id', None)
                            ))
                        elif msg.type == 'human':
                            cleaned_messages.append(HumanMessage(content=cleaned_content))
                        elif msg.type == 'system':
                            cleaned_messages.append(SystemMessage(content=cleaned_content))
                        else:
                            cleaned_messages.append(msg)
                    else:
                        cleaned_messages.append(msg)
                else:
                    # Content non stringa → mantieni originale
                    cleaned_messages.append(msg)

            return await handler(request.override(messages=cleaned_messages))

        # --- STEP 7: Creazione Agent con Middleware ---
        agent_executor = create_agent(
            model=chat_model,
            tools=all_tools,
            system_prompt=system_prompt,
            middleware=[pre_model_cleaning_middleware]
        )

        # --- STEP 8: Invocazione Agent ---
        # Estrai testo per il campo 'input'
        text_prompt = " ".join([
            item["text"] for item in message_content if item["type"] == "text"
        ]).strip()

        logger.info(f"Invoking agent with {len(message_content)} content items")

        response = await agent_executor.ainvoke({
            "input": text_prompt,
            "messages": [HumanMessage(content=message_content)]
        })

        # PULIZIA FINALE: Pulisci i messaggi nella risposta per rimuovere base64 lunghi
        if "messages" in response:
            logger.info(f"Final cleaning of {len(response['messages'])} messages")
            cleaned_messages = []
            for msg in response['messages']:
                if not hasattr(msg, 'content'):
                    cleaned_messages.append(msg)
                    continue

                content = msg.content
                if isinstance(content, str):
                    cleaned_content, was_cleaned = clean_message_content(content)
                    if was_cleaned:
                        # Ricostruisci messaggio con content pulito
                        if msg.type == 'tool':
                            cleaned_messages.append(ToolMessage(
                                content=cleaned_content,
                                tool_call_id=msg.tool_call_id,
                                name=getattr(msg, 'name', None),
                                id=getattr(msg, 'id', None)
                            ))
                        elif msg.type == 'ai':
                            cleaned_messages.append(AIMessage(
                                content=cleaned_content,
                                tool_calls=getattr(msg, 'tool_calls', None),
                                id=getattr(msg, 'id', None)
                            ))
                        elif msg.type == 'human':
                            cleaned_messages.append(HumanMessage(content=cleaned_content))
                        elif msg.type == 'system':
                            cleaned_messages.append(SystemMessage(content=cleaned_content))
                        else:
                            cleaned_messages.append(msg)
                    else:
                        cleaned_messages.append(msg)
                else:
                    cleaned_messages.append(msg)
            response['messages'] = cleaned_messages

        # Estrae TUTTI i ToolMessage e AIMessage per avere la risposta completa
        result = extract_conversation_flow(response['messages'])
        logger.info(f"============== \n{response} \n====================")
        logger.info(f"Extracted conversation flow:\n{result}")
        logger.info("Agent completed successfully")

        # Estrae i token consumati da tutti i messaggi AIMessage
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0

        for msg in response.get('messages', []):
            if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                total_input_tokens += msg.usage_metadata.get('input_tokens', 0)
                total_output_tokens += msg.usage_metadata.get('output_tokens', 0)
                total_tokens += msg.usage_metadata.get('total_tokens', 0)

        # Crea l'oggetto PromptTokenInfo
        prompt_token_info = PromptTokenInfo(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_tokens
        )

        logger.info(f"Token usage - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens}")

        response_data = SimpleAnswer(
            answer=result.get("ai_message", "No answer"),  # Assegna 'ai_message' ad 'answer'
            tools_log=result.get("tools"),  # Assegna 'tools' a 'tools_log'
            chat_history_dict={},
            prompt_token_info=prompt_token_info
        )


        return JSONResponse(
            content=response_data.model_dump()
        )

    except Exception as e:
        return handle_agent_exception(e, "ask_mcp_agent_llm")



@inject_llm_async
async def ask_mcp_agent_llm_simple(question: QuestionToLLM, chat_model=None):
    """
    Optimized version that resolves the context overflow issue.
    PROBLEM: When MCP tools return base64 images, they are included in the agent's messages, causing context overflow (671K tokens).
    SOLUTION:
        1. Extract and store base64 content in a separate storage
        2. Replace base64 in messages with text references
        3. The internal multimodal tool accesses storage when invoked
        4. Agent messages remain lightweight (text + references only)

    :param question:
    :param chat_model:
    :return:
    """
    mcp_client = question.create_mcp_client()

    try:
        #tools = await mcp_client.get_tools()
        tools = await get_all_filtered_tools(mcp_client, question.servers)

        logger.info(f"Available MCP tools: {[tool.name for tool in tools]}")

        # Crea il tool multimodale interno con accesso allo storage
        from tilellm.tools.multimodal_llm_tool import create_multimodal_llm_tool
        from tilellm.modules.tools_registry.services.tool_registry import resolve_tools

        # Storage condiviso per contenuti base64
        # Sarà popolato durante il preprocessing dei messaggi
        base64_storage = {}

        # Crea sempre il tool multimodale interno
        multimodal_tool = create_multimodal_llm_tool(chat_model, base64_storage)

        # Risolvi i tool dal registry se specificati
        registry_tools = []
        if question.tools:
            logger.info(f"Resolving {len(question.tools)} tools from registry: {question.tools}")
            registry_tools = resolve_tools(
                question.tools,
                chat_model=chat_model,
                base64_storage=base64_storage
            )
            logger.info(f"Resolved {len(registry_tools)} tools from registry")

        # Combina tutti i tool: MCP + multimodal interno + registry
        all_tools = tools + [multimodal_tool] + registry_tools

        logger.info(f"Simple mode optimized: Total tools: {len(all_tools)} ({len(tools)} MCP + 1 internal multimodal + {len(registry_tools)} registry)")

        # Converti question.question in lista se è stringa
        if isinstance(question.question, str):
            messages = [question.question]
        else:
            messages = question.question

        # Analizza i messaggi e estrai base64
        processed_messages = []
        base64_counter = 0

        for msg in messages:
            # Se è una stringa semplice, mantienila
            if isinstance(msg, str):
                processed_messages.append(msg)
                continue

            # Se è un oggetto con content che contiene base64
            if hasattr(msg, 'content'):
                content = msg.content

                # Se content è una lista (multimodale)
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        # Controlla se è un dizionario
                        if isinstance(item, dict):
                            item_type = item.get('type', '')

                            # Se è un'immagine con base64
                            if item_type == 'image':
                                base64_counter += 1
                                ref_id = f"base64_ref_{base64_counter}"

                                # Estrai il base64
                                source = item.get('source', {})
                                if isinstance(source, dict) and source.get('type') == 'base64':
                                    base64_data = source.get('data', '')
                                    media_type = source.get('media_type', 'image/png')

                                    # Salva nello storage
                                    base64_storage[ref_id] = {
                                        'data': base64_data,
                                        'type': 'image',
                                        'media_type': media_type
                                    }

                                    # Sostituisci con riferimento testuale
                                    new_content.append({
                                        'type': 'text',
                                        'text': f"[IMAGE_REF:{ref_id}:length={len(base64_data)}:type={media_type}]"
                                    })
                                    logger.info(f"Extracted {ref_id}: {media_type}, length: {len(base64_data)}")
                                    continue

                            # Se è un'immagine con image_url
                            elif item_type == 'image_url':
                                image_url = item.get('image_url', {}).get('url', '')

                                # Se è un data URI con base64
                                if image_url.startswith('data:'):
                                    base64_counter += 1
                                    ref_id = f"base64_ref_{base64_counter}"

                                    # Estrai mime type e base64
                                    try:
                                        parts = image_url.split(',', 1)
                                        if len(parts) == 2:
                                            mime_part = parts[0].split(';')[0].replace('data:', '')
                                            base64_data = parts[1]

                                            # Salva nello storage
                                            base64_storage[ref_id] = {
                                                'data': base64_data,
                                                'type': 'image',
                                                'media_type': mime_part
                                            }

                                            # Sostituisci con riferimento testuale
                                            new_content.append({
                                                'type': 'text',
                                                'text': f"[IMAGE_REF:{ref_id}:length={len(base64_data)}:type={mime_part}]"
                                            })
                                            logger.info(f"Extracted {ref_id} from data URI: {mime_part}, length: {len(base64_data)}")
                                            continue
                                    except Exception as e:
                                        logger.warning(f"Failed to parse data URI: {e}")

                            # Se è un documento con base64
                            elif item_type == 'document':
                                base64_counter += 1
                                ref_id = f"base64_ref_{base64_counter}"

                                source = item.get('source', {})
                                mime_type = item.get('mime_type', 'application/pdf')

                                if isinstance(source, dict) and source.get('type') == 'base64':
                                    base64_data = source.get('data', '')

                                    # Salva nello storage
                                    base64_storage[ref_id] = {
                                        'data': base64_data,
                                        'type': 'document',
                                        'media_type': mime_type
                                    }

                                    # Sostituisci con riferimento testuale
                                    new_content.append({
                                        'type': 'text',
                                        'text': f"[DOCUMENT_REF:{ref_id}:length={len(base64_data)}:type={mime_type}]"
                                    })
                                    logger.info(f"Extracted {ref_id}: {mime_type}, length: {len(base64_data)}")
                                    continue

                        # Se non è base64, mantieni l'item originale
                        new_content.append(item)

                    # Crea nuovo messaggio con contenuto processato
                    # HumanMessage, AIMessage, SystemMessage già importati all'inizio

                    if hasattr(msg, 'type'):
                        if msg.type == 'human':
                            processed_messages.append(HumanMessage(content=new_content))
                        elif msg.type == 'ai':
                            processed_messages.append(AIMessage(content=new_content))
                        elif msg.type == 'system':
                            processed_messages.append(SystemMessage(content=new_content))
                        else:
                            processed_messages.append(msg)
                    else:
                        processed_messages.append(HumanMessage(content=new_content))
                else:
                    # Content non è lista, mantieni il messaggio originale
                    processed_messages.append(msg)
            else:
                # Non ha content, mantieni originale
                processed_messages.append(msg)

        logger.info(f"Preprocessing completed: extracted {len(base64_storage)} base64 items")
        logger.info(f"Storage keys: {list(base64_storage.keys())}")

        def process_messages_state_modifier(state):
            """
            State modifier that removes base64 from ALL messages at every step.
            Intercepts both inputs and tool outputs (ToolMessage).
            """
            messages = state.get("messages", [])
            cleaned_messages = []

            nonlocal base64_counter  # Usa il contatore globale

            for msg in messages:
                # 1. Controlla se il messaggio ha contenuto
                if not hasattr(msg, 'content'):
                    cleaned_messages.append(msg)
                    continue

                content = msg.content
                was_cleaned = False  # Flag per tracciare se il 'content' è stato modificato

                # 2. Pulisci gli argomenti dei tool_calls (SEMPRE, se presenti)
                # (Questa logica gestisce i base64 inviati *dall'LLM al tool*)
                tool_calls_were_cleaned = False
                final_tool_calls = None

                if hasattr(msg, 'tool_calls') and msg.tool_calls:

                    final_tool_calls = []
                    for tc in msg.tool_calls:
                        # ... (logica di pulizia args) ...
                        # if args_were_cleaned:
                        #    tool_calls_were_cleaned = True
                        final_tool_calls.append(tc)  # Aggiungi il tool_call pulito o originale

                # 3. Pulisco il 'content' del messaggio

                # --- CASO A: CONTENT È UNA STRINGA (es. ToolMessage o testo semplice) ---
                if isinstance(content, str):
                    cleaned_content = content

                    if len(content) > 10000:  # Controlla solo stringhe lunghe

                        # --- A.1: Prova a parsarla come JSON (LA NUOVA LOGICA) ---
                        try:
                            # Il content è JSON, come: '{"images_base64": ["iVBOR..."]}'
                            data = json.loads(content)
                            keys_to_clean = ['images_base64', 'documents_base64', 'file_content']

                            if isinstance(data, dict):
                                for key in keys_to_clean:
                                    # Pulisci liste di base64
                                    if key in data and isinstance(data[key], list):
                                        cleaned_list = []
                                        for item in data[key]:
                                            if isinstance(item, str) and len(item) > 10000:
                                                base64_counter += 1
                                                ref_id = f"base64_ref_{base64_counter}"

                                                storage_type = 'image' if key == 'images_base64' else 'document'
                                                base64_storage[ref_id] = {
                                                    'data': re.sub(r'[\n\r]', '', item),  # Pulisci \n e salva
                                                    'type': storage_type,
                                                    'media_type': 'image/jpeg'  # Assunzione, migliora se possibile
                                                }

                                                cleaned_list.append(f"<{ref_id}>")
                                                was_cleaned = True
                                                logger.info(
                                                    f"[STATE_MODIFIER] Extracted {ref_id} from JSON key '{key}'")
                                            else:
                                                cleaned_list.append(item)
                                        data[key] = cleaned_list

                                    # Pulisci stringhe singole di base64
                                    elif key in data and isinstance(data[key], str) and len(data[key]) > 10000:
                                        base64_counter += 1
                                        ref_id = f"base64_ref_{base64_counter}"
                                        storage_type = 'image' if key == 'images_base64' else 'document'

                                        base64_storage[ref_id] = {
                                            'data': re.sub(r'[\n\r]', '', data[key]),  # Pulisci \n e salva
                                            'type': storage_type,
                                            'media_type': 'image/jpeg'
                                        }

                                        data[key] = f"<{ref_id}>"
                                        was_cleaned = True
                                        logger.info(
                                            f"[STATE_MODIFIER] Extracted {ref_id} from JSON key '{key}' (string)")

                                if was_cleaned:
                                    # Riconverti il dizionario PULITO in una stringa JSON
                                    cleaned_content = json.dumps(data)

                        except json.JSONDecodeError:
                            # --- A.2: Non era JSON, prova l'euristica Base64 Raw ---
                            logger.warning(
                                f"[STATE_MODIFIER] Content (len {len(content)}) is not JSON. Trying RAW Base64 heuristic...")

                            sample_raw = (content[:500] + content[-500:]).strip()
                            sample_cleaned = re.sub(r'\s+', '', sample_raw)  # Pulisci \n, \r, spazi

                            if re.fullmatch(r'[A-Za-z0-9+/=]+', sample_cleaned):
                                logger.warning("[STATE_MODIFIER] Detected large RAW string. Storing as ref.")

                                base64_counter += 1
                                ref_id = f"base64_ref_{base64_counter}"

                                full_cleaned_data = re.sub(r'[\n\r]', '', content)  # Pulisci l'intera stringa

                                base64_storage[ref_id] = {
                                    'data': full_cleaned_data,
                                    'type': 'binary',
                                    'media_type': 'application/octet-stream'
                                }

                                cleaned_content = f"[BINARY_REF:{ref_id}:length={len(full_cleaned_data)}]"
                                was_cleaned = True
                            else:
                                logger.debug(
                                    f"[STATE_MODIFIER] Large string FAILED heuristic. Cleaned sample: '{sample_cleaned[:100]}...'")

                    # --- A.3: Ricostruisci il messaggio se 'content' è cambiato ---
                    if was_cleaned or tool_calls_were_cleaned:
                        logger.info(
                            f"[STATE_MODIFIER] Rebuilding message (type: {msg.type}) due to content/tool_call cleaning.")
                        tool_calls_to_use = final_tool_calls if final_tool_calls is not None else (
                            msg.tool_calls if hasattr(msg, 'tool_calls') else None)

                        if msg.type == 'tool':
                            cleaned_messages.append(ToolMessage(
                                content=cleaned_content,
                                tool_call_id=msg.tool_call_id,
                                name=msg.name if hasattr(msg, 'name') else None,
                                id=msg.id if hasattr(msg, 'id') else None,
                                additional_kwargs=msg.additional_kwargs if hasattr(msg, 'additional_kwargs') else {}
                            ))
                        elif msg.type == 'ai':
                            cleaned_messages.append(AIMessage(
                                content=cleaned_content,
                                tool_calls=tool_calls_to_use,
                                id=msg.id if hasattr(msg, 'id') else None,
                                additional_kwargs=msg.additional_kwargs if hasattr(msg, 'additional_kwargs') else {}
                            ))
                        elif msg.type == 'system':
                            cleaned_messages.append(SystemMessage(content=cleaned_content))
                        elif msg.type == 'human':
                            cleaned_messages.append(HumanMessage(content=cleaned_content))
                        else:
                            logger.warning(
                                f"[STATE_MODIFIER] Unhandled message type '{msg.type}' with string content, appending original.")
                            cleaned_messages.append(msg)
                    else:
                        cleaned_messages.append(msg)  # Invariato

                # --- CASO B: CONTENT È UNA LISTA (Multimodale) ---
                elif isinstance(content, list):
                    # ... (La tua logica esistente per pulire le liste multimodali va qui) ...
                    # ... (Sembrava corretta per gli input) ...
                    cleaned_messages.append(msg)  # Sostituisci con la tua logica di pulizia liste

                # --- CASO C: ALTRO ---
                else:
                    # Contenuto non stringa/lista, o messaggio senza 'content'
                    cleaned_messages.append(msg)

            # Ritorna lo stato modificato
            return {"messages": cleaned_messages}

        # --- AGGIORNA IL SYSTEM PROMPT ---
        system_instructions = MCP_BASE64_MANAGEMENT_TEMPLATE.format(storage_count=len(base64_storage))

        @wrap_model_call
        async def pre_model_cleaning_middleware(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            """
            Middleware che pulisce i messaggi prima di ogni chiamata LLM.
            """
            current_messages = request.messages

            logger.debug("[MIDDLEWARE] ========== CALLED ==========")
            logger.debug(f"[MIDDLEWARE] Processing {len(current_messages)} messages")

            # DEBUG: Analizza messaggi PRIMA della pulizia
            total_size_before = 0
            for idx, msg in enumerate(current_messages):
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str):
                        total_size_before += len(content)
                        if len(content) > 10000:
                            has_base64 = 'base64' in content or 'data:image' in content
                            logger.debug(f"[MIDDLEWARE] ⚠️ Message {idx} ({msg.type}): LARGE STRING content, length={len(content)}, has_base64={has_base64}")
                            if has_base64:
                                # Mostra un piccolo sample
                                sample_start = content[:100]
                                logger.debug(f"[MIDDLEWARE] Sample start: {sample_start}...")
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'image_url':
                                    url = item.get('image_url', {}).get('url', '')
                                    total_size_before += len(url)
                                    if len(url) > 10000:
                                        logger.debug(f"[MIDDLEWARE] ⚠️ Message {idx} has LARGE image_url: {len(url)} chars")

            logger.debug(f"[MIDDLEWARE] Total content size BEFORE: {total_size_before} chars (~{total_size_before//4} tokens)")

            # Pulisci i messaggi
            cleaned_state = process_messages_state_modifier({"messages": current_messages})
            cleaned_messages = cleaned_state["messages"]

            # DEBUG: Verifica messaggi DOPO la pulizia
            total_size_after = 0
            for idx, msg in enumerate(cleaned_messages):
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str):
                        total_size_after += len(content)
                        if len(content) > 10000:
                            logger.debug(f"[MIDDLEWARE] ❌ STILL LARGE AFTER! Message {idx}, length={len(content)}")

            logger.debug(f"[MIDDLEWARE] Total content size AFTER: {total_size_after} chars (~{total_size_after//4} tokens)")
            logger.debug(f"[MIDDLEWARE] Reduction: {total_size_before - total_size_after} chars")

            # Conta quanti base64 sono stati estratti in questo step
            extracted_count = len(base64_storage)
            logger.debug(f"[MIDDLEWARE] Total refs in storage: {extracted_count}")

            # DEBUG FINALE: Verifica tool_calls negli AIMessage puliti
            for idx, cleaned_msg in enumerate(cleaned_messages):
                if hasattr(cleaned_msg, 'tool_calls') and cleaned_msg.tool_calls:
                    logger.debug(f"[MIDDLEWARE] Message {idx} (AIMessage) has {len(cleaned_msg.tool_calls)} tool_calls")
                    for tc_idx, tc in enumerate(cleaned_msg.tool_calls):
                        args = tc.get('args', {})
                        if 'images_base64' in args:
                            imgs = args['images_base64']
                            if isinstance(imgs, list) and len(imgs) > 0:
                                first_img = imgs[0] if len(imgs) > 0 else ''
                                is_ref = first_img.startswith('<base64_ref_')
                                logger.debug(f"[MIDDLEWARE] ⭐ tool_call {tc_idx} has images_base64: {len(imgs)} items, first={'REFERENCE' if is_ref else 'RAW BASE64'}, sample={first_img[:50]}")

            return await handler(request.override(messages=cleaned_messages))

        # Pulisci i messaggi iniziali
        cleaned_input = process_messages_state_modifier({"messages": processed_messages})
        agent_input = {"messages": cleaned_input["messages"]}

        logger.info(f"Starting agent with {len(agent_input['messages'])} initial cleaned messages")

        # Crea agent con middleware (LangChain 1.1)
        base_agent = create_agent(
            model=chat_model,
            tools=all_tools,
            system_prompt=system_instructions,
            middleware=[pre_model_cleaning_middleware]  # ← CHIAVE! Pulisce ad ogni step
        )

        # Invoca l'agent
        logger.info("Invoking agent...")
        response = await base_agent.ainvoke(agent_input)

        # PULIZIA FINALE: Pulisci i messaggi nella risposta (per sicurezza)
        if "messages" in response:
            logger.info(f"Final cleaning of {len(response['messages'])} messages")
            final_cleaned = process_messages_state_modifier({"messages": response["messages"]})
            response["messages"] = final_cleaned["messages"]

        logger.info("Agent response received")
        result = extract_conversation_flow(response['messages'])
        logger.debug(result)

        # Estrae i token consumati da tutti i messaggi AIMessage
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0

        for msg in response.get('messages', []):
            if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                total_input_tokens += msg.usage_metadata.get('input_tokens', 0)
                total_output_tokens += msg.usage_metadata.get('output_tokens', 0)
                total_tokens += msg.usage_metadata.get('total_tokens', 0)

        # Crea l'oggetto PromptTokenInfo
        prompt_token_info = PromptTokenInfo(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_tokens
        )

        logger.info(f"Token usage - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens}")

        response_data = SimpleAnswer(
            answer=result.get("ai_message", "No answer"),  # Assegna 'ai_message' ad 'answer'
            tools_log=result.get("tools"),  # Assegna 'tools' a 'tools_log'
            chat_history_dict={},
            prompt_token_info=prompt_token_info
        )

        return JSONResponse(
            content=response_data.model_dump()
        )

    except Exception as e:
        return handle_agent_exception(e, "ask_mcp_agent_llm_simple")

@inject_repo_async
@inject_llm_chat_async
async def ask_with_memory(question_answer, repo=None, llm=None, callback_handler=None, llm_embeddings=None, embedding_config_key=None) -> RetrievalResult:
    """
    Ask to LLM your questions
    :param question_answer:
    :param repo:
    :param llm:
    :param callback_handler:
    :param llm_embeddings:
    :param embedding_config_key:
    :return: RetrievalResult
    """
    try:

        logger.info(question_answer)

        # Preprocess chat history
        chat_history_list, question_answer_list = preprocess_chat_history(question_answer)

        # Modifiche
        # Initialize embeddings and retrievers (con supporto per re-ranking)


        base_retriever = await initialize_retrievers(question_answer, repo, llm_embeddings, embedding_config_key)

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
        #get_session_history = lambda session_id: get_or_create_session_history(
        #    store, session_id, question_answer.chat_history_dict
        #)
        def get_session_history(session_id):
            return get_or_create_session_history(store,
                                                 session_id,
                                                 question_answer.chat_history_dict
                                                 )

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


@inject_repo_async
async def add_item(item, repo=None) -> IndexingResult:
    """
    Add items to namespace
    :type repo: PineconeRepositoryBase
    :param item:
    :param repo:
    :return: PineconeIndexingResult
    """
    try:
        return await repo.add_item(item)
    except Exception as e:
        raise e


@inject_repo_async
async def add_item_hybrid(item, repo=None) -> IndexingResult:
    """

    :return:
    """
    try:
        return await repo.add_item_hybrid(item)
    except Exception as e:
        raise e


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
async def get_listitems_namespace(repository_engine: RepositoryEngine, namespace: str, with_text=False, repo=None) -> RepositoryItems:
    """
    Get all items from given namespace
    :param repository_engine: RepositoryEngine
    :param namespace: namespace_id
    :param with_text: Text of chunk
    :param repo:
    :return: list of al items PineconeItems
    """
    try:
        return await repo.get_all_obj_namespace(engine=repository_engine.engine,
                                                   namespace=namespace, with_text=with_text)
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
    Checks if the 'reasoning_content' key exists in the nested
    'additional_kwargs' dictionary of a chunk and returns its value.

    Args:
        chunk (dict): The dictionary containing the data.
        llm: The LLM used.

    Returns:
        tuple: (is_thinking, content_text, reasoning_text)
            - is_thinking: True if this chunk contains thinking content
            - content_text: The main response content
            - reasoning_text: The reasoning/thinking content
    """
    if llm == "openai":
        # OpenAI GPT-5 con responses/v1 restituisce una lista di oggetti
        full_text = ""
        full_reasoning = ""
        is_reasoning = False

        if isinstance(chunk.content, list):
            # Formato responses/v1: lista di oggetti con type e text/reasoning
            for item in chunk.content:
                if isinstance(item, dict):
                    # Reasoning content ha type='reasoning' o 'reason'
                    if item.get('type') == 'reasoning' or item.get('type') == 'reason':
                        full_reasoning += item.get('text', item.get('reasoning', ''))
                        is_reasoning = True
                    # Text content ha type='text' o 'message'
                    elif item.get('type') == 'text' or item.get('type') == 'message':
                        full_text += item.get('text', item.get('content', ''))
                    # Fallback: cerca direttamente 'text' o 'content'
                    elif 'text' in item:
                        full_text += item['text']
                    elif 'content' in item:
                        full_text += item['content']
            return is_reasoning, full_text, full_reasoning
        else:
            # Formato standard: stringa semplice
            return False, chunk.content, ''

    elif llm == "deepseek":
        # DeepSeek usa additional_kwargs['reasoning_content']
        if 'reasoning_content' in chunk.additional_kwargs:
            return True, chunk.content, chunk.additional_kwargs['reasoning_content']
        else:
            return False, chunk.content, ''

    elif llm == "anthropic":
        # Claude usa una lista di elementi con 'thinking' e 'text'
        full_thinking = ""
        full_text = ""
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

    elif llm == "google":
        # Gemini usa diverse strutture a seconda della versione API
        full_thinking = ""
        full_text = ""
        is_thinking = False

        # Controlla se chunk.content è una lista (formato v1alpha)
        if isinstance(chunk.content, list):
            for content_part in chunk.content:
                if isinstance(content_part, dict):
                    # Formato: {'type': 'thinking', 'thinking': '...'}
                    if content_part.get('type') == 'thinking':
                        full_thinking += content_part.get('thinking', '')
                        is_thinking = True
                    # Formato: {'type': 'text', 'text': '...'}
                    elif content_part.get('type') == 'text':
                        full_text += content_part.get('text', '')
                        is_thinking = False
                    # Fallback: cerca direttamente 'thinking' o 'text'
                    elif 'thinking' in content_part:
                        full_thinking += content_part['thinking']
                        is_thinking = True
                    elif 'text' in content_part:
                        full_text += content_part['text']
                        is_thinking = False

        # Controlla anche additional_kwargs per compatibilità
        elif hasattr(chunk, 'additional_kwargs') and 'thinking' in chunk.additional_kwargs:
            full_thinking = chunk.additional_kwargs['thinking']
            full_text = chunk.content if isinstance(chunk.content, str) else ''
            is_thinking = True

        else:
            # Fallback: tratta come testo normale
            full_text = chunk.content if isinstance(chunk.content, str) else ''

        return is_thinking, full_text, full_thinking

    else:
        # Provider non supportato per reasoning
        return False, chunk.content if hasattr(chunk, 'content') else '', ''