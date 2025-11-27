import base64
import uuid
import json
from datetime import datetime
from typing import List, Any

import re
from langchain_core.messages import ToolMessage

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
    generate_answer_with_history, handle_exception, initialize_retrievers, _create_event, extract_conversation_flow, create_contextualize_query
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
    QuestionToAgent,
    QuestionToLLM,
    QuestionAnswer,
    SimpleAnswer,
    PromptTokenInfo
    )

from tilellm.shared.utility import inject_repo_async, \
    inject_llm_chat_async, inject_llm_async, inject_reason_llm_async

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from tilellm.agents.shopify_agent import lookup as shopify_lookup_agent

import logging

from tilellm.tools.reranker import RerankedRetriever, TileReranker

logger = logging.getLogger(__name__)

@inject_repo_async
@inject_llm_chat_async
async def ask_hybrid_with_memory(question_answer, repo=None, llm=None, callback_handler=None, llm_embeddings=None, emb_dimension=None, embedding_config_key=None):
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
                                                                     "configurable": {"session_id": uuid.uuid4().hex}}
                                                                 ):

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


# Import necessari
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


@inject_llm_async
async def ask_to_llm(question: QuestionToLLM, chat_model=None):
    """
    Gestisce una richiesta a un LLM, supportando contenuti multimodali,
    cronologia della chat e risposte in streaming.
    Utilizza la costruzione manuale dei messaggi per evitare errori di tokenizzazione
    con input di immagini base64.
    """
    try:
        # --- 1. Gestione della Cronologia ---
        # Per questo esempio, creiamo uno store e un session_id temporanei.

        temp_store = {}
        session_id = uuid.uuid4().hex  # Usa un ID di sessione reale se disponibile

        get_session_history = lambda sid: get_or_create_session_history(
            temp_store, sid, question.chat_history_dict
        )
        history = get_session_history(session_id)

        # --- 2. Costruzione Manuale della Lista di Messaggi ---
        # Questo è il passaggio chiave per risolvere il problema dei token.
        final_messages = [SystemMessage(content=question.system_context)]
        final_messages.extend(history.messages)

        # Prepara il contenuto della domanda (testo o lista multimodale)
        question_content = question.get_question_content()
        new_human_message = HumanMessage(content=question_content)
        final_messages.append(new_human_message)

        # --- 3. Logica per lo Streaming e la Risposta ---
        if question.stream:
            async def get_stream_llm():
                full_response_content = ""
                message_id = str(uuid.uuid4())
                start_time = datetime.now()

                # Yield del metadato di avvio
                yield _create_event("metadata", {
                    "message_id": message_id,
                    "status": "started",
                    "timestamp": start_time.isoformat()
                })

                # Chiamata in streaming DIRETTA al modello
                final_response_message = None
                async for chunk in chat_model.astream(final_messages):
                    if hasattr(chunk, 'content'):
                        full_response_content += chunk.content
                        yield _create_event("chunk", {"content": chunk.content, "message_id": message_id})
                        await asyncio.sleep(0.01)  # Opzionale, per un flusso più fluido

                # Costruisci il messaggio di risposta completo per la history e i metadati
                final_response_message = AIMessage(content=full_response_content)
                end_time = datetime.now()

                # Aggiorna la cronologia con la domanda e la risposta completa
                #history.add_messages([new_human_message, final_response_message])
                #question.chat_history_dict = {str(i): entry for i, entry in
                #                              enumerate(history.messages)}  # Ricostruisci il dizionario

                if not question.chat_history_dict:
                    question.chat_history_dict = {}

                num_question = len(question.chat_history_dict.keys())
                question.chat_history_dict[str(num_question)] = ChatEntry(question=question.question,
                                                                          # Salva l'originale (str o List[MultimodalContent])
                                                                          answer=full_response_content
                                                                          )
                # Ottieni i metadati sull'utilizzo (se disponibili post-streaming)
                # Nota: `usage_metadata` potrebbe non essere sempre disponibile nello streaming.
                # In tal caso, i token saranno 0.
                usage_meta = final_response_message.usage_metadata or {}
                prompt_token_info = PromptTokenInfo(
                    input_tokens=usage_meta.get("input_tokens", 0),
                    output_tokens=usage_meta.get("output_tokens", 0),
                    total_tokens=usage_meta.get("total_tokens", 0),
                )

                simple_answer = SimpleAnswer(
                    answer=full_response_content,
                    chat_history_dict=question.chat_history_dict,
                    prompt_token_info=prompt_token_info
                )

                # Yield del metadato di completamento
                yield _create_event("metadata", {
                    "message_id": message_id,
                    "status": "completed",
                    "timestamp": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "full_response": full_response_content,
                    "model_used": simple_answer.model_dump()
                })

            return StreamingResponse(
                get_stream_llm(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )

        else:  # Logica per la risposta non in streaming
            # Chiamata DIRETTA al modello
            result_message = await chat_model.ainvoke(final_messages)

            # Aggiorna la cronologia
            #history.add_messages(ChatEntry(question=new_human_message.content, answer=result_message))
            #question.chat_history_dict = {str(i): entry for i, entry in enumerate(history.messages)}

            if not question.chat_history_dict:
                question.chat_history_dict = {}

            num = len(question.chat_history_dict.keys())
            question.chat_history_dict[str(num)] = ChatEntry(question=question.question,
                                                             answer=result_message.content
                                                             )

            prompt_token_info = PromptTokenInfo(
                input_tokens=result_message.usage_metadata.get("input_tokens", 0),
                output_tokens=result_message.usage_metadata.get("output_tokens", 0),
                total_tokens=result_message.usage_metadata.get("total_tokens", 0),
            )

            return JSONResponse(content=SimpleAnswer(
                answer=result_message.content,
                chat_history_dict=question.chat_history_dict,
                prompt_token_info=prompt_token_info
            ).model_dump())

    except Exception as e:
        import traceback
        traceback.print_exc()
        result_to_return = SimpleAnswer(answer=repr(e), chat_history_dict={}, prompt_token_info=None)
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())




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


        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", question.system_context),
                MessagesPlaceholder("chat_history", n_messages=question.n_messages),
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
    Invoca un agent MCP gestendo input multimodali (testo, immagini, documenti).

    Strategia di pulizia:
    1. Normalizza l'input in una lista di contenuti
    2. Estrae e memorizza documenti/immagini base64 in storage separato
    3. Sostituisce base64 con riferimenti testuali nei messaggi
    4. Il pre_model_hook pulisce ulteriormente i messaggi prima di ogni chiamata LLM
    5. Il tool multimodale interno accede allo storage quando necessario
    """
    mcp_client = question.create_mcp_client()

    try:
        tools = await mcp_client.get_tools()
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
                    logger.info(f"Converted image to data URI format")
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
            system_prompt += "\n\n=== DOCUMENT PROCESSING INSTRUCTIONS ===\n"
            system_prompt += f"\nYou have {len(document_metadata)} document(s) attached and {len(tools)} tool(s) available.\n"

            system_prompt += "\nDOCUMENTS PROVIDED:\n"
            for doc_info in document_metadata:
                doc_id = doc_info["id"]
                mime_type = doc_info["mime_type"]
                source_type = doc_info.get("source_type", "base64")
                storage_entry = base64_storage.get(doc_id, {})

                if source_type == "url":

                    url = doc_info.get("url", storage_entry.get("url", ""))
                    system_prompt += f"  • {doc_id}: {mime_type}\n"
                    system_prompt += f"    Type: URL\n"
                    system_prompt += f"    URL: {url}\n"
                    system_prompt += f"    Referenced in message as: [DOCUMENT_{doc_id}]\n"
                else:
                    size = len(storage_entry.get("data", ""))
                    system_prompt += f"  • {doc_id}: {mime_type}\n"
                    system_prompt += f"    Type: BASE64\n"
                    system_prompt += f"    Size: {size} bytes\n"
                    system_prompt += f"    Referenced in message as: [DOCUMENT_{doc_id}]\n"

            system_prompt += "\nHOW TO PROCESS DOCUMENTS:\n"
            system_prompt += "1. You will see document references like:\n"
            system_prompt += "   - [DOCUMENT_doc_1: application/pdf, URL=https://example.com/file.pdf] (URL type)\n"
            system_prompt += "   - [DOCUMENT_doc_2: application/pdf, 52341 bytes] (BASE64 type)\n"
            system_prompt += "\n2. Check the MCP tool's parameters to understand what it accepts:\n"
            system_prompt += "   - If tool has 'url' parameter → use it for URL documents\n"
            system_prompt += "   - If tool has 'pdf_base64' or 'file_data' → use it for BASE64 documents\n"
            system_prompt += "\n3. CRITICAL - How to retrieve and pass document data:\n"
            system_prompt += "   a) Look up the document ID in the 'DOCUMENTS PROVIDED' section above\n"
            system_prompt += "   b) Check if it's Type: URL or Type: BASE64\n"
            system_prompt += "   c) For URL documents:\n"
            system_prompt += "      - Find the URL listed in 'DOCUMENTS PROVIDED'\n"
            system_prompt += "      - Pass it directly to the tool's 'url' parameter (e.g., url='https://...')\n"
            system_prompt += "      - DO NOT try to download or convert it - the tool handles this!\n"
            system_prompt += "   d) For BASE64 documents:\n"
            system_prompt += "      - The base64 data is in storage (you don't see it to avoid context overflow)\n"
            system_prompt += "      - Pass the document ID reference to the tool (the system resolves it automatically)\n"
            system_prompt += "      - Or use placeholder like 'pdf_base64=<base64_data_from_storage> '\n"
            system_prompt += "\nEXAMPLES:\n"
            system_prompt += "  Example 1 - URL Document:\n"
            system_prompt += "    User: 'Convert the PDF to images'\n"
            system_prompt += "    You see: [DOCUMENT_doc_1: application/pdf, URL=https://pdfobject.com/pdf/sample.pdf]\n"
            system_prompt += "    Tool param: 'url' (accepts URL)\n"
            system_prompt += "    ✓ CORRECT: convert_pdf_to_images(url='https://pdfobject.com/pdf/sample.pdf')\n"
            system_prompt += "    ✗ WRONG: convert_pdf_to_images(pdf_base64='...')  ← Don't download it yourself!\n"
            system_prompt += "\n  Example 2 - BASE64 Document:\n"
            system_prompt += "    User: 'Extract text from PDF'\n"
            system_prompt += "    You see: [DOCUMENT_doc_2: application/pdf, 52341 bytes]\n"
            system_prompt += "    Tool param: 'pdf_base64' (accepts base64)\n"
            system_prompt += "    ✓ CORRECT: extract_text(pdf_base64='<base64_data_from_storage>')\n"
            system_prompt += "\nNote: The system automatically manages base64 data to avoid context overflow.\n"

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
            system_prompt += "\n\nINTERNAL TOOL AVAILABLE:\n"
            system_prompt += "  • invoke_multimodal_llm: Analyzes images/documents with vision capabilities\n"
            system_prompt += "  Use this after converting documents to images for visual analysis.\n"
            # --- AGGIORNA IL SYSTEM PROMPT ---
            system_prompt += f"""
                                           IMPORTANT: Base64 Content Management

                                           Large base64-encoded images are automatically extracted and replaced with references
                                           to avoid context overflow. You will see references like:
                                           - [IMAGE_REF:base64_ref_1:length=52341]

                                           When you need to analyze image content:
                                           1. Use the invoke_multimodal_llm tool
                                           2. Pass the reference: images_base64=["<base64_ref_1>"]
                                           3. The tool will automatically resolve and analyze it

                                           Currently available: {len(base64_storage)} references
                                           """

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

        def pre_model_cleaning_hook(state):
            """Hook chiamato prima di ogni invocazione LLM per pulire i messaggi."""
            messages = state.get("messages", [])
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

            return {"llm_input_messages": cleaned_messages}

        # --- STEP 7: Creazione Agent con Pre-Model Hook ---
        agent_executor = create_react_agent(
            model=chat_model,
            tools=all_tools,
            prompt=system_prompt,
            pre_model_hook=pre_model_cleaning_hook,
            version="v2"
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
        logger.info(f"Agent completed successfully")

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
        return handle_exception(e, "Exception in MCP dialog")



@inject_llm_async
async def ask_mcp_agent_llm_simple(question: QuestionToLLM, chat_model=None):
    """
    Versione ottimizzata che risolve il problema del context overflow.

    PROBLEMA: Quando i tool MCP restituiscono immagini in base64, queste vengono
    incluse nei messaggi dell'agent causando l'overflow del contesto (671K tokens).

    SOLUZIONE:
    1. Estrai e memorizza i contenuti base64 in uno storage separato
    2. Sostituisci il base64 nei messaggi con riferimenti testuali
    3. Il tool multimodale interno accede allo storage quando invocato
    4. I messaggi dell'agent rimangono leggeri (solo testo + riferimenti)
    """
    mcp_client = question.create_mcp_client()

    try:
        tools = await mcp_client.get_tools()

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
                    from langchain.schema import HumanMessage, AIMessage, SystemMessage

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
            State modifier che rimuove il base64 da TUTTI i messaggi ad ogni step.
            Intercetta sia gli input che gli output dei tool (ToolMessage).
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
                                logger.warning(f"[STATE_MODIFIER] Detected large RAW string. Storing as ref.")

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
        system_instructions = f"""
                                IMPORTANT: Base64 Content Management
                                
                                Large base64-encoded images are automatically extracted and replaced with references
                                to avoid context overflow. You will see references like:
                                - [IMAGE_REF:base64_ref_1:length=52341]
                                
                                When you need to analyze image content:
                                1. Use the invoke_multimodal_llm tool
                                2. Pass the reference: images_base64=["<base64_ref_1>"]
                                3. The tool will automatically resolve and analyze it
                                
                                Currently available: {len(base64_storage)} references
                                """

        def pre_model_cleaning_hook(state):
            """
            Pre-model hook che pulisce i messaggi prima di ogni chiamata LLM.
            Questo viene chiamato automaticamente da LangGraph prima del nodo "agent".

            Deve ritornare un dict con:
            - "messages": per AGGIORNARE i messaggi nello stato
            - "llm_input_messages": per usare messaggi diversi come input all'LLM
            """
            current_messages = state.get("messages", [])

            logger.debug(f"[PRE_MODEL_HOOK] ========== CALLED ==========")
            logger.debug(f"[PRE_MODEL_HOOK] Processing {len(current_messages)} messages")

            # DEBUG: Analizza messaggi PRIMA della pulizia
            total_size_before = 0
            for idx, msg in enumerate(current_messages):
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str):
                        total_size_before += len(content)
                        if len(content) > 10000:
                            has_base64 = 'base64' in content or 'data:image' in content
                            logger.debug(f"[PRE_MODEL_HOOK] ⚠️ Message {idx} ({msg.type}): LARGE STRING content, length={len(content)}, has_base64={has_base64}")
                            if has_base64:
                                # Mostra un piccolo sample
                                sample_start = content[:100]
                                logger.debug(f"[PRE_MODEL_HOOK] Sample start: {sample_start}...")
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'image_url':
                                    url = item.get('image_url', {}).get('url', '')
                                    total_size_before += len(url)
                                    if len(url) > 10000:
                                        logger.debug(f"[PRE_MODEL_HOOK] ⚠️ Message {idx} has LARGE image_url: {len(url)} chars")

            logger.debug(f"[PRE_MODEL_HOOK] Total content size BEFORE: {total_size_before} chars (~{total_size_before//4} tokens)")

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
                            logger.debug(f"[PRE_MODEL_HOOK] ❌ STILL LARGE AFTER! Message {idx}, length={len(content)}")

            logger.debug(f"[PRE_MODEL_HOOK] Total content size AFTER: {total_size_after} chars (~{total_size_after//4} tokens)")
            logger.debug(f"[PRE_MODEL_HOOK] Reduction: {total_size_before - total_size_after} chars")

            # Conta quanti base64 sono stati estratti in questo step
            extracted_count = len(base64_storage)
            logger.debug(f"[PRE_MODEL_HOOK] Total refs in storage: {extracted_count}")

            # DEBUG FINALE: Verifica tool_calls negli AIMessage puliti
            for idx, cleaned_msg in enumerate(cleaned_messages):
                if hasattr(cleaned_msg, 'tool_calls') and cleaned_msg.tool_calls:
                    logger.debug(f"[PRE_MODEL_HOOK] Message {idx} (AIMessage) has {len(cleaned_msg.tool_calls)} tool_calls")
                    for tc_idx, tc in enumerate(cleaned_msg.tool_calls):
                        args = tc.get('args', {})
                        if 'images_base64' in args:
                            imgs = args['images_base64']
                            if isinstance(imgs, list) and len(imgs) > 0:
                                first_img = imgs[0] if len(imgs) > 0 else ''
                                is_ref = first_img.startswith('<base64_ref_')
                                logger.debug(f"[PRE_MODEL_HOOK] ⭐ tool_call {tc_idx} has images_base64: {len(imgs)} items, first={'REFERENCE' if is_ref else 'RAW BASE64'}, sample={first_img[:50]}")

            # Ritorna i messaggi puliti che saranno usati come input all'LLM
            return {
                "llm_input_messages": cleaned_messages,
            }

        # Pulisci i messaggi iniziali
        cleaned_input = process_messages_state_modifier({"messages": processed_messages})
        agent_input = {"messages": cleaned_input["messages"]}

        logger.info(f"Starting agent with {len(agent_input['messages'])} initial cleaned messages")

        # Crea agent con pre_model_hook (disponibile in LangGraph v2)
        base_agent = create_react_agent(
            model=chat_model,
            tools=all_tools,
            prompt=system_instructions,
            pre_model_hook=pre_model_cleaning_hook,  # ← CHIAVE! Pulisce ad ogni step
            version="v2"  # Usa la versione v2 che supporta pre_model_hook
        )

        # Invoca l'agent
        logger.info("Invoking agent...")
        response = await base_agent.ainvoke(agent_input)

        # PULIZIA FINALE: Pulisci i messaggi nella risposta (per sicurezza)
        if "messages" in response:
            logger.info(f"Final cleaning of {len(response['messages'])} messages")
            final_cleaned = process_messages_state_modifier({"messages": response["messages"]})
            response["messages"] = final_cleaned["messages"]

        logger.info(f"Agent response received")
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
        return handle_exception(e, "Exception in MCP dialog")

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
            # "chat_history": "Human: My name is Bob\nAI: Hello Bob!",

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

            sys_template = """{system_context}.\n\n                              {context}\n                           """

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