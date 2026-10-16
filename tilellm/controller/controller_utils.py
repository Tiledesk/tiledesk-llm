from datetime import datetime
import asyncio
import time

import tilellm.analytics as analytics


from fastapi.responses import StreamingResponse
import traceback
import uuid
from typing import Any, List, Optional, Dict


from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.tools import BaseTool

from starlette.responses import JSONResponse
from tilellm.models.schemas import (RetrievalResult,
                                    QuotedAnswer,
                                    Citation,
                                    _QuotedAnswer)
from tilellm.models import ChatEntry, ServerConfig
from tilellm.shared.untrusted_content import (
    RAG_USER_TURN,
    StreamLinkFilter,
    neutralize_external_links,
    url_hosts,
    with_security_clause,
)

from tilellm.shared.sparse_util import HybridRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_classic.retrievers import ContextualCompressionRetriever

import tilellm.shared.const as const
from tilellm.shared.tags_query_parser import build_tags_filter

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    # SystemMessage
)

from tilellm.tools.reranker import TileReranker


import logging

logger = logging.getLogger(__name__)


def _extract_content_text(content) -> str:
    """Extract plain text from an AIMessage content that may be a str or a list of content blocks.

    Handles multiple OpenAI API formats:
    - str: Chat Completions API (standard streaming)
    - list[{"type":"text","text":"..."}]: Chat Completions structured content
    - list[{"type":"output_text","text":"..."}]: Responses API streaming
    - list[{"type":"output_text","output_text":"..."}]: Responses API alt format
    - list[{"delta":"..."}]: Responses API delta events
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("output_text")
                    or item.get("delta")
                    or ""
                )
                parts.append(text)
        return "".join(parts)
    return str(content)


# Function to trim chat history
def trim_history(chat_history_dict: Optional[Dict[str, ChatEntry]], max_messages: int) -> Dict[str, ChatEntry]:
    if not chat_history_dict or max_messages <= 0:
        return {}
    
    # Sort keys as integers to get chronological order
    sorted_keys = sorted(chat_history_dict.keys(), key=lambda x: int(x))
    
    # Take the last 'max_messages' turns
    trimmed_keys = sorted_keys[-max_messages:]
    
    return {k: chat_history_dict[k] for k in trimmed_keys}

# Function to summarize history
async def summarize_history(history_text: str, llm) -> str:
    """
    Use the LLM to create a concise summary of the old history.
    """
    if not history_text or history_text == "No previous conversation.":
        return ""

    summarization_prompt = f"""You are tasked with summarizing a conversation history.
                            Create a concise summary that captures:
                            - Main topics discussed
                            - Key information exchanged
                            - Important context for continuing the conversation
                            
                            Conversation history to summarize:
                            {history_text}
                            
                            Provide a brief summary (max 200 words):"""

    try:
        from langchain_core.messages import HumanMessage
        summary_msg = await llm.ainvoke([
            HumanMessage(content=summarization_prompt)
        ])
        return summary_msg.content.strip()
    except Exception as e:
        logger.error(f"Error summarizing history: {e}")
        # Fallback: Returns a truncated version.
        return f"Earlier conversation summary (auto-generated): {history_text[:500]}..."

def history_for_prompt(question_answer):
    """The chat history every LLM call of the request sees, trimmed to max_history_messages."""
    history = question_answer.chat_history_dict
    if getattr(question_answer, "conversation_summary", False):
        # ponytail: this path has no slot for the summary of older turns, so they stay in full
        return history
    max_messages = getattr(question_answer, "max_history_messages", 0)
    return trim_history(history, max_messages) if history and max_messages > 0 else history


# Function to preprocess chat history
def preprocess_chat_history(question_answer):
    logger.debug(question_answer.chat_history_dict)
    question_answer_list = []
    chat_history_list = []
    
    history_to_process = question_answer.chat_history_dict
    if history_to_process is not None:
        # If max_history_messages is set, trim the history
        if hasattr(question_answer, 'max_history_messages') and question_answer.max_history_messages > 0:
            history_to_process = trim_history(history_to_process, question_answer.max_history_messages)
            
        sorted_keys = sorted(history_to_process.keys(), key=lambda x: int(x))
        for key in sorted_keys:
            entry = history_to_process[key]
            chat_history_list.append(HumanMessage(content=entry.question))
            chat_history_list.append(AIMessage(content=entry.answer))
            question_answer_list.append((entry.question, entry.answer))
    return chat_history_list, question_answer_list


# Function to convert chat history to text string
def chat_history_to_text(question_answer, max_messages=None):
    """
    Converte la chat history in una stringa formattata.
    Utile per iniettare la history direttamente nel system prompt come variabile.
    """
    if not question_answer.chat_history_dict:
        return "No previous conversation."

    # Gestisci anche il caso di dict vuoto
    if len(question_answer.chat_history_dict) == 0:
        return "No previous conversation."

    history_to_process = question_answer.chat_history_dict
    if max_messages and max_messages > 0:
        history_to_process = trim_history(history_to_process, max_messages)

    history_lines = []
    try:
        sorted_keys = sorted(history_to_process.keys(), key=lambda x: int(x))
        for key in sorted_keys:
            entry = history_to_process[key]
            # Usa il metodo get_question_text() di ChatEntry per gestire correttamente
            # sia stringhe che contenuti multimodali
            question_text = entry.get_question_text()

            history_lines.append(f"User: {question_text}")
            history_lines.append(f"Assistant: {entry.answer}")
    except Exception as e:
        logger.error(f"Error converting chat history to text: {e}")
        return "Error reading previous conversation."

    return "\n".join(history_lines) if history_lines else "No previous conversation."


# Function to initialize embeddings and encoders
#async def initialize_embeddings_and_index(question_answer, repo, llm_embeddings):
#    emb_dimension = repo.get_embeddings_dimension(question_answer.embedding)
#    sparse_encoder = TiledeskSparseEncoders(question_answer.sparse_encoder)
#    vector_store = await repo.create_index(question_answer.engine, llm_embeddings, emb_dimension)
#    index = vector_store.async_index

#    return emb_dimension, sparse_encoder, index


# Function to initialize embeddings and retrievers
async def initialize_retrievers(question_answer, repo, llm_embeddings, embedding_config_key=None):

    emb_dimension = await repo.get_embeddings_dimension(question_answer.embedding)
    vector_store = await repo.create_index(question_answer.engine, llm_embeddings, emb_dimension, embedding_config_key)

    # Aggiunto e modificato il codice per avere invece che top_k il retrieval_k###
    retrieval_k = question_answer.top_k * question_answer.reranking_multiplier if question_answer.reranking and question_answer.top_k > 0 else question_answer.top_k
    # TODO Bisogna cercare una soluzione più elegante

    # Build tags filter if tags are provided
    filter_dict = None
    if question_answer.tags:
        filter_dict = build_tags_filter(question_answer.tags, field="tags")

    if question_answer.engine.name == "qdrant":
        search_kwargs = {'k': retrieval_k}
        # For Qdrant, build store-specific filter that includes namespace
        if filter_dict:
            search_kwargs['filter'] = repo.build_filter(question_answer.namespace, filter_dict)
        else:
            # Still need namespace filter for Qdrant
            search_kwargs['filter'] = repo.build_filter(question_answer.namespace, None)
    else:
        search_kwargs = {'k': retrieval_k, 'namespace': question_answer.namespace}
        # Add filter to search_kwargs if present (Pinecone-style)
        if filter_dict:
            search_kwargs['filter'] = filter_dict

    vs_retriever = vector_store.as_retriever(
        search_type=question_answer.search_type,
        search_kwargs=search_kwargs
    )

    if float(question_answer.similarity_threshold) != 1.0:
        redundant_filter = EmbeddingsRedundantFilter(
            embeddings=llm_embeddings,
            similarity_threshold=question_answer.similarity_threshold
        )

        pipeline_compressor = DocumentCompressorPipeline(transformers=[redundant_filter])
        retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=vs_retriever
        )
    else:
        retriever=vs_retriever

    return retriever




# Function to fetch vectors for the given question
async def fetch_question_vectors_nopar(question_answer, sparse_encoder, llm_embeddings):
    retrieval_q = question_answer.retrieval_query or question_answer.question
    if sparse_encoder is None:
        sparse_vector = None
    else:
        sparse_vector = await sparse_encoder.aencode_queries(retrieval_q)
    dense_vector = await llm_embeddings.aembed_query(retrieval_q)
    return dense_vector, sparse_vector

async def fetch_question_vectors(question_answer, sparse_encoder, llm_embeddings):
    """Generate dense and sparse vectors, reusing precomputed embedding when available."""
    retrieval_q = question_answer.retrieval_query or question_answer.question
    precomputed = getattr(question_answer, "precomputed_query_embedding", None)
    if precomputed is not None:
        dense_task = asyncio.sleep(0, result=precomputed)
    else:
        dense_task = llm_embeddings.aembed_query(retrieval_q)

    if sparse_encoder:
        sparse_task = sparse_encoder.aencode_queries(retrieval_q)
    else:
        sparse_task = asyncio.sleep(0, result=None)

    dense_vector, sparse_vector = await asyncio.gather(dense_task, sparse_task)
    return dense_vector, sparse_vector

#
# Function to retrieve documents based on search results
def retrieve_documents_old(question_answer, results):
    if not results.get("matches"):
        raise ValueError("No chunks found with the current filters.")
    documents = [Document(page_content=match["metadata"]["text"], metadata=match["metadata"]) for match in results["matches"]]
    retriever = HybridRetriever(documents=documents, k=question_answer.top_k)
    return retriever

# Function to retrieve documents based on search results
def retrieve_documents(question_answer, results, contextualized_query=None):
    if not results.get("matches"):
        raise ValueError("No chunks found with the current filters.")
    # Aggiunto
    retrieval_k = question_answer.top_k * question_answer.reranking_multiplier if question_answer.reranking else question_answer.top_k

    # Get all available result or retrieval_k
    matches_to_use = results["matches"][:retrieval_k] if len(results["matches"]) >= retrieval_k else results["matches"]

    documents = [Document(page_content=match["metadata"]["text"], metadata=match["metadata"]) for match in
                 matches_to_use]
    # Applica re-ranking se necessario
    if question_answer.reranker_config and len(documents) > question_answer.top_k:
        # Per l'hybrid search, dobbiamo applicare il re-ranking manualmente
        ranking_query = contextualized_query if contextualized_query else question_answer.question

        reranker = TileReranker(model_name=question_answer.reranker_config)
        reranked_docs = reranker.rerank_documents(ranking_query, documents, question_answer.top_k)
        retriever = HybridRetriever(documents=reranked_docs, k=question_answer.top_k)
    else:
        retriever = HybridRetriever(documents=documents, k=question_answer.top_k)
    return retriever


# Async version of retrieve_documents for use in async contexts
async def aretrieve_documents(question_answer, results, contextualized_query=None):
    """
    Async version of retrieve_documents that uses async reranking when available.
    """
    if not results.get("matches"):
        raise ValueError("No chunks found with the current filters.")
    # Aggiunto
    retrieval_k = question_answer.top_k * question_answer.reranking_multiplier if question_answer.reranking else question_answer.top_k

    # Get all available result or retrieval_k
    matches_to_use = results["matches"][:retrieval_k] if len(results["matches"]) >= retrieval_k else results["matches"]

    documents = [Document(page_content=match["metadata"]["text"], metadata=match["metadata"]) for match in
                 matches_to_use]
    # Applica re-ranking se necessario
    if question_answer.reranker_config and len(documents) > question_answer.top_k:
        # Per l'hybrid search, dobbiamo applicare il re-ranking manualmente
        ranking_query = contextualized_query if contextualized_query else question_answer.question

        reranker = TileReranker(model_name=question_answer.reranker_config)
        # Use async reranking if available
        reranked_docs = await reranker.arerank_documents(ranking_query, documents, question_answer.top_k)
        retriever = HybridRetriever(documents=reranked_docs, k=question_answer.top_k)
    else:
        retriever = HybridRetriever(documents=documents, k=question_answer.top_k)
    return retriever



# Function to create chains for contextualization and Q&A
async def create_chains(llm, question_answer, retriever, contextualized_query=None):
    # --- Contextualization for Retrieval ---
    if question_answer.contextualize_prompt and contextualized_query is not None:
        # The question was already rewritten once (create_contextualize_query): rewriting it again
        # here cost a second sequential LLM call whose result was overridden or ignored.
        query = contextualized_query or question_answer.retrieval_query or question_answer.question
        history_aware_retriever = RunnableLambda(lambda _: query) | retriever
    elif question_answer.contextualize_prompt:
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
    else:
        history_aware_retriever = retriever

    # --- Q&A Prompt Construction ---
    messages = []
    
    # Base system prompt
    qa_system_prompt = question_answer.system_context if question_answer.system_context else const.qa_system_prompt
    
    # If using history in prompt as text (legacy/simple mode)
    if not question_answer.include_history_in_prompt:
        # No history in prompt or handled externally
        messages.append(("system", with_security_clause(qa_system_prompt)))
    elif not question_answer.contextualize_prompt:
        # History injected as text (contextualize_prompt=False)
        # We'll use a placeholder or handle it in generate_answer_with_history
        if question_answer.system_context:
            messages.append(("system", with_security_clause(qa_system_prompt)))
        else:
            messages.append(("system", with_security_clause(const.qa_system_prompt_with_history_injected)))
    else:
        # Standard mode: History as MessagesPlaceholder
        messages.append(("system", with_security_clause(qa_system_prompt)))
        messages.append(MessagesPlaceholder("chat_history"))

    messages.append(("human", RAG_USER_TURN))
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return history_aware_retriever, question_answer_chain, qa_prompt


async def create_contextualize_query(llm, question_answer):
    # --- INIZIO MODIFICA: gestione condizionale di contextualize_prompt ---
    if not question_answer.contextualize_prompt:
        # NUOVO: NON contestualizza la query, restituisce quella originale
        # Massimo risparmio token per hybrid search
        retrieval_q = question_answer.retrieval_query or question_answer.question
        logger.info(f"Contextualize disabled - using retrieval query: {retrieval_q}")
        return retrieval_q

    # CODICE ORIGINALE: contestualizza la query usando l'LLM
    contextualize_q_system_prompt = const.contextualize_q_system_prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    if question_answer.retrieval_query:
        # A HyDE passage (or any explicit retrieval text): rewriting it as a standalone *question*
        # would shrink it back to a question and undo HyDE.
        contextualized_query = question_answer.retrieval_query
    elif question_answer.chat_history_dict:
        c_h, _ = preprocess_chat_history(question_answer)
        # Se c'è history, contestualizza la query
        retrieval_q = question_answer.retrieval_query or question_answer.question
        raw_content = (await llm.ainvoke(
            contextualize_q_prompt.format_messages(
                chat_history=c_h,
                input=retrieval_q
            )
        )).content
        # GPT-5.x returns content as a list of content blocks; normalize to str
        if isinstance(raw_content, list):
            contextualized_query = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw_content
            )
        else:
            contextualized_query = raw_content
    else:
        # Se non c'è history, usa la query di retrieval (o fallback a question originale)
        contextualized_query = question_answer.retrieval_query or question_answer.question

    logger.info(f"Contextualized query (FULL): {contextualized_query}")
    # --- FINE MODIFICA ---

    return contextualized_query


# Function to get or create session history
def get_or_create_session_history(store, session_id, chat_history_dict):
    if session_id not in store:
        store[session_id] = load_session_history(chat_history_dict)
    return store[session_id]


# Function to generate answer with chat history consideration
async def generate_answer_with_history(llm, question_answer, rag_chain, retriever, get_session_history, qa_prompt, callback_handler, question_answer_list, contextualized_query=None):

    # Every branch retrieves with the same query: the rewrite (HyDE passage or contextualized question)
    # when there is one, the raw input otherwise.
    retrieval = RunnableLambda(lambda x: contextualized_query or x["input"]) | retriever

    def extract_content(input_dict):
        # print(input_dict["answer"])
        return {"input": input_dict.get("input"), "answer": input_dict.get("answer")}

    # --- Hybrid Approach Logic ---
    include_history = getattr(question_answer, 'include_history_in_prompt', True)
    use_summary = getattr(question_answer, 'conversation_summary', False)
    max_history = getattr(question_answer, 'max_history_messages', 10)

    # 1. Prepare History String / Summary if needed
    chat_history_text = ""
    summary_text = ""
    
    if include_history or use_summary:
        if question_answer.chat_history_dict:
            # If we want a summary, we might summarize the older part of history
            # Only the contextualize_prompt=False prompts have a place for the summary: on the
            # other path it was generated and then dropped — an LLM call for nothing.
            if use_summary and not question_answer.contextualize_prompt:
                # If we have many messages, summarize the ones before the last 'max_history'
                sorted_keys = sorted(question_answer.chat_history_dict.keys(), key=lambda x: int(x))
                if len(sorted_keys) > max_history:
                    old_keys = sorted_keys[:-max_history]
                    old_history_dict = {k: question_answer.chat_history_dict[k] for k in old_keys}
                    # Temporary mock for summary or real call
                    old_history_text = chat_history_to_text(question_answer, max_messages=len(old_keys))
                    summary_text = await summarize_history(old_history_text, llm)
            
            chat_history_text = chat_history_to_text(question_answer, max_messages=max_history)

    # --- INIZIO MODIFICA: gestione history come stringa o MessagesPlaceholder ---
    # Se contextualize_prompt=False, usiamo la history iniettata come testo
    if not question_answer.contextualize_prompt:
        # Determina il system prompt da usare
        if question_answer.system_context:
            # CASO 1: system_context custom
            system_with_context = question_answer.system_context
            if "{context}" not in system_with_context:
                system_with_context += "\n\nRetrieved context:\n{context}"
            
            # Build the augmented system prompt
            system_final = system_with_context
            if summary_text:
                system_final += f"\n\nConversation Summary:\n{summary_text}"
            
            if include_history and chat_history_text:
                system_final += f"\n\n<previous_conversation>\n{chat_history_text}\n</previous_conversation>"
        else:
            # CASO 2: prompt default
            system_final = const.qa_system_prompt_with_history_injected
            if summary_text:
                system_final = f"Summary of previous conversation: {summary_text}\n\n" + system_final

        qa_prompt_final = ChatPromptTemplate.from_messages([
            ("system", with_security_clause(system_final)),
            ("human", RAG_USER_TURN),
        ])

        chain_input = {"input": question_answer.question}
        if not question_answer.system_context:
            chain_input["chat_history_text"] = chat_history_text


        if question_answer.stream:

            if question_answer.citations:
                rag_chain_from_docs_stream = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
                    | qa_prompt_final
                    | llm
                )
                retrieve_docs = retrieval
                runnable = (
                    RunnablePassthrough.assign(context=retrieve_docs)
                    .assign(answer=rag_chain_from_docs_stream)
                )
            else:
                runnable = create_retrieval_chain(retrieval, create_stuff_documents_chain(llm, qa_prompt_final))

            async def get_stream_llm_simple():
                full_response = ""
                message_id = str(uuid.uuid4())
                start_time = datetime.now()
                result_context = None
                _llm_t0 = time.monotonic()

                # Add citation prompt tail if needed
                local_chain_input = chain_input.copy()
                if question_answer.citations:
                    local_chain_input["input"] = local_chain_input["input"] + "\n" + const.stream_citations_tail

                yield _create_event("metadata", {"message_id": message_id, "status": "started", "timestamp": start_time.isoformat()})

                async for chunk in runnable.astream(local_chain_input):
                    content_to_stream = None
                    if "answer" in chunk:
                        if isinstance(chunk["answer"], str):
                            content_to_stream = chunk["answer"]
                        elif hasattr(chunk["answer"], "content"):
                            content_to_stream = _extract_content_text(chunk["answer"].content)

                    if content_to_stream:
                        full_response += content_to_stream
                        yield _create_event("chunk", {"content": content_to_stream, "message_id": message_id})

                    if "context" in chunk and result_context is None: # Capture context once
                        result_context = chunk["context"]

                _llm_latency_ms = int((time.monotonic() - _llm_t0) * 1000)

                # Final processing after stream ends
                citations = extract_citations(full_response) if question_answer.citations else []
                if question_answer.citations:
                    full_response = _strip_citation_markers(full_response)
                end_time = datetime.now()
                full_response, success = verify_answer(full_response)

                final_result_dict = {
                    "input": question_answer.question,
                    "answer": full_response,
                    "context": result_context if result_context else [],
                    "chat_history": []
                }

                question_answer_list.append((question_answer.question, full_response))

                result_to_return = format_result(
                    result=final_result_dict,
                    citations=citations,
                    question_answer=question_answer,
                    callback_handler=callback_handler,
                    question_answer_list=question_answer_list,
                    success=success,
                    duration=(end_time - start_time).total_seconds()
                )

                # Analytics: emit token_usage + model_call (fire-and-forget)
                _cb = callback_handler
                _model_str = getattr(question_answer, "model", None) or ""
                if not isinstance(_model_str, str):
                    _model_str = getattr(_model_str, "name", str(_model_str))
                _et, _pl = analytics.events.token_usage(
                    model=_model_str,
                    prompt_tokens=getattr(_cb, "prompt_tokens", 0),
                    completion_tokens=getattr(_cb, "completion_tokens", 0),
                    total_tokens=getattr(_cb, "total_tokens", 0),
                    operation="ask",
                    source="rag",
                    request_id=getattr(question_answer, "request_id", None),
                    agent_id=getattr(question_answer, "agent_id", None),
                )
                analytics.publish_nowait(_et, question_answer.id_project, _pl)
                _et, _pl = analytics.events.model_call(
                    model=_model_str,
                    provider=getattr(question_answer, "llm", None),
                    operation="ask",
                    latency_ms=_llm_latency_ms,
                    success=True,
                    request_id=getattr(question_answer, "request_id", None),
                )
                analytics.publish_nowait(_et, question_answer.id_project, _pl)

                yield _create_event("metadata", {
                    "message_id": message_id,
                    "status": "completed",
                    "timestamp": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "full_response": full_response,
                    "model_used": result_to_return.model_dump()
                })

            return StreamingResponse(
                _neutralize_stream(get_stream_llm_simple()),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else: # NOT STREAMING
            start_time = datetime.now() if question_answer.debug else 0
            _llm_t0 = time.monotonic()
            if question_answer.citations:
                retrieve_docs = retrieval
                rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
                    | qa_prompt_final
                    | llm.with_structured_output(_QuotedAnswer)
                )
                chain_w_citations = (
                    RunnablePassthrough.assign(context=retrieve_docs)
                    .assign(answer=rag_chain_from_docs)
                    .assign(only_answer=lambda text: text["answer"].answer)
                )
                result = await chain_w_citations.ainvoke(chain_input)


                # Check if no context found
                if not result.get("context"):
                    raise ValueError("No chunks found with the current filters.")

                citations = [Citation.from_llm(c) for c in result['answer'].citations]
                result['answer'] = result['answer'].answer
            else:
                rag_chain_simple = create_retrieval_chain(retrieval, create_stuff_documents_chain(llm, qa_prompt_final))
                result = await rag_chain_simple.ainvoke(chain_input)
                # Check if no context found
                if not result.get("context"):
                    raise ValueError("No chunks found with the current filters.")
                
                citations = None

            end_time = datetime.now() if question_answer.debug else 0
            _llm_latency_ms = int((time.monotonic() - _llm_t0) * 1000)
            duration = (end_time - start_time).total_seconds() if question_answer.debug else 0.0
            result['answer'], success = verify_answer(result['answer'])
            result['chat_history'] = []
            question_answer_list.append((question_answer.question, result['answer']))
            result_to_return = format_result(
                result=result,
                citations=citations,
                question_answer=question_answer,
                callback_handler=callback_handler,
                question_answer_list=question_answer_list,
                success=success,
                duration=duration
            )

            # Analytics: emit token_usage + model_call (fire-and-forget)
            _cb = callback_handler
            _model_str = getattr(question_answer, "model", None) or ""
            if not isinstance(_model_str, str):
                _model_str = getattr(_model_str, "name", str(_model_str))
            _et, _pl = analytics.events.token_usage(
                model=_model_str,
                prompt_tokens=getattr(_cb, "prompt_tokens", 0),
                completion_tokens=getattr(_cb, "completion_tokens", 0),
                total_tokens=getattr(_cb, "total_tokens", 0),
                operation="ask",
                source="rag",
                request_id=getattr(question_answer, "request_id", None),
                agent_id=getattr(question_answer, "agent_id", None),
            )
            analytics.publish_nowait(_et, question_answer.id_project, _pl)
            _et, _pl = analytics.events.model_call(
                model=_model_str,
                provider=getattr(question_answer, "llm", None),
                operation="ask",
                latency_ms=_llm_latency_ms,
                success=True,
                request_id=getattr(question_answer, "request_id", None),
            )
            analytics.publish_nowait(_et, question_answer.id_project, _pl)

            return JSONResponse(content=result_to_return.model_dump())
    
    # --- Standard Flow (contextualize_prompt=True) ---
    # Add summary to system prompt if available
    if summary_text:
        # We need to wrap the rag_chain to include the summary in the system prompt
        # but create_chains already created it. 
        # For simplicity, we can inject it into the input if the prompt supports it, 
        # or recreate the chain.
        logger.info(f"Conversation summary generated: {len(summary_text)} chars")
        # In this implementation, we'll append it to the system message if we can.
        # However, create_retrieval_chain hides the internal stuff chain.
        pass

    if question_answer.stream:
        if question_answer.citations:
            # RunnableWithMessageHistory buffers streaming chunks and breaks streaming.
            # Use text-injected history (like contextualize_prompt=False) instead of
            # MessagesPlaceholder to avoid "Reply to previous question" confusion with
            # stream_citations_tail when there are prior conversation turns.
            if question_answer.system_context:
                system_stream = question_answer.system_context
                if "{context}" not in system_stream:
                    system_stream += "\n\nRetrieved context:\n{context}"
            else:
                system_stream = const.qa_system_prompt_with_history_injected

            qa_prompt_stream = ChatPromptTemplate.from_messages([
                ("system", with_security_clause(system_stream)),
                ("human", RAG_USER_TURN),
            ])

            retrieve_docs = retrieval
            rag_chain_from_docs_stream = (
                RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
                | qa_prompt_stream
                | llm
            )
            chain_w_citations_stream = (
                RunnablePassthrough.assign(context=retrieve_docs)
                .assign(answer=rag_chain_from_docs_stream)
            )
            _chain_input_ctx = {
                "input": question_answer.question + "\n" + const.stream_citations_tail,
                "chat_history_text": chat_history_text,
            }

            async def get_stream_llm_citations_ctx():
                full_response = ""
                message_id = str(uuid.uuid4())
                start_time = datetime.now()
                result_context = None
                _llm_t0 = time.monotonic()

                yield _create_event("metadata", {"message_id": message_id, "status": "started", "timestamp": start_time.isoformat()})

                async for chunk in chain_w_citations_stream.astream(_chain_input_ctx):
                    content_to_stream = None
                    if "answer" in chunk:
                        if isinstance(chunk["answer"], str):
                            content_to_stream = chunk["answer"]
                        elif hasattr(chunk["answer"], "content"):
                            content_to_stream = _extract_content_text(chunk["answer"].content)
                    if content_to_stream:
                        full_response += content_to_stream
                        yield _create_event("chunk", {"content": content_to_stream, "message_id": message_id})
                    if "context" in chunk and result_context is None:
                        result_context = chunk["context"]

                _llm_latency_ms = int((time.monotonic() - _llm_t0) * 1000)

                citations = extract_citations(full_response)
                full_response = _strip_citation_markers(full_response)
                end_time = datetime.now()
                full_response, success = verify_answer(full_response)

                final_result_dict = {
                    "input": question_answer.question,
                    "answer": full_response,
                    "context": result_context if result_context else [],
                    "chat_history": [],
                }
                question_answer_list.append((question_answer.question, full_response))

                result_to_return = format_result(
                    result=final_result_dict,
                    citations=citations,
                    question_answer=question_answer,
                    callback_handler=callback_handler,
                    question_answer_list=question_answer_list,
                    success=success,
                    duration=(end_time - start_time).total_seconds()
                )

                _cb = callback_handler
                _model_str = getattr(question_answer, "model", None) or ""
                if not isinstance(_model_str, str):
                    _model_str = getattr(_model_str, "name", str(_model_str))
                _et, _pl = analytics.events.token_usage(
                    model=_model_str,
                    prompt_tokens=getattr(_cb, "prompt_tokens", 0),
                    completion_tokens=getattr(_cb, "completion_tokens", 0),
                    total_tokens=getattr(_cb, "total_tokens", 0),
                    operation="ask",
                    source="rag",
                    request_id=getattr(question_answer, "request_id", None),
                    agent_id=getattr(question_answer, "agent_id", None),
                )
                analytics.publish_nowait(_et, question_answer.id_project, _pl)
                _et, _pl = analytics.events.model_call(
                    model=_model_str,
                    provider=getattr(question_answer, "llm", None),
                    operation="ask",
                    latency_ms=_llm_latency_ms,
                    success=True,
                    request_id=getattr(question_answer, "request_id", None),
                )
                analytics.publish_nowait(_et, question_answer.id_project, _pl)

                yield _create_event("metadata", {
                    "message_id": message_id,
                    "status": "completed",
                    "timestamp": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "full_response": full_response,
                    "model_used": result_to_return.model_dump()
                })

            return StreamingResponse(
                _neutralize_stream(get_stream_llm_citations_ctx()),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            return await get_streaming_response(conversational_rag_chain, question_answer, callback_handler, question_answer_list)
    else:

        if question_answer.citations:
            retrieve_docs = retrieval
            rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
                    | qa_prompt
                    | llm.with_structured_output(_QuotedAnswer)
            )

            chain_w_citations = (RunnablePassthrough.assign(context=retrieve_docs)
                                 .assign(answer=rag_chain_from_docs)
                                 .assign(only_answer=lambda text: text["answer"].answer)
                                 )

            conversational_rag_chain = RunnableWithMessageHistory(
                chain_w_citations,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="only_answer",
            )
        else:
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        start_time = datetime.now() if question_answer.debug else 0
        _llm_t0 = time.monotonic()
        result = await conversational_rag_chain.ainvoke(
            {"input": question_answer.question},
            config={"configurable": {"session_id": uuid.uuid4().hex}}
        )
        _llm_latency_ms = int((time.monotonic() - _llm_t0) * 1000)
        # Check if no context found
        if not result.get("context"):
            raise ValueError("No chunks found with the current filters.")


        end_time = datetime.now() if question_answer.debug else 0
        duration = (end_time - start_time).total_seconds() if question_answer.debug else 0.0

        citations = [Citation.from_llm(c) for c in result['answer'].citations] if question_answer.citations else None
        result['answer'], success = verify_answer(result['answer'].answer
                                                  if question_answer.citations else result['answer'])
        #return result, citations, success

        ######################

        question_answer_list.append((result['input'], result['answer']))

        result_to_return = format_result(result=result,
                                         citations=citations,
                                         question_answer=question_answer,
                                         callback_handler=callback_handler,
                                         question_answer_list=question_answer_list,
                                         success=success,
                                         duration=duration)

        # Analytics: emit token_usage + model_call (fire-and-forget)
        _cb = callback_handler
        _model_str = getattr(question_answer, "model", None) or ""
        if not isinstance(_model_str, str):
            _model_str = getattr(_model_str, "name", str(_model_str))
        _et, _pl = analytics.events.token_usage(
            model=_model_str,
            prompt_tokens=getattr(_cb, "prompt_tokens", 0),
            completion_tokens=getattr(_cb, "completion_tokens", 0),
            total_tokens=getattr(_cb, "total_tokens", 0),
            operation="ask",
            source="rag",
            request_id=getattr(question_answer, "request_id", None),
            agent_id=getattr(question_answer, "agent_id", None),
        )
        analytics.publish_nowait(_et, question_answer.id_project, _pl)
        _et, _pl = analytics.events.model_call(
            model=_model_str,
            provider=getattr(question_answer, "llm", None),
            operation="ask",
            latency_ms=_llm_latency_ms,
            success=True,
            request_id=getattr(question_answer, "request_id", None),
        )
        analytics.publish_nowait(_et, question_answer.id_project, _pl)

        return JSONResponse(content=result_to_return.model_dump())



# Function to format the result into the expected output structure
def format_result(result, citations, question_answer, callback_handler, question_answer_list, success, duration=0.0):
    docs = result["context"]
    ids, sources, content_chunks, source_file_map, content_chunks_metadata = extract_ids_sources(
        docs, question_answer.debug
    )
    source = format_sources(citations, sources, question_answer.citations)
    if citations:
        for cit in citations:
            cit.source_file_name = source_file_map.get(cit.source_name)
            # source_id indexes directly into `docs` (format_docs_with_id numbers
            # sources the same way: "Source ID: {i}" over enumerate(docs)) — but it's
            # LLM-produced, so bounds-check defensively against hallucination.
            if 0 <= cit.source_id < len(docs):
                cit.page = docs[cit.source_id].metadata.get('page_number')
    
    # Avoid IndexError if no documents are found
    metadata_id = ids[0] if ids else None

    if callback_handler:
        prompt_token_size = callback_handler.total_tokens
    else:
        prompt_token_size = 0


    logger.info(f"input: {result['input']}")
    logger.info(f"chat_history: {result['chat_history']}")
    logger.info(f"answer: {result['answer']}")

    chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
    chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

    result_to_return = RetrievalResult(
        answer=result['answer'],
        namespace=question_answer.namespace,
        sources=sources,
        ids=ids,
        source=source,
        id=metadata_id,
        citations=citations,
        prompt_token_size=prompt_token_size,
        content_chunks=content_chunks,
        content_chunks_metadata=content_chunks_metadata,
        success=success,
        duration=duration,
        error_message=None,
        chat_history_dict=chat_history_dict
    )
    return result_to_return


# Function to extract IDs and sources from documents
def extract_ids_sources(docs, debug):
    ids = []
    sources = []
    content_chunks = None
    content_chunks_metadata = None
    source_file_map = {}
    if debug:
        content_chunks = []
        content_chunks_metadata = []
        for doc in docs:
            ids.append(doc.metadata.get('id'))
            sources.append(doc.metadata.get('source'))
            content_chunks.append(doc.page_content)
            content_chunks_metadata.append({
                'source': doc.metadata.get('source'),
                'file_name': doc.metadata.get('file_name'),
                'page_number': doc.metadata.get('page_number'),
            })
            src = doc.metadata.get('source')
            fn = doc.metadata.get('file_name')
            if src and fn:
                source_file_map[src] = fn
    else:
        for doc in docs:
            ids.append(doc.metadata.get('id'))
            sources.append(doc.metadata.get('source'))
            src = doc.metadata.get('source')
            fn = doc.metadata.get('file_name')
            if src and fn:
                source_file_map[src] = fn

    # Filter out None values and get unique elements
    ids = list(set([i for i in ids if i is not None]))
    sources = list(set([s for s in sources if s is not None]))
    return ids, sources, content_chunks, source_file_map, content_chunks_metadata


# Function to format sources based on citations
def format_sources(citations, sources, with_citations):
    if with_citations and citations is not None:
        source = " ".join(set([cit.source_name for cit in citations]))
    else:
        source = " ".join(sources)
    return source


# Function to handle exceptions
def handle_exception(e, question_answer):
    logger.error(f"Exception in controller: {repr(e)}", exc_info=True)
    # traceback.print_exc()
    question_answer_list = []
    if question_answer.chat_history_dict is not None:
        for key, entry in question_answer.chat_history_dict.items():
            question_answer_list.append((entry.question, entry.answer))
    chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
    chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

    result_to_return = RetrievalResult(
        answer="No answer",
        success=False,
        namespace=question_answer.namespace,
        error_message=repr(e),
        chat_history_dict=chat_history_dict
    )
    return JSONResponse(status_code=400, content=result_to_return.model_dump())

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

def verify_answer(s):
    if s.endswith("<NOANS>"):
        s = s[:-7]  # Rimuove <NOANS> dalla fine della stringa
        success = False
    else:
        success = True
    # Final answer (also what enters the chat history): no images/links to hosts
    # outside ANSWER_ALLOWED_LINK_HOSTS — the exfiltration channel of injected chunks.
    return neutralize_external_links(s), success

async def get_streaming_response(runnable_with_history, question, callback_handler, question_answer_list):
    async def get_stream_llm(run_with_history, q):
        full_response = ""
        message_id = str(uuid.uuid4())
        start_time = datetime.now()
        result = dict()
        _llm_t0 = time.monotonic()

        orig_question = q.question
        q.question = q.question+"\n"+const.stream_citations_tail if q.citations else q.question

        #print(q.question)

        yield _create_event("metadata", {
            "message_id": message_id,
            "status": "started",
            "timestamp": start_time.isoformat()
        })

        async for chunk in run_with_history.astream({"input": q.question},
                                                         config={
                                                             "configurable": {"session_id": uuid.uuid4().hex}}):

            if not isinstance(chunk, dict):
                # Non-dict chunk: try direct content extraction (fallback for some LangChain versions)
                if hasattr(chunk, "content"):
                    _text = _extract_content_text(chunk.content)
                    if _text:
                        full_response += _text
                        yield _create_event("chunk", {"content": _text, "message_id": message_id})
                continue

            if "answer" in chunk:
                _answer = chunk["answer"]
                if isinstance(_answer, str):
                    if _answer:
                        full_response += _answer
                        yield _create_event("chunk", {"content": _answer, "message_id": message_id})
                elif hasattr(_answer, "content"):
                    _text = _extract_content_text(_answer.content)
                    if _text:
                        full_response += _text
                        yield _create_event("chunk", {"content": _text, "message_id": message_id})
                    else:
                        logger.debug(
                            f"[stream] answer chunk empty — "
                            f"content={_answer.content!r:.200} "
                            f"add_kwargs={getattr(_answer, 'additional_kwargs', {})!r:.100}"
                        )

            elif 'only_answer' in chunk and not full_response:
                # Fallback: RunnableWithMessageHistory may buffer "answer" chunks;
                # the final accumulated answer surfaces here when streaming is
                # not granular (e.g. Responses API models).
                _oa = chunk['only_answer']
                _answer_val = _oa.get("answer") if isinstance(_oa, dict) else None
                if _answer_val is not None:
                    _text = _extract_content_text(
                        _answer_val.content if hasattr(_answer_val, "content") else str(_answer_val)
                    )
                    if _text:
                        full_response += _text
                        yield _create_event("chunk", {"content": _text, "message_id": message_id})

            elif 'context' in chunk:
                result["context"] = chunk["context"]
            elif 'input' in chunk:
                result['input'] = chunk['input']
                result['chat_history'] = chunk['chat_history']

        _llm_latency_ms = int((time.monotonic() - _llm_t0) * 1000)

        citations = extract_citations(full_response) if question.citations else []
        if question.citations:
            full_response = _strip_citation_markers(full_response)
        end_time = datetime.now()
        full_response, success = verify_answer(full_response)


        result["answer"]=full_response



        question_answer_list.append((orig_question,full_response) )

        result_to_return = format_result(result=result,
                                         citations=citations,
                                         question_answer=q,
                                         callback_handler=callback_handler,
                                         question_answer_list=question_answer_list,
                                         success=success)

        # Analytics: emit token_usage + model_call (fire-and-forget)
        _cb = callback_handler
        _model_str = getattr(q, "model", None) or ""
        if not isinstance(_model_str, str):
            _model_str = getattr(_model_str, "name", str(_model_str))
        _et, _pl = analytics.events.token_usage(
            model=_model_str,
            prompt_tokens=getattr(_cb, "prompt_tokens", 0),
            completion_tokens=getattr(_cb, "completion_tokens", 0),
            total_tokens=getattr(_cb, "total_tokens", 0),
            operation="ask",
            source="rag",
            request_id=getattr(q, "request_id", None),
            agent_id=getattr(q, "agent_id", None),
        )
        analytics.publish_nowait(_et, q.id_project, _pl)
        _et, _pl = analytics.events.model_call(
            model=_model_str,
            provider=getattr(q, "llm", None),
            operation="ask",
            latency_ms=_llm_latency_ms,
            success=True,
            request_id=getattr(q, "request_id", None),
        )
        analytics.publish_nowait(_et, q.id_project, _pl)

        yield _create_event("metadata", {
            "message_id": message_id,
            "status": "completed",
            "timestamp": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
            "full_response": full_response,
            "model_used": result_to_return.model_dump()
        })

    return StreamingResponse(
        _neutralize_stream(get_stream_llm(runnable_with_history, question)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


_CITATION_RE = r"[Cc][Ii][Tt]:\s*id:\s*(?:\[)?(\d+)(?:\])?\s*(?:,\s*[Ss][Oo][Uu][Rr][Cc][Ee]:\s*(?:\[)?(.*?)(?:\]?\s*(?:\(.*?\))?))?;"


def extract_citations(testo: str) -> List[Citation]:
    import re

    citations = []
    for match in re.finditer(_CITATION_RE, testo):
        id_cit = int(match.group(1))
        source_cit = match.group(2).strip()
        citations.append(Citation(source_id=id_cit, source_name=source_cit))
    return citations


def _strip_citation_markers(text: str) -> str:
    import re
    return re.sub(_CITATION_RE, "", text).strip()

def _create_event(event_type: str, data: dict) -> str:
    import json
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _neutralize_stream(events):
    """Neutralize external links in the streamed answer: chunk events pass
    through one StreamLinkFilter (a link split across chunks is caught); what it
    holds back is released, cleaned, before the next non-chunk event."""
    import json

    link_filter = StreamLinkFilter()
    message_id = None
    async for event in events:
        if event.startswith("event: chunk\n"):
            data = json.loads(event.split("data: ", 1)[1])
            message_id = data.get("message_id", message_id)
            content = link_filter.feed(data.get("content") or "")
            if content:
                yield _create_event("chunk", {**data, "content": content})
            continue
        pending = link_filter.flush()
        if pending:
            yield _create_event("chunk", {"content": pending, "message_id": message_id})
        yield event
    pending = link_filter.flush()
    if pending:
        yield _create_event("chunk", {"content": pending, "message_id": message_id})



def extract_conversation_flow_old(messages: list) -> str:
    """
    Estrae tutti i ToolMessage e AIMessage dalla conversazione.
    Preserva la formattazione originale (newline, ecc.)

    Formato output:
    tool: <contenuto tool 1>
    tool: <contenuto tool 2>
    ai message: <risposta AI>
    """
    conversation = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            # Estrae il contenuto testuale (potrebbe avere multiple parti)
            main_text = ""
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        main_text = part.get('text', '')
                        break
            else:
                main_text = str(msg.content)

            # Preserva la formattazione originale (NON rimuovere newline!)
            cleaned_text = main_text.strip()
            if cleaned_text:
                conversation.append(f"ai message: {cleaned_text}")

        elif isinstance(msg, ToolMessage):
            # Preserva il contenuto del tool completo
            tool_content = str(msg.content).strip()
            if tool_content:
                conversation.append(f"tool: {tool_content}")

    return '\n'.join(conversation)


def extract_conversation_flow(messages: list) -> dict:
    """
    Estrae tutti i ToolMessage e AIMessage dalla conversazione.

    Formato output:
    {
        "tools": ["contenuto tool 1", "contenuto tool 2", ...],
        "ai_message": "risposta dell'AI"
    }
    """
    tools = []
    ai_message = ""

    for msg in messages:
        if isinstance(msg, AIMessage):
            # Estrae il contenuto testuale (potrebbe avere multiple parti)
            main_text = ""
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        main_text = part.get('text', '')
                        break
            else:
                main_text = str(msg.content)

            # Preserva la formattazione originale (NON rimuovere newline!)
            cleaned_text = main_text.strip()
            if cleaned_text:
                ai_message = cleaned_text

        elif isinstance(msg, ToolMessage):
            # Preserva il contenuto del tool completo
            tool_content = str(msg.content).strip()
            if tool_content:
                tools.append(tool_content)

    # F4/F5: links may point to hosts a tool returned; images never leave the answer
    link_hosts = set().union(*(url_hosts(t) for t in tools)) if tools else set()
    return {
        "tools": tools,
        "ai_message": neutralize_external_links(ai_message, link_hosts=link_hosts)
    }
