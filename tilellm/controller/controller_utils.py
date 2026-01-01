from datetime import datetime
import asyncio


from fastapi.responses import StreamingResponse
import traceback
import uuid
from typing import List, Optional, Dict

import fastapi

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.tools import BaseTool

from starlette.responses import JSONResponse
from tilellm.models.schemas import (RetrievalResult,
                                    QuotedAnswer,
                                    Citation)
from tilellm.models import ChatEntry, ServerConfig

from tilellm.shared.sparse_util import HybridRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_classic.retrievers import ContextualCompressionRetriever

import tilellm.shared.const as const

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

# Function to preprocess chat history
def preprocess_chat_history(question_answer):
    logger.debug(question_answer.chat_history_dict)
    question_answer_list = []
    chat_history_list = []
    if question_answer.chat_history_dict is not None:
        for key, entry in question_answer.chat_history_dict.items():
            chat_history_list.append(HumanMessage(content=entry.question))
            chat_history_list.append(AIMessage(content=entry.answer))
            question_answer_list.append((entry.question, entry.answer))
    return chat_history_list, question_answer_list


# Function to convert chat history to text string
def chat_history_to_text(question_answer):
    """
    Converte la chat history in una stringa formattata.
    Utile per iniettare la history direttamente nel system prompt come variabile.
    """
    if not question_answer.chat_history_dict:
        return "No previous conversation."

    # Gestisci anche il caso di dict vuoto
    if len(question_answer.chat_history_dict) == 0:
        return "No previous conversation."

    history_lines = []
    try:
        for key, entry in question_answer.chat_history_dict.items():
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

    if question_answer.engine.name == "qdrant":
        search_kwargs = {'k': retrieval_k}
    else:
        search_kwargs = {'k': retrieval_k, 'namespace': question_answer.namespace}

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
    if sparse_encoder is None:
        sparse_vector = None
    else:
        sparse_vector = sparse_encoder.encode_queries(question_answer.question)
    dense_vector = await llm_embeddings.aembed_query(question_answer.question)
    return dense_vector, sparse_vector

async def fetch_question_vectors(question_answer, sparse_encoder, llm_embeddings):
     """Generate dense and sparse vectors in parallel"""
     tasks = [llm_embeddings.aembed_query(question_answer.question)]
     # Dense vector task

     # Sparse vector task (wrapped in async)
     if sparse_encoder:
         async def get_sparse():
             return sparse_encoder.encode_queries(question_answer.question)
         tasks.append(get_sparse())
     else:
         tasks.append(asyncio.sleep(0, result=None))

     dense_vector, sparse_vector = await asyncio.gather(*tasks)
     return dense_vector, sparse_vector

# Function to perform hybrid search
#async def perform_hybrid_search(question_answer, index, dense_vector, sparse_vector):
#    dense, sparse = hybrid_score_norm(dense_vector, sparse_vector, alpha=question_answer.alpha)
#    results = await index.query(
#        top_k=question_answer.top_k,
#        vector=dense,
#        sparse_vector=sparse,
#        namespace=question_answer.namespace,
#        include_metadata=True
#    )
#    return results


# Function to retrieve documents based on search results
def retrieve_documents_old(question_answer, results):
    documents = [Document(page_content=match["metadata"]["text"], metadata=match["metadata"]) for match in results["matches"]]
    retriever = HybridRetriever(documents=documents, k=question_answer.top_k)
    return retriever

# Function to retrieve documents based on search results
def retrieve_documents(question_answer, results, contextualized_query=None):
    #Aggiunto
    retrieval_k = question_answer.top_k * question_answer.reranking_multiplier if question_answer.reranking else question_answer.top_k

    # Get all available result or retrieval_k
    matches_to_use = results["matches"][:retrieval_k] if len(results["matches"]) >= retrieval_k else results["matches"]

    documents = [Document(page_content=match["metadata"]["text"], metadata=match["metadata"]) for match in matches_to_use]
    # Applica re-ranking se necessario
    if question_answer.reranking and len(documents) > question_answer.top_k:
        # Per l'hybrid search, dobbiamo applicare il re-ranking manualmente
        ranking_query = contextualized_query if contextualized_query else question_answer.question
        
        # Determine model config
        if isinstance(question_answer.reranking, bool): # True
             model_config = question_answer.reranker_model
        else:
             model_config = question_answer.reranking

        reranker = TileReranker(model_name=model_config)
        reranked_docs = reranker.rerank_documents(ranking_query, documents, question_answer.top_k)
        retriever = HybridRetriever(documents=reranked_docs, k=question_answer.top_k)
    else:
        retriever = HybridRetriever(documents=documents, k=question_answer.top_k)
    return retriever



# Function to create chains for contextualization and Q&A
async def create_chains(llm, question_answer, retriever):
    # Contextualize question

    # --- INIZIO MODIFICA: gestione condizionale di contextualize_prompt ---
    if question_answer.contextualize_prompt:
        # CODICE ORIGINALE: usa contextualize_q_system_prompt
        # Fa una chiamata LLM per contestualizzare la query per il retrieval
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
        # NUOVO CODICE: NON contestualizza la query, usa direttamente il retriever
        # Nessuna chiamata LLM extra per il retrieval → massimo risparmio token
        # La query originale viene usata così com'è per cercare nel repository
        history_aware_retriever = retriever
    # --- FINE MODIFICA ---

    # --- INIZIO MODIFICA: scegli il prompt corretto in base a contextualize_prompt ---
    if question_answer.contextualize_prompt:
        # PROMPT CON MessagesPlaceholder (history come messaggi)
        qa_system_prompt = question_answer.system_context if question_answer.system_context else const.qa_system_prompt

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    else:
        # PROMPT SENZA MessagesPlaceholder
        # History verrà iniettata direttamente (o nel prompt o nel system context)
        if question_answer.system_context:
            # Se c'è un system_context custom, NON usiamo {chat_history_text} come variabile
            # La history verrà appendata direttamente al system_context in generate_answer_with_history()
            qa_system_prompt = question_answer.system_context
        else:
            # Se NON c'è system_context custom, usa il prompt con {chat_history_text}
            qa_system_prompt = const.qa_system_prompt_with_history_injected

        # Prompt SENZA MessagesPlaceholder
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                ("human", "{input}"),
            ]
        )
    # --- FINE MODIFICA ---

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return history_aware_retriever, question_answer_chain, qa_prompt


async def create_contextualize_query(llm, question_answer):
    # --- INIZIO MODIFICA: gestione condizionale di contextualize_prompt ---
    if not question_answer.contextualize_prompt:
        # NUOVO: NON contestualizza la query, restituisce quella originale
        # Massimo risparmio token per hybrid search
        logger.info(f"Contextualize disabled - using original query: {question_answer.question}")
        return question_answer.question

    # CODICE ORIGINALE: contestualizza la query usando l'LLM
    contextualize_q_system_prompt = const.contextualize_q_system_prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    if question_answer.chat_history_dict:
        c_h, _ = preprocess_chat_history(question_answer)
        # Se c'è history, contestualizza la query
        contextualized_query =  (await llm.ainvoke(
            contextualize_q_prompt.format_messages(
                chat_history=c_h,
                input=question_answer.question
            )
        )).content
    else:
        # Se non c'è history, usa la query originale
        contextualized_query = question_answer.question

    logger.info(f"Contextualized query (FULL): {contextualized_query}")
    # --- FINE MODIFICA ---

    return contextualized_query


# Function to get or create session history
def get_or_create_session_history(store, session_id, chat_history_dict):
    if session_id not in store:
        store[session_id] = load_session_history(chat_history_dict)
    return store[session_id]


# Function to generate answer with chat history consideration
async def generate_answer_with_history(llm, question_answer, rag_chain, retriever, get_session_history, qa_prompt, callback_handler, question_answer_list):

    def extract_content(input_dict):
        # print(input_dict["answer"])
        return {"input": input_dict.get("input"), "answer": input_dict.get("answer")}

    # --- INIZIO MODIFICA: gestione history come stringa quando contextualize_prompt=False ---
    if not question_answer.contextualize_prompt:
        # NUOVO FLUSSO: history iniettata come stringa nel system prompt
        # NON usa RunnableWithMessageHistory, invoca direttamente rag_chain
        # Questo risparmia token perché non c'è overhead di MessagesPlaceholder

        # Converti la history in stringa
        chat_history_text = chat_history_to_text(question_answer)

        # Determina il system prompt da usare
        if question_answer.system_context:
            # CASO 1: system_context custom -> appende la history direttamente
            logger.info(f"Using custom system_context with appended history (length: {len(chat_history_text)} chars)")

            # Verifica che system_context contenga {context} (necessario per RAG)
            if "{context}" not in question_answer.system_context:
                logger.warning("Custom system_context does not contain {context} placeholder - adding it automatically")
                system_with_context = f"""{question_answer.system_context}
                                Retrieved context:
                                                {{context}}
                                """
            else:
                system_with_context = question_answer.system_context

            # Appende la history al system prompt
            system_final = f"""{system_with_context}
            
            <previous_conversation>
            {chat_history_text}
            </previous_conversation>
            """
        else:
            # CASO 2: prompt default già contiene {chat_history_text}
            logger.info(f"Using default prompt with injected history variable (length: {len(chat_history_text)} chars)")
            system_final = const.qa_system_prompt_with_history_injected

        qa_prompt_final = ChatPromptTemplate.from_messages([
            ("system", system_final),
            ("human", "{input}"),
        ])

        # Se non c'è system_context custom, la chain si aspetta 'chat_history_text'
        if question_answer.system_context:
            chain_input = {"input": question_answer.question}
        else:
            chain_input = {"input": question_answer.question, "chat_history_text": chat_history_text}


        if question_answer.stream:

            if question_answer.citations:
                rag_chain_from_docs_stream = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
                    | qa_prompt_final
                    | llm
                )
                retrieve_docs = (lambda x: x["input"]) | retriever
                runnable = (
                    RunnablePassthrough.assign(context=retrieve_docs)
                    .assign(answer=rag_chain_from_docs_stream)
                )
            else:
                runnable = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, qa_prompt_final))

            async def get_stream_llm_simple():
                full_response = ""
                message_id = str(uuid.uuid4())
                start_time = datetime.now()
                result_context = None

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
                            content_to_stream = chunk["answer"].content

                    if content_to_stream:
                        full_response += content_to_stream
                        yield _create_event("chunk", {"content": content_to_stream, "message_id": message_id})
                        await asyncio.sleep(0.02)

                    if "context" in chunk and result_context is None: # Capture context once
                        result_context = chunk["context"]

                # Final processing after stream ends
                citations = extract_citations(full_response) if question_answer.citations else []
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

                yield _create_event("metadata", {
                    "message_id": message_id,
                    "status": "completed",
                    "timestamp": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "full_response": full_response,
                    "model_used": result_to_return.model_dump()
                })

            return StreamingResponse(
                get_stream_llm_simple(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else: # NOT STREAMING
            start_time = datetime.now() if question_answer.debug else 0
            if question_answer.citations:
                retrieve_docs = (lambda x: x["input"]) | retriever
                rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
                    | qa_prompt_final
                    | llm.with_structured_output(QuotedAnswer)
                )
                chain_w_citations = (
                    RunnablePassthrough.assign(context=retrieve_docs)
                    .assign(answer=rag_chain_from_docs)
                    .assign(only_answer=lambda text: text["answer"].answer)
                )
                result = await chain_w_citations.ainvoke(chain_input)
                citations = result['answer'].citations
                result['answer'] = result['answer'].answer
            else:
                rag_chain_simple = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, qa_prompt_final))
                result = await rag_chain_simple.ainvoke(chain_input)
                citations = None

            end_time = datetime.now() if question_answer.debug else 0
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
            return JSONResponse(content=result_to_return.model_dump())
    # --- FINE MODIFICA ---

    # CODICE ORIGINALE: usa RunnableWithMessageHistory (quando contextualize_prompt=True)
    # citation_rag_chain = None
    if question_answer.stream:
        if question_answer.citations:
            retrieve_docs = (lambda x: x["input"]) | retriever

            rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
                    | qa_prompt
                    | llm
            )

            chain_w_citations = (
                RunnablePassthrough.assign(context=retrieve_docs)
                .assign(answer=rag_chain_from_docs)
                .assign(only_answer=RunnableLambda(extract_content)#lambda text: text["answer"].answer)
                )
            )

            conversational_rag_chain = RunnableWithMessageHistory(
                chain_w_citations,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        else:
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        return await get_streaming_response(conversational_rag_chain, question_answer, callback_handler, question_answer_list )
    else:

        if question_answer.citations:
            retrieve_docs = (lambda x: x["input"]) | retriever
            rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
                    | qa_prompt
                    | llm.with_structured_output(QuotedAnswer)
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
        result = await conversational_rag_chain.ainvoke(
            {"input": question_answer.question},
            config={"configurable": {"session_id": uuid.uuid4().hex}}
        )


        end_time = datetime.now() if question_answer.debug else 0
        duration = (end_time - start_time).total_seconds() if question_answer.debug else 0.0

        citations = result['answer'].citations if question_answer.citations else None
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

        return JSONResponse(content=result_to_return.model_dump())



# Function to format the result into the expected output structure
def format_result(result, citations, question_answer, callback_handler, question_answer_list, success, duration=0.0):
    docs = result["context"]
    ids, sources, content_chunks = extract_ids_sources(docs, question_answer.debug)
    source = format_sources(citations, sources, question_answer.citations)
    metadata_id = ids[0]

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
    if debug:
        content_chunks = []
        for doc in docs:
            ids.append(doc.metadata['id'])
            sources.append(doc.metadata['source'])
            content_chunks.append(doc.page_content)
    else:
        for doc in docs:
            ids.append(doc.metadata['id'])
            sources.append(doc.metadata['source'])
    ids = list(set(ids))
    sources = list(set(sources))
    return ids, sources, content_chunks


# Function to format sources based on citations
def format_sources(citations, sources, with_citations):
    if with_citations and citations is not None:
        source = " ".join(set([cit.source_name for cit in citations]))
    else:
        source = " ".join(sources)
    return source


# Function to handle exceptions
def handle_exception(e, question_answer):
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
    return s, success

async def get_streaming_response(runnable_with_history, question, callback_handler, question_answer_list):
    async def get_stream_llm(run_with_history, q):
        full_response = ""
        message_id = str(uuid.uuid4())
        start_time = datetime.now()
        result = dict()

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

            #no citations
            if "answer" in chunk and isinstance(chunk["answer"], str):
                full_response += chunk["answer"]
                yield _create_event("chunk", {"content": chunk["answer"], "message_id": message_id})
                await asyncio.sleep(0.02)  # Per un flusso più regolare
            # with citations
            elif 'answer' in chunk and hasattr(chunk['answer'], "content"):
                full_response += chunk['answer'].content
                yield _create_event("chunk", {"content": chunk['answer'].content, "message_id": message_id})
                await asyncio.sleep(0.02)  # Per un flusso più regolare
            elif 'context' in chunk:
                result["context"]=chunk["context"]
            elif 'input' in chunk:
                result['input'] = chunk['input']
                result['chat_history'] = chunk['chat_history']

        citations = extract_citations(full_response) if question.citations else []
        # q_answer = QuotedAnswer(answer = full_response, citations=citations)
        # print(f"======={q_answer}")
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

        yield _create_event("metadata", {
            "message_id": message_id,
            "status": "completed",
            "timestamp": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
            "full_response": full_response,
            "model_used": result_to_return.model_dump()
        })

    return StreamingResponse(
        get_stream_llm(runnable_with_history, question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


def extract_citations(testo: str) -> List[Citation]:
    import re

    # regex_citazione =r'(?i)cit:\s*id:\[?(\d+)\]?,\s*source:\[?([^;\]]+?)\]?;'
    # regex_citazione = r"Cit: id:(.+), source:(.+)"
    # regex_citazione = r"[Cc][Ii][Tt]:\s*id:\s*(?:\[)?(\d+)(?:\])?\s*(?:,\s*source:\s*(?:\[)?(.+))(?:\])?\s*"
    # regex_citazione = r"[Cc][Ii][Tt]:\s*id:\s*(?:\[)?(\d+)(?:\])?\s*(?:,\s*[Ss][Oo][Uu][Rr][Cc][Ee]:\s*(?:\[)?([^\]]+)?(?:\])?)?"
    # regex_citazione = r"[Cc][Ii][Tt]:\s*id:\s*(?:\[)?(\d+)(?:\])?\s*(?:,\s*[Ss][Oo][Uu][Rr][Cc][Ee]:\s*(?:\[)?([^\]]+?)(?:\])?)?"
    regex_citazione = r"[Cc][Ii][Tt]:\s*id:\s*(?:\[)?(\d+)(?:\])?\s*(?:,\s*[Ss][Oo][Uu][Rr][Cc][Ee]:\s*(?:\[)?(.*?)(?:\]?\s*(?:\(.*?\))?))?;"

    citations = []
    for match in re.finditer(regex_citazione, testo):
        # print(match)
        id_cit = int(match.group(1))
        source_cit = match.group(2).strip()

        citations.append(Citation(source_id=id_cit, source_name=source_cit))
    return citations

def _create_event(event_type: str, data: dict) -> str:
    import json
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"



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

    return {
        "tools": tools,
        "ai_message": ai_message
    }


async def get_filtered_tools_by_metadata(
    all_tools: List[BaseTool],
    servers_config: Dict[str, ServerConfig]
) -> List[BaseTool]:
    """
    Filtra i tool analizzando i metadati dell'oggetto BaseTool.
    """
    filtered_tools = []

    for tool in all_tools:
        # 1. Recuperiamo il nome del server dai metadati.
        # Nota: La chiave potrebbe essere "server_name" o "server" a seconda
        # di come il tuo MultiServerMCPClient popola i metadata.
        origin_server = tool.metadata.get("server_name") or tool.metadata.get("server")

        # Se il tool non ha informazioni sull'origine, o il server non è in config, lo saltiamo
        if not origin_server or origin_server not in servers_config:
            continue

        config = servers_config[origin_server]

        # 2. Verifichiamo se il tool è abilitato per quel server specifico
        # Qui usiamo tool.name perché è il nome originale registrato sul server MCP
        if "all" in config.enabled_tools or tool.name in config.enabled_tools:
            filtered_tools.append(tool)

    return filtered_tools

async def get_filtered_tools(
    all_tools: List[BaseTool],
    servers_config: Dict[str, ServerConfig]
) -> List[BaseTool]:
    """
    Filtra i tool di LangGraph basandosi esclusivamente sulla proprietà .name
    """
    # 1. Raccogliamo tutti i nomi dei tool abilitati esplicitamente
    enabled_names = set()
    has_global_all = False

    for config in servers_config.values():
        if "all" in config.enabled_tools:
            # Se un server ha "all", teoricamente vorremmo tutti i SUOI tool.
            # Senza metadata/prefissi, "all" diventa un comando "passa tutto".
            has_global_all = True
        else:
            enabled_names.update(config.enabled_tools)

    # 2. Logica di filtraggio
    if has_global_all:
        # Nota: Se hai "all" in un server e nomi specifici in un altro,
        # senza metadati l'unica opzione sicura è restituire tutto,
        # altrimenti rischieresti di tagliare fuori i tool del server "all".
        return all_tools

    # 3. Ritorna solo i tool il cui nome è nella whitelist
    return [t for t in all_tools if t.name in enabled_names]


async def get_all_filtered_tools(mcp_client, servers_config: Dict[str, ServerConfig]) -> List[BaseTool]:
    """
    Recupera e filtra i tool server per server prima di unirli.
    """
    final_tools = []

    for server_name, config in servers_config.items():
        try:
            # 1. Recuperiamo i tool solo per questo specifico server
            server_tools = await mcp_client.get_tools(server_name=server_name)
            print(server_tools)
            # 2. Applichiamo il filtro basato sulla config di QUESTO server
            if not config.enabled_tools:
                continue

            if "all" in config.enabled_tools:
                # Se è "all", prendiamo tutto quello che ha restituito questo server
                final_tools.extend(server_tools)
            else:
                # Altrimenti prendiamo solo i tool esplicitamente nominati
                filtered = [
                    t for t in server_tools
                    if t.name in config.enabled_tools
                ]
                final_tools.extend(filtered)

        except Exception as e:
            # Gestione errore se un server specifico non risponde o non esiste
            print(f"Errore nel recupero tool per il server {server_name}: {e}")
            continue

    return final_tools