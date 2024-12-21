import traceback
import uuid
from typing import List

import fastapi

from langchain_core.documents import Document

from langchain_core.runnables import RunnablePassthrough

from tilellm.models.item_model import (RetrievalResult,
                                       ChatEntry,
                                       QuotedAnswer
                                       )

from tilellm.shared.sparse_util import hybrid_score_norm, HybridRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever

import tilellm.shared.const as const

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage

)

from tilellm.tools.sparse_encoders import TiledeskSparseEncoders

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


# Function to initialize embeddings and encoders
async def initialize_embeddings_and_index(question_answer, repo, llm_embeddings):
    emb_dimension = repo.get_embeddings_dimension(question_answer.embedding)
    sparse_encoder = TiledeskSparseEncoders(question_answer.sparse_encoder)
    vector_store = await repo.create_pc_index(question_answer.engine, llm_embeddings, emb_dimension)
    index = vector_store.get_pinecone_index(question_answer.engine.index_name, pinecone_api_key=question_answer.engine.apikey)
    return emb_dimension, sparse_encoder, index


# Function to initialize embeddings and retrievers
async def initialize_retrievers(question_answer, repo, llm_embeddings):
    emb_dimension = repo.get_embeddings_dimension(question_answer.embedding)
    vector_store = await repo.create_pc_index(question_answer.engine, llm_embeddings, emb_dimension)

    vs_retriever = vector_store.as_retriever(
        search_type=question_answer.search_type,
        search_kwargs={'k': question_answer.top_k, 'namespace': question_answer.namespace}
    )

    redundant_filter = EmbeddingsRedundantFilter(
        embeddings=llm_embeddings,
        similarity_threshold=question_answer.similarity_threshold
    )

    pipeline_compressor = DocumentCompressorPipeline(transformers=[redundant_filter])

    retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=vs_retriever
    )

    return retriever


# Function to fetch vectors for the given question
async def fetch_question_vectors(question_answer, sparse_encoder, llm_embeddings):
    sparse_vector = sparse_encoder.encode_queries(question_answer.question)
    dense_vector = await llm_embeddings.aembed_query(question_answer.question)
    return dense_vector, sparse_vector


# Function to perform hybrid search
def perform_hybrid_search(question_answer, index, dense_vector, sparse_vector):
    dense, sparse = hybrid_score_norm(dense_vector, sparse_vector, alpha=question_answer.alpha)
    results = index.query(
        top_k=question_answer.top_k,
        vector=dense,
        sparse_vector=sparse,
        namespace=question_answer.namespace,
        include_metadata=True
    )
    return results


# Function to retrieve documents based on search results
def retrieve_documents(question_answer, results):
    documents = [Document(page_content=match["metadata"]["text"], metadata=match["metadata"]) for match in results["matches"]]
    retriever = HybridRetriever(documents=documents, k=question_answer.top_k)
    return retriever


# Function to create chains for contextualization and Q&A
def create_chains(llm, question_answer, retriever):
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
    qa_system_prompt = question_answer.system_context if question_answer.system_context else const.qa_system_prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return history_aware_retriever, question_answer_chain, qa_prompt


# Function to get or create session history
def get_or_create_session_history(store, session_id, chat_history_dict):
    if session_id not in store:
        store[session_id] = load_session_history(chat_history_dict)
    return store[session_id]


# Function to generate answer with chat history consideration
async def generate_answer_with_history(llm, question_answer, rag_chain, retriever, get_session_history, qa_prompt):
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

    result = conversational_rag_chain.invoke(
        {"input": question_answer.question},
        config={"configurable": {"session_id": uuid.uuid4().hex}}
    )

    citations = result['answer'].citations if question_answer.citations else None
    result['answer'], success = verify_answer(result['answer'].answer
                                              if question_answer.citations else result['answer'])
    return result, citations, success


# Function to format the result into the expected output structure
def format_result(result, citations, question_answer, callback_handler, question_answer_list, success):
    docs = result["context"]
    ids, sources, content_chunks = extract_ids_sources(docs, question_answer.debug)
    source = format_sources(citations, sources, question_answer.citations)
    metadata_id = ids[0]

    prompt_token_size = callback_handler.total_tokens

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
    if with_citations:
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