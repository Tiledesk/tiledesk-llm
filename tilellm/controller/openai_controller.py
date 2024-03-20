from langchain.chains import ConversationalRetrievalChain, LLMChain # Per la conversazione va usata questa classe
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tilellm.store.pinecone_repository import add_pc_item as pinecone_add_item
from tilellm.store.pinecone_repository import create_pc_index, get_embeddings_dimension
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from tilellm.models.item_model import RetrievalResult,ChatHistory, ChatEntry

import logging

logger = logging.getLogger(__name__)

def ask_with_memory(question_answer):
    
    try:
        logger.info(question_answer)

        #question = str
        #namespace: str
        #gptkey: str
        #model: str =Field(default="gpt-3.5-turbo") 
        #temperature: float = Field(default=0.0)
        #top_k: int = Field(default=5)
        #max_tokens: int = Field(default=128)
        #system_context: Optional[str]
        #chat_history_dict : Dict[str, ChatEntry]
        
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
        
        emb_dimension = get_embeddings_dimension(question_answer.embedding)
        oai_embeddings = OpenAIEmbeddings(api_key=question_answer.gptkey, model=question_answer.embedding) 
        
        vector_store= create_pc_index(oai_embeddings, emb_dimension)

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': question_answer.top_k, 'namespace':question_answer.namespace})
        

        docs = retriever.get_relevant_documents(question_answer.question)

        ids =[]
        sources = []
        for doc in docs:
            ids.append(doc.metadata['id'])
            sources.append(doc.metadata['source'])
        ids = list(set(ids))
        sources = list(set(sources))

        if question_answer.system_context is not None and question_answer.system_context:
            template = (
                            f"{question_answer.system_context}."
                            "Chat History: {chat_history}"
                            "Follow up question: {question}"
                        )
            prompt = PromptTemplate.from_template(template)
            #question_generator_chain = LLMChain(llm=llm, prompt=prompt)
            crc = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                condense_question_prompt=prompt
                )
        else:
            crc = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
        
        result = crc.invoke({'question': question_answer.question, 'chat_history': question_answer_list})
        print(result)
        question_answer_list.append((result['question'], result['answer']))
        
        chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
        chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

        
        success= bool(openai_callback_handler.successful_requests)
        prompt_token_size=openai_callback_handler.total_tokens

        result_to_return = RetrievalResult(
            answer=result['answer'],
            sources=sources,
            namespace=question_answer.namespace,
            ids=ids,
            prompt_token_size=prompt_token_size,
            success=success,
            error_message = None, 
            chat_history_dict = chat_history_dict

        )

        
    except Exception as e:
        print(e)
        question_answer_list = []
        for key, entry in question_answer.chat_history_dict.items():
            question_answer_list.append((entry.question, entry.answer))
        chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
        chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

        result_to_return = RetrievalResult(
            namespace=question_answer.namespace,
            error_message = repr(e), 
            chat_history_dict = chat_history_dict

        )

    #print("Prompt tokens:", openai_callback_handler.prompt_tokens)
    #print("Completion tokens:", openai_callback_handler.total_cost)
    return result_to_return.dict()

def add_pc_item(item):
    return pinecone_add_item(item)

def delete_namespace(namespace:str):
    from tilellm.store.pinecone_repository import delete_pc_namespace
    try:
        return delete_pc_namespace(namespace)
    except:
        raise
