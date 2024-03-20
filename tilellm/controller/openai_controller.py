from langchain.chains import ConversationalRetrievalChain # Per la conversazione va usata questa classe
from langchain_openai import ChatOpenAI
from tilellm.store.pinecone_repository import add_pc_item as pinecone_add_item
from tilellm.store.pinecone_repository import create_pc_index, get_embeddings_dimension
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from tilellm.models.item_model import RetrievalResult,ChatHistory, ChatEntry


def ask_with_memory(question_answer):
    ## FIXME mettere tutto sotto try e verificare che non ci siano errori, altrimenti mandare l'errore 
    try:
        print(question_answer)

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

        print(question_answer_list)
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
        
        #template = (
        #            "Combine the chat history and follow up question into "
        #            "a standalone question. Chat History: {chat_history}"
        #            "Follow up question: {question}"
        #        )
        #        prompt = PromptTemplate.from_template(template)
        #        llm = OpenAI()
        #        question_generator_chain = LLMChain(llm=llm, prompt=prompt)
        docs = retriever.get_relevant_documents(question_answer.question)
        #new_list = list(set(students))
        ids =[]
        sources = []
        for doc in docs:
            ids.append(doc.metadata['id'])
            sources.append(doc.metadata['source'])
        ids = list(set(ids))
        sources = list(set(sources))

        crc = ConversationalRetrievalChain.from_llm(llm, retriever )
        
        result = crc.invoke({'question': question_answer.question, 'chat_history': question_answer_list})
        print(result)
        question_answer_list.append((result['question'], result['answer']))
        
        chat_entries = [ChatEntry(question=q, answer=a) for q, a in question_answer_list]
        chat_history_dict = {str(i): entry for i, entry in enumerate(chat_entries)}

        
        #chat_history.from_dict(data=question_answer_list)
        #question_answer_list.append((question_answer.question, result['answer']))
        #result = {answer: result['answer'],chat_history:question_answer_list}
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
