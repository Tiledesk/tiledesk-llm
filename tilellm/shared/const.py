import os

STREAM_NAME = "stream:single"
STREAM_CONSUMER_NAME = "llmconsumer"
STREAM_CONSUMER_GROUP = "llmconsumergroup"

PINECONE_API_KEY = None
PINECONE_INDEX = None
PINECONE_TEXT_KEY = None
VOYAGEAI_API_KEY = None

contextualize_q_system_prompt = """Given a chat history and the latest user question \
                        which might reference context in the chat history, formulate a standalone question \
                        which can be understood without the chat history. Do NOT answer the question, \
                        just reformulate it if needed and otherwise return it as is."""

qa_system_prompt = """You are an helpful assistant for question-answering tasks. \
                      Use ONLY the pieces of retrieved context delimited by #### to answer the question. \
                      The first step is to extrac relevant information to the question from retrieved context.
                      If you don't know the answer, just say that you don't know. \
                      Respond with "No relevant information were found <NOANS>" if no relevant information were found.
                      
                      
                        

                      ####
                      {context}
                      ####
                      """


def populate_constant():
    global PINECONE_API_KEY, PINECONE_INDEX, PINECONE_TEXT_KEY, VOYAGEAI_API_KEY
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
    PINECONE_TEXT_KEY = os.environ.get("PINECONE_TEXT_KEY")
    VOYAGEAI_API_KEY = os.environ.get("VOYAGEAI_API_KEY")




