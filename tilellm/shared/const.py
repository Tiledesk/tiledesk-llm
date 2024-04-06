import os

STREAM_NAME="stream:single"
STREAM_CONSUMER_NAME="llmconsumer"
STREAM_CONSUMER_GROUP="llmconsumergroup"

PINECONE_API_KEY = None
PINECONE_INDEX = None
PINECONE_TEXT_KEY = None

def populate_constant():
    global PINECONE_API_KEY, PINECONE_INDEX, PINECONE_TEXT_KEY
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
    PINECONE_TEXT_KEY = os.environ.get("PINECONE_TEXT_KEY")



