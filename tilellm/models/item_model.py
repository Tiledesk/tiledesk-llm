from huggingface_hub import snapshot_download
from langchain_core.documents import Document
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator, RootModel, SecretStr
from typing import Dict, Optional, List, Union, Any, Literal
import datetime
from enum import Enum

#from pydantic.v1 import validator


class EmbeddingProviders(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GOOGLE = "google"
    COHERE = "cohere"
    VOYAGE = "voyage"
    VLLM = "vllm"

EMBEDDING_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "normalize": True,
        "device": "auto"
    },
    "BAAI/bge-m3": {
        "dimension": 1024,
        "normalize": False,
        "device": "cuda"
    },
    "voyage-multilingual-2": {
        "dimension": 1024,
        "voyage_api_key": "your-default-key"
    }
}

def prepare_huggingface_model(model_name: str):
    """Scarica e cachea il modello Hugging Face"""
    return snapshot_download(
        repo_id=model_name,
        #local_dir=f"./models/{model_name.replace('/', '_')}"
    )

class LlmEmbeddingModel(BaseModel):
    provider: EmbeddingProviders
    name: str
    api_key: Optional[SecretStr] | None = None
    url: Optional[str] = Field(default_factory=lambda: "")
    dimension: Optional[int] = 1024 #qwel2-deepseek 3584, llama3.2 3072

    @field_validator('name')
    def validate_model_name(cls, v, values):
        if values.get('provider') == EmbeddingProviders.HUGGINGFACE:
            prepare_huggingface_model(v)  # Scarica il modello all'validazione
        return v

class EmbeddingModel(BaseModel):
    embedding_provider: str
    embedding_key: Optional[SecretStr]| None = None
    embedding_model: str
    embedding_host: Optional[str] = Field(default=None)
    embedding_dimension: Optional[int] = None


class Engine(BaseModel):
    name: str = Field(default="pinecone")
    type: Optional[str] = Field(default="serverless")
    apikey: Optional[SecretStr]| None = None
    vector_size: int = Field(default=1536)
    index_name: str = Field(default="tilellm")
    text_key: Optional[str] = Field(default="text")
    metric: str = Field(default="cosine")
    host: Optional[str] = Field(default="localhost")
    port: Optional[int] = Field(default=6333)
    deployment: Optional[Literal["local", "cloud"]] = Field(default="local")

    @model_validator(mode='after')
    def validate_fields(self):
        if self.name == "pinecone":
            if self.type is None:
                self.type = "serverless"
            if self.type not in ("serverless", "pod"):
                raise ValueError("Type must be 'serverless' or 'pod' for Pinecone")

        elif self.name == "qdrant":
            if not self.deployment:
                raise ValueError("Deployment is required for Qdrant (local/cloud)")

            if self.deployment == "local" or self.deployment == "cloud":
                if not (self.host and self.port):
                    raise ValueError("Host and port are required for local Qdrant")

        else:
            raise ValueError(f"Unsupported engine: {self.name}")

        return self

    @model_validator(mode='after')
    def set_text_key(self):
        if self.name == "pinecone":
            if self.type == "serverless":
                self.text_key = "text"
            elif self.type == "pod":
                self.text_key = "content"
        return self


class ParametersScrapeType4(BaseModel):
    unwanted_tags: Optional[List[str]] = Field(default_factory=list)
    tags_to_extract: Optional[List[str]] = Field(default_factory=list)
    unwanted_classnames: Optional[List[str]] = Field(default_factory=list)
    desired_classnames: Optional[List[str]] = Field(default_factory=list)
    remove_lines: Optional[bool] = Field(default=True)
    remove_comments: Optional[bool] = Field(default=True)
    time_sleep: Optional[float] = Field(default=2)

    @model_validator(mode='after')
    def check_booleans(self):
        if self.remove_lines is None or self.remove_comments is None:
            raise ValueError('remove_lines and remove_comments must be provided in ParametersScrapeType4')
        return self


class ItemSingle(BaseModel):
    id: str
    source: str | None = None
    type: str | None = None
    content: str | None = None
    hybrid: Optional[bool] = Field(default=False)
    hybrid_batch_size: Optional[int] = Field(default=10)
    sparse_encoder: Optional[str] = Field(default="splade") # spade|bge-m3
    gptkey: SecretStr | None = None
    scrape_type: int = Field(default_factory=lambda: 0)
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-ada-002")
    #model: Optional[LlmEmbeddingModel] | None = None
    namespace: str | None = None
    webhook: str = Field(default_factory=lambda: "")
    semantic_chunk: Optional[bool] = Field(default=False)
    breakpoint_threshold_type: Optional[str] = Field(default="percentile")
    chunk_size: int = Field(default_factory=lambda: 1000)
    chunk_overlap: int = Field(default_factory=lambda: 400)
    parameters_scrape_type_4: Optional[ParametersScrapeType4] = None
    engine: Engine

    @model_validator(mode='after')
    def check_scrape_type(self):
        if self.scrape_type in (2, 4, 5):
            if self.parameters_scrape_type_4 is None:
                raise ValueError('parameters_scrape_type_4 must be provided when scrape_type is 2, 4 or 5')
        else:
            self.parameters_scrape_type_4 = None
        return self


class MetadataItem(BaseModel):
    id: str
    source: str | None = None
    type: str | None = None
    embedding: Union[str,LlmEmbeddingModel] = Field(default_factory=lambda: "text-embedding-ada-002")
    date: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"))
    namespace: Optional[str] = None


class ChatEntry(BaseModel):
    question: str
    answer: str
    # metadata: Optional[Dict[str, str]] = None  # Optional field for additional data


class ChatHistory(BaseModel):
    chat_history: Dict[str, ChatEntry]

    @classmethod
    def from_dict(cls, data: dict) -> "ChatHistory":
        """Custom constructor to handle potential issues during initialization."""
        chat_history = {}
        for key, entry_data in data.items():
            try:
                if not isinstance(key, str):
                    raise ValueError(f"Invalid key type '{type(key)}'. Expected string.")
                chat_history[key] = ChatEntry(**entry_data)
            except (TypeError, ValueError) as e:
                raise ValidationError(f"Error processing entry '{key}': {str(e)}")
        return cls(chat_history=chat_history)
    

class QuestionAnswer(BaseModel):
    question: str
    namespace: str
    llm: Optional[str] = Field(default="openai")
    gptkey: SecretStr
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o")
    sparse_encoder: Optional[str] = Field(default="splade") #bge-m3
    temperature: float = Field(default=0.0)
    top_k: int = Field(default=5)
    max_tokens: int = Field(default=512)
    top_p: Optional[float] = Field(default=1.0)
    stream: Optional[bool] = Field(default_factory=lambda: False)
    embedding: Union[str,EmbeddingModel] = Field(default_factory=lambda: "text-embedding-ada-002")
    similarity_threshold: float = Field(default_factory=lambda: 1.0)
    debug: bool = Field(default_factory=lambda: False)
    citations: bool = Field(default_factory=lambda: False)
    alpha: Optional[float] = Field(default=0.5)
    system_context: Optional[str] = None
    search_type: str = Field(default_factory=lambda: "similarity")
    chunks_only: Optional[bool] = Field(default_factory=lambda: False)
    engine: Engine
    chat_history_dict: Optional[Dict[str, ChatEntry]] = None

    @field_validator("temperature")
    def temperature_range(cls, v):
        """Ensures temperature is within valid range (usually 0.0 to 1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        return v

    @field_validator("top_p")
    def top_p_range(cls, v):
        """Ensures temperature is within valid range (usually 0.0 to 1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        return v

    @field_validator("top_k")
    def top_k_range(cls, v):
        """Ensures top_k is a positive integer."""
        if v <= 0:
            raise ValueError("top_k must be a positive integer.")
        return v

    @model_validator(mode='after')
    def check_citations_max_tokens(self):
        """Sets max_tokens to at least 1024 if citations=True."""
        if self.citations and self.max_tokens < 1024:
            self.max_tokens = 1024
        return self


class AWSAuthentication(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str



class ServerConfig(BaseModel):
    """Modello per la configurazione di un server MCP"""
    transport: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    api_key: Optional[str] = None
    parameters: Optional[dict] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_transport_specific_fields(self):
        # Validazione per trasporto SSE
        if self.transport == "sse":
            if not self.url:
                raise ValueError("URL è obbligatorio per il trasporto SSE")

        # Validazione per trasporto stdio
        elif self.transport == "stdio":
            if not self.command or not self.args:
                raise ValueError("Command e args sono obbligatori per il trasporto stdio")

        return self


#thinking: Optional[Dict[str, Any]] = Field(default=None)
#    """Parameters for Claude reasoning,
#    e.g., ``{"type": "enabled", "budget_tokens": 10_000}

class QuestionToLLM(BaseModel):
    question: str
    llm_key: Union[SecretStr, AWSAuthentication]
    llm: str
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=128),
    top_p: Optional[float] = Field(default=1.0)
    stream: Optional[bool] = Field(default_factory=lambda: False)
    debug: bool = Field(default_factory=lambda: False)
    thinking: Optional[Dict[str, Any]] = Field(default=None)
    system_context: str = Field(default="You are a helpful AI bot. Always reply in the same language of the question.")
    chat_history_dict: Optional[Dict[str, ChatEntry]] = None
    n_messages: int = Field(default_factory=lambda: None)
    servers: Optional[Dict[str, ServerConfig]] = Field(default_factory=dict)

    @field_validator("temperature")
    def temperature_range(cls, v):
        """Ensures temperature is within valid range (usually 0.0 to 1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        return v

    @field_validator("n_messages")
    def n_messages_range(cls, v):
        """Ensures n_messages is within greater than 0"""
        if not v > 0:
            raise ValueError("n_messages must be greater than 0")
        return v

    @field_validator("max_tokens")
    def max_tokens_range(cls, v):
        """Ensures max_tokens is a positive integer."""
        if not 50 <= v <= 132000:
            raise ValueError("top_k must be a positive integer.")
        return v

    @field_validator("top_p")
    def top_p_range(cls, v):
        """Ensures temperature is within valid range (usually 0.0 to 1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        return v

    def create_mcp_client(self):
        """Crea un'istanza di MultiServerMCPClient dalla configurazione"""
        config_dict = {
            name: server_config.model_dump(exclude_unset=True)
            for name, server_config in self.servers.items()
        }
        return MultiServerMCPClient(config_dict)


class ToolOptions(RootModel[Dict[str, Any]]):
    #__root__: Dict[str, Any] = Field(default_factory=dict)
    pass


class QuestionToAgent(BaseModel):
    question: str
    llm_key: Union[SecretStr, AWSAuthentication]
    llm: str
    model: str = Field(default="gpt-3.5-turbo")
    tools: Optional[List[Dict[str, ToolOptions]]] = Field(default_factory=dict)
    system_context: str = Field(default="You are a helpful AI bot. Always reply in the same language of the question.")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=128)
    chat_history_dict: Optional[Dict[str, ChatEntry]] = None
    n_messages: int = Field(default_factory=lambda: None)

    @field_validator("n_messages")
    def n_messages_range(cls, v):
        """Ensures n_messages is within greater than 0"""
        if not v > 0:
            raise ValueError("n_messages must be greater than 0")
        return v

class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    source_name: str = Field(
        ...,
        description="The Article Source as URL (if available) of a SPECIFIC source which justifies the answer.",
    )
    #quote: str = Field(
    #    ...,
    #    description="The VERBATIM quote from the specified source that justifies the answer.",
    #)

class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


class PartialQuotedAnswer(BaseModel):
    delta: str = Field(
        ...,
        description="token for the answer to the user question, which is based only on the given sources.",
    ) # Singolo token
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )

class QuotedAnswerForStream(PartialQuotedAnswer):
    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    complete: bool = True

class SimpleAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    chat_history_dict: Optional[Dict[str, ChatEntry]]

class ReasoningAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    reasoning_content: Union[str, Dict[str, Any], list] = Field(default="No reasoningn answer")
    chat_history_dict: Optional[Dict[str, ChatEntry]]


class RetrievalResult(BaseModel):
    answer: str = Field(default="No answer")
    success: bool = Field(default=False)
    namespace: str
    id: str | None = None
    ids: Optional[List[str]] | None = None
    source: str | None = None
    sources: Optional[List[str]] | None = None
    citations: Optional[List[Citation]] | None = None
    content_chunks: Optional[List[str]] | None = None
    prompt_token_size: int = Field(default=0)
    error_message: Optional[str] | None = None
    duration: Optional[float]= Field(default=0)
    chat_history_dict: Optional[Dict[str, ChatEntry]]

class RetrievalChunksResult(BaseModel):
    success: bool = Field(default=False)
    namespace: str
    chunks: Optional[List[str]] | None = None
    metadata: Optional[List[dict]] | None = None
    error_message: Optional[str] | None = None
    duration: Optional[float]= Field(default=0)



class RepositoryQueryResult(BaseModel):
    id: str
    metadata_id: str
    metadata_source: str
    metadata_type: str
    date: Optional[str] = Field(default="Date not defined")
    text: Optional[str] | None = None


class RepositoryItems(BaseModel):
    matches: List[RepositoryQueryResult]

class RepositoryEngine(BaseModel):
    engine: Engine

class RepositoryNamespace(BaseModel):
    namespace: str
    engine: Engine


class RepositoryItem(BaseModel):
    id: str
    namespace: str
    engine: Engine


class ScrapeStatusReq(BaseModel):
    id: str
    namespace: str
    namespace_list: Optional[List[str]] | None = None
    engine: Engine


class ScrapeStatusResponse(BaseModel):
    status_message: str = Field(default="Crawling is not started")
    status_code: int = Field(default=0)
    queue_order: int = Field(default=-1)


class IndexingResult(BaseModel):
    #  {"id": f"{id}", "chunks": f"{len(chunks)}", "total_tokens": f"{total_tokens}", "cost": f"{cost:.6f}"}
    id: str | None = None
    chunks: int | None = None
    total_tokens: int | None = None
    cost: str | None = None
    status: int = Field(default=300)
    date: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"))
    error: Optional[str] | None = None


class RepositoryItemNamespaceResult(BaseModel):
    namespace: str
    vector_count: int


class RepositoryIdSummaryResult(BaseModel):
    metadata_id: str
    source: str
    chunks_count: int


class RepositoryNamespaceResult(BaseModel):
    namespaces: Optional[List[RepositoryItemNamespaceResult]]


class RepositoryDescNamespaceResult(BaseModel):
    namespace_desc: RepositoryItemNamespaceResult
    ids: Optional[List[RepositoryIdSummaryResult]]

class MyDocument(Document):
    sparse_values: Optional[dict]