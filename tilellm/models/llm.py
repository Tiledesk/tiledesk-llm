import datetime
from typing import Dict, Optional, List, Union, Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, RootModel, model_validator
from langchain_mcp_adapters.client import MultiServerMCPClient

from tilellm.models.base import AWSAuthentication, ServerConfig
from tilellm.models.embedding import LlmEmbeddingModel, EmbeddingModel
from tilellm.models.vector_store import Engine
from tilellm.models.chat import ChatEntry


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
    browser_headers: Dict[str, str] = Field(
        default_factory=lambda: {'user-agent': 'Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36'}
    )
    namespace: str | None = None
    webhook: str = Field(default_factory=lambda: "")
    semantic_chunk: Optional[bool] = Field(default=False)
    breakpoint_threshold_type: Optional[str] = Field(default="percentile")
    chunk_size: int = Field(default_factory=lambda: 1000)
    chunk_overlap: int = Field(default_factory=lambda: 400)
    parameters_scrape_type_4: Optional[Any] = None # Will be importing ParametersScrapeType4
    engine: Engine

    @model_validator(mode='after')
    def validate_browser_headers(self):
        if 'user-agent' not in self.browser_headers:
            self.browser_headers['user-agent'] = 'Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36'
        return self

    @model_validator(mode='after')
    def check_scrape_type(cls, values):
        # Questo import va qui per evitare dipendenze circolari
        from tilellm.models.scraping import ParametersScrapeType4
        scrape_type = values.scrape_type
        parameters_scrape_type_4 = values.parameters_scrape_type_4

        if scrape_type in (2, 4, 5):
            if parameters_scrape_type_4 is None:
                raise ValueError('parameters_scrape_type_4 must be provided when scrape_type is 2, 4 or 5')
            # Valida il dizionario in ParametersScrapeType4
            if isinstance(parameters_scrape_type_4, dict):
                values.parameters_scrape_type_4 = ParametersScrapeType4(**parameters_scrape_type_4)
        else:
            values.parameters_scrape_type_4 = None
        return values


class MetadataItem(BaseModel):
    id: str
    source: str | None = None
    type: str | None = None
    embedding: Union[str,LlmEmbeddingModel] = Field(default_factory=lambda: "text-embedding-ada-002")
    date: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"))
    namespace: Optional[str] = None


class QuestionAnswer(BaseModel):
    question: str
    namespace: str
    llm: Optional[str] = Field(default="openai")
    gptkey: SecretStr
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-3.5-turbo")
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
    reranking : Optional[bool] = Field(default_factory=lambda: False)
    reranking_multiplier: int = 3  # moltiplicatore per top_k
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
    def check_citations_max_tokens(cls, values):
        """Sets max_tokens to at least 1024 if citations=True."""
        if values.citations and values.max_tokens < 1024:
            values.max_tokens = 1024
        return values


class QuestionToLLM(BaseModel):
    question: str
    llm_key: Union[SecretStr, AWSAuthentication]
    llm: str
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-3.5-turbo")
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
        if v is not None and not v > 0: # Aggiungi la verifica per None
            raise ValueError("n_messages must be greater than 0")
        return v

    @field_validator("max_tokens")
    def max_tokens_range(cls, v):
        """Ensures max_tokens is a positive integer."""
        if isinstance(v, tuple): # Gestisce il caso in cui max_tokens è un tuple (default (128,))
            v = v[0]
        if not 50 <= v <= 132000:
            raise ValueError("max_tokens must be between 50 and 132000.")
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
    pass


class QuestionToAgent(BaseModel):
    question: str
    llm_key: Union[SecretStr, AWSAuthentication]
    llm: str
    model: str = Field(default="gpt-3.5-turbo")
    tools: Optional[List[Dict[str, ToolOptions]]] = Field(default_factory=list) # Default changed to list
    system_context: str = Field(default="You are a helpful AI bot. Always reply in the same language of the question.")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=128)
    chat_history_dict: Optional[Dict[str, ChatEntry]] = None
    n_messages: int = Field(default_factory=lambda: None)

    @field_validator("n_messages")
    def n_messages_range(cls, v):
        """Ensures n_messages is within greater than 0"""
        if v is not None and not v > 0: # Aggiungi la verifica per None
            raise ValueError("n_messages must be greater than 0")
        return v