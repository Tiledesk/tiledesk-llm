import datetime
from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, RootModel, model_validator, computed_field
from langchain_mcp_adapters.client import MultiServerMCPClient

from tilellm.models.base import AWSAuthentication, ServerConfig
from tilellm.models.embedding import LlmEmbeddingModel, EmbeddingModel
from tilellm.models.schemas.multimodal_content import MultimodalContent
from tilellm.models.schemas.general_schemas import ReasoningConfig
from tilellm.models.vector_store import Engine

if TYPE_CHECKING:
    from tilellm.models.chat import ChatEntry


class TEIConfig(BaseModel):
    provider: Literal["tei"] = "tei"
    name: str
    api_key: Optional[SecretStr] | None = None
    url: Optional[str] = Field(default_factory=lambda: "")
    custom_headers: Optional[Dict[str, Any]] = None


class PineconeRerankerConfig(BaseModel):
    provider: Literal["pinecone"] = "pinecone"
    api_key: SecretStr
    name: str = Field(default="bge-reranker-v2-m3")
    top_n: Optional[int] = Field(default=None)
    rank_fields: Optional[List[str]] = Field(default_factory=lambda: ["chunk_text"])
    parameters: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"truncate": "END"})


class ItemSingle(BaseModel):
    id: str
    source: str | None = None
    type: str | None = None
    content: str | None = None
    hybrid: Optional[bool] = Field(default=False)
    hybrid_batch_size: Optional[int] = Field(default=10)
    doc_batch_size: Optional[int] = Field(default=100)
    sparse_encoder: Union[str, TEIConfig, None] = Field(default="splade") # spade|bge-m3 or TEIConfig
    gptkey: SecretStr | None = None
    scrape_type: int = Field(default_factory=lambda: 0)
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-ada-002")
    browser_headers: Dict[str, str] = Field(
        default_factory=lambda: {'user-agent': 'Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36'}
    )
    namespace: str | None = None
    tags: Optional[List[str]] = None
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
    def check_scrape_type(self):
        # Questo import va qui per evitare dipendenze circolari
        from tilellm.models.scraping import ParametersScrapeType4

        if self.scrape_type in (2, 4, 5):
            if self.parameters_scrape_type_4 is None:
                raise ValueError('parameters_scrape_type_4 must be provided when scrape_type is 2, 4 or 5')
            # Valida il dizionario in ParametersScrapeType4
            if isinstance(self.parameters_scrape_type_4, dict):
                self.parameters_scrape_type_4 = ParametersScrapeType4(**self.parameters_scrape_type_4)
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
    tags: Optional[list[str]] = None

class QuestionAnswer(BaseModel):
    question: str
    namespace: str
    tags: Optional[Union[str, List[str]]] = None
    llm: Optional[str] = Field(default="openai")
    gptkey: Optional[SecretStr] = "sk"
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-3.5-turbo")
    sparse_encoder: Union[str, "TEIConfig", None] = Field(default="splade") #bge-m3
    temperature: float = Field(default=0.0)
    top_k: int = Field(default=5)
    max_tokens: int = Field(default=512)
    top_p: Optional[float] = Field(default=1.0)
    stream: Optional[bool] = Field(default_factory=lambda: False)
    embedding: Union[str,LlmEmbeddingModel] = Field(default_factory=lambda: "text-embedding-ada-002")
    similarity_threshold: float = Field(default_factory=lambda: 1.0)
    debug: bool = Field(default_factory=lambda: False)
    citations: bool = Field(default_factory=lambda: False)
    alpha: Optional[float] = Field(default=0.5)
    system_context: Optional[str] = None
    search_type: str = Field(default_factory=lambda: "similarity")
    chunks_only: Optional[bool] = Field(default_factory=lambda: False)
    reranking : Union[bool, "TEIConfig", "PineconeRerankerConfig"] = Field(default_factory=lambda: False)
    reranking_multiplier: int = 3  # moltiplicatore per top_k
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    contextualize_prompt: Optional[bool] = Field(default=False, description="Enable/disable contextualize_q_system_prompt usage")
    engine: Engine
    chat_history_dict: Optional[Dict[str, "ChatEntry"]] = None

    #@field_validator("temperature")
    #def temperature_range(cls, v):
    #    """Ensures temperature is within valid range (usually 0.0 to 1.0)."""
    #    if not 0.0 <= v <= 1.0:
    #        raise ValueError("Temperature must be between 0.0 and 1.0.")
    #    return v

    @model_validator(mode="after")
    def adjust_temperature_and_validate(self):
        # Ricava il nome del modello come stringa
        model_name: Optional[str] = None
        if isinstance(self.model, str):
            model_name = self.model
        elif isinstance(self.model, LlmEmbeddingModel):
            model_name = self.model.name

        # Se è gpt-5 o gpt-5-*, forza temperature a 1.0
        if model_name and model_name.startswith("gpt-5"):
            self.temperature = 1.0
            self.top_p = None
            return self

        # Se è claude-4 o claude-sonnet-4-*, rimuovi top_p se presente
        if model_name and ("claude-4" in model_name or "claude-sonnet-4" in model_name):
            if self.temperature is not None and self.top_p is not None:
                self.top_p = None

        # Se entrambi sono None, imposta default temperature
        if self.temperature is None and self.top_p is None:
            self.temperature = 0.0

        # Se entrambi sono specificati, gestisci in base al provider
        elif self.temperature is not None and self.top_p is not None:
            # Provider che richiedono temperature (non può essere None)
            if self.llm in ["google"]:
                # Mantieni temperature, rimuovi top_p
                self.top_p = None
            # Provider che richiedono top_p
            elif self.llm in []:  # Aggiungi qui provider che richiedono top_p
                self.temperature = None
            # Provider che accettano entrambi
            elif self.llm in ["openai", "vllm", "groq", "deepseek", "mistralai", "ollama"]:
                # Mantieni entrambi
                pass
            # Anthropic: gestione speciale per claude-4
            elif self.llm in ["anthropic"]:
                # claude-4 non supporta entrambi, mantieni solo temperature
                self.top_p = None
            # Provider che non supportano top_p
            elif self.llm in ["cohere"]:
                self.top_p = None
            else:
                # Default: priorità a temperature
                self.top_p = None

        # Valida i range
        if self.temperature is not None and not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")

        if self.top_p is not None and not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0.")

        return self

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


    @model_validator(mode='after')
    def validate_reranking_consistency(self):
        """
        Assicura che se reranking è True, esista un modello definito.
        """
        if self.reranking is True and not self.reranker_model:
            raise ValueError("reranker_model must be specified if reranking is True")
        return self


    # 2. Campo Calcolato per la configurazione risolta
    @computed_field(return_type=Optional[Union[str, "TEIConfig", "PineconeRerankerConfig"]])
    @property
    def reranker_config(self) -> Optional[Union[str, "TEIConfig", "PineconeRerankerConfig"]]:
        """
        Restituisce la configurazione pronta per TileReranker.
        - Se False: None (nessun reranking)
        - Se True: La stringa del modello (reranker_model)
        - Se Config: L'oggetto configurazione stesso
        """
        if self.reranking is False:
            return None

        if self.reranking is True:
            return self.reranker_model

        # Se è già un oggetto TEIConfig o PineconeRerankerConfig
        return self.reranking


class QuestionToLLM(BaseModel):
    question: Union[str, List[MultimodalContent]]
    llm_key: Union[SecretStr, AWSAuthentication]
    llm: str
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=128)
    top_p: Optional[float] = Field(default=1.0)
    stream: Optional[bool] = Field(default_factory=lambda: False)
    debug: bool = Field(default_factory=lambda: False)
    thinking: Optional[ReasoningConfig] = Field(
        default=None,
        description="Reasoning configuration for advanced models (GPT-5, Claude 4/4.5, Gemini 2.5/3.0, DeepSeek)"
    )
    system_context: str = Field(default="You are a helpful AI bot. Always reply in the same language of the question.")
    chat_history_dict: Optional[Dict[str, "ChatEntry"]] = None
    n_messages: int = Field(default_factory=lambda: None)
    structured_output: Optional[bool] = Field(default=False)
    output_schema: Optional[Any] = Field(default=None)
    servers: Optional[Dict[str, ServerConfig]] = Field(default_factory=dict)
    tools: Optional[List[str]] = Field(default=None, description="List of internal tool names from tool_registry")

    # Modalità di gestione history
    contextualize_prompt: Optional[bool] = Field(
        default=False,description="If True, injects the history as text into the system prompt. "
                                  "If False, passes the history as structured messages (recommended for modern LLMs)."
    )

    # Limitazione history
    max_history_messages: Optional[int] = Field(
        default=None,
        description="Maximum number of turns (question/answer pairs) to keep. None = unlimited. "
                    "E.g.: 10 = last 10 turns (20 messages)."
    )

    # Summarization
    summarize_old_history: bool = Field(
        default=False,
        description="If True and max_history_messages is set, automatically summarizes the oldest history "
                    "instead of discarding it. Requires an extra LLM call."
    )

    @model_validator(mode="after")
    def adjust_temperature_and_validate(self):
        # Ricava il nome del modello come stringa
        model_name: Optional[str] = None
        if isinstance(self.model, str):
            model_name = self.model
        elif isinstance(self.model, LlmEmbeddingModel):
            model_name = self.model.name

        # Se è gpt-5 o gpt-5-*, forza temperature a 1.0
        if model_name and model_name.startswith("gpt-5"):
            self.temperature = 1.0
            self.top_p = None
            return self

        # Se è claude-4 o claude-sonnet-4-*, rimuovi top_p se presente
        if model_name and ("claude-4" in model_name or "claude-sonnet-4" in model_name):
            if self.temperature is not None and self.top_p is not None:
                self.top_p = None

        # Se entrambi sono None, imposta default temperature
        if self.temperature is None and self.top_p is None:
            self.temperature = 0.0

        # Se entrambi sono specificati, gestisci in base al provider
        elif self.temperature is not None and self.top_p is not None:
            # Provider che richiedono temperature (non può essere None)
            if self.llm in ["google"]:
                # Mantieni temperature, rimuovi top_p
                self.top_p = None
            # Provider che richiedono top_p
            elif self.llm in []:  # Aggiungi qui provider che richiedono top_p
                self.temperature = None
            # Provider che accettano entrambi
            elif self.llm in ["openai", "vllm", "groq", "deepseek", "mistralai", "ollama"]:
                # Mantieni entrambi
                pass
            # Anthropic: gestione speciale per claude-4
            elif self.llm in ["anthropic"]:
                # claude-4 non supporta entrambi, mantieni solo temperature
                self.top_p = None
            # Provider che non supportano top_p
            elif self.llm in ["cohere"]:
                self.top_p = None
            else:
                # Default: priorità a temperature
                self.top_p = None

        # Valida i range
        if self.temperature is not None and not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")

        if self.top_p is not None and not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0.")

        return self

    @field_validator("n_messages")
    def n_messages_range(cls, v):
        """Ensures n_messages is within greater than 0"""
        if v is not None and not v > 0: # Aggiungi la verifica per None
            raise ValueError("n_messages must be greater than 0")
        return v

    @field_validator("max_tokens")
    def max_tokens_range(cls, v):
        """Ensures max_tokens is a positive integer."""

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
            name: server_config.model_dump(exclude_unset=True, exclude={"enabled_tools"})
            for name, server_config in self.servers.items()
        }
        return MultiServerMCPClient(config_dict)


    def get_question_content(self) -> Union[str, List[Dict[str, Any]]]:
        """
        Prepare the question content in the correct format for LangChain.
            - If the question is a string, it returns it.
            - If it is a multimodal list, it formats it as required.
        """
        if isinstance(self.question, str):
            # Caso semplice: solo testo
            return self.question

        if isinstance(self.question, list):
            # Caso multimodale: costruisce la lista di dizionari
            formatted_content = []
            for item in self.question:
                # Grazie al polimorfismo, ogni oggetto (TextContent, ImageContent)
                # sa come formattarsi correttamente.
                formatted_content.append(item.to_langchain_format())
            return formatted_content

        # Fallback nel caso di un tipo non previsto
        raise TypeError("The 'question' type is not supported.")


class ToolOptions(RootModel[Dict[str, Any]]):
    pass


# Risolvi forward references dopo che ChatEntry è caricato
def rebuild_llm_models():
    from tilellm.models.chat import ChatEntry
    QuestionAnswer.model_rebuild()
    QuestionToLLM.model_rebuild()
