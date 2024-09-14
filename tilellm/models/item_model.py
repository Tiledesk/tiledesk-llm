from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator, RootModel, root_validator
from typing import Dict, Optional, List, Union, Any
import datetime




class ParametersScrapeType4(BaseModel):
    unwanted_tags: Optional[List[str]] = Field(default_factory=list)
    tags_to_extract: Optional[List[str]] = Field(default_factory=list)
    unwanted_classnames: Optional[List[str]] = Field(default_factory=list)
    desired_classnames: Optional[List[str]] = Field(default_factory=list)
    remove_lines: Optional[bool] = Field(default=True)
    remove_comments: Optional[bool] = Field(default=True)

    @model_validator(mode='after')
    def check_booleans(cls, values):
        remove_lines = values.remove_lines
        remove_comments = values.remove_comments
        if remove_lines is None or remove_comments is None:
            raise ValueError('remove_lines and remove_comments must be provided in ParametersScrapeType4')
        return values


class ItemSingle(BaseModel):
    id: str
    source: str | None = None
    type: str | None = None
    content: str | None = None
    gptkey: str | None = None
    scrape_type: int = Field(default_factory=lambda: 0)
    embedding: str = Field(default_factory=lambda: "text-embedding-ada-002")
    namespace: str | None = None
    webhook: str = Field(default_factory=lambda: "")
    chunk_size: int = Field(default_factory=lambda: 1000)
    chunk_overlap: int = Field(default_factory=lambda: 400)
    parameters_scrape_type_4: Optional[ParametersScrapeType4] = None

    @model_validator(mode='after')
    def check_scrape_type(cls, values):
        scrape_type = values.scrape_type
        parameters_scrape_type_4 = values.parameters_scrape_type_4

        if scrape_type == 4:
            if parameters_scrape_type_4 is None:
                raise ValueError('parameters_scrape_type_4 must be provided when scrape_type is 4')
        else:
            values.parameters_scrape_type_4 = None
        return values


class MetadataItem(BaseModel):
    id: str
    source: str | None = None
    type: str | None = None
    embedding: str = Field(default_factory=lambda: "text-embedding-ada-002")
    date: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"))


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
    gptkey: str
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.0)
    top_k: int = Field(default=5)
    max_tokens: int = Field(default=128)
    embedding: str = Field(default_factory=lambda: "text-embedding-ada-002")
    similarity_threshold: float = Field(default_factory=lambda: 1.0)
    debug: bool = Field(default_factory=lambda: False)
    citations: bool = Field(default_factory=lambda: True)
    system_context: Optional[str] = None
    search_type: str = Field(default_factory=lambda: "similarity")
    chat_history_dict: Optional[Dict[str, ChatEntry]] = None

    @field_validator("temperature")
    def temperature_range(cls, v):
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


class AWSAuthentication(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str


class QuestionToLLM(BaseModel):
    question: str
    llm_key: Union[str, AWSAuthentication]
    llm: str
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=128)
    debug: bool = Field(default_factory=lambda: False)
    system_context: str = Field(default="You are a helpful AI bot. Always reply in the same language of the question.")
    chat_history_dict: Optional[Dict[str, ChatEntry]] = None
    n_messages: int = Field(default_factory=lambda: None)

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
        if not 50 <= v <= 2000:
            raise ValueError("top_k must be a positive integer.")
        return v


class ToolOptions(RootModel[Dict[str, Any]]):
    #__root__: Dict[str, Any] = Field(default_factory=dict)
    pass


class QuestionToAgent(BaseModel):
    question: str
    llm_key: Union[str, AWSAuthentication]
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


class SimpleAnswer(BaseModel):
    answer: str = Field(default="No answer")
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
    chat_history_dict: Optional[Dict[str, ChatEntry]]


class PineconeQueryResult(BaseModel):
    id: str
    metadata_id: str
    metadata_source: str
    metadata_type: str
    date: Optional[str] = Field(default="Date not defined")
    text: Optional[str] | None = None


class PineconeItems(BaseModel):
    matches: List[PineconeQueryResult]


class PineconeNamespaceToDelete(BaseModel):
    namespace: str


class PineconeItemToDelete(BaseModel):
    id: str
    namespace: str


class ScrapeStatusReq(BaseModel):
    id: str
    namespace: str
    namespace_list: Optional[List[str]] | None = None


class ScrapeStatusResponse(BaseModel):
    status_message: str = Field(default="Crawling is not started")
    status_code: int = Field(default=0)
    queue_order: int = Field(default=-1)


class PineconeIndexingResult(BaseModel):
    #  {"id": f"{id}", "chunks": f"{len(chunks)}", "total_tokens": f"{total_tokens}", "cost": f"{cost:.6f}"}
    id: str | None = None
    chunks: int | None = None
    total_tokens: int | None = None
    cost: str | None = None
    status: int = Field(default=300)
    date: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"))
    error: Optional[str] | None = None


class PineconeItemNamespaceResult(BaseModel):
    namespace: str
    vector_count: int


class PineconeIdSummaryResult(BaseModel):
    metadata_id: str
    source: str
    chunks_count: int


class PineconeNamespaceResult(BaseModel):
    namespaces: Optional[List[PineconeItemNamespaceResult]]


class PineconeDescNamespaceResult(BaseModel):
    namespace_desc: PineconeItemNamespaceResult
    ids: Optional[List[PineconeIdSummaryResult]]
