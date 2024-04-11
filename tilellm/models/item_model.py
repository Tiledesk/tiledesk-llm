from pydantic import BaseModel, Field,  validator, field_validator, ValidationError
from typing import Dict, Optional, List


class ItemSingle(BaseModel):
    id: str
    source: str | None = None
    type: str | None = None
    content: str | None =None
    gptkey: str | None = None
    embedding: str = Field(default_factory=lambda: "text-embedding-ada-002")
    namespace: str | None =None
    webhook: str = Field(default_factory=lambda: "")

class MetadataItem(BaseModel):
    id: str
    source: str | None = None
    type: str | None = None
    embedding: str = Field(default_factory=lambda: "text-embedding-ada-002")

class ChatEntry(BaseModel):
    question: str
    answer: str
    #metadata: Optional[Dict[str, str]] = None  # Optional field for additional data


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
    model: str =Field(default="gpt-3.5-turbo") 
    temperature: float = Field(default=0.0)
    top_k: int = Field(default=5)
    max_tokens: int = Field(default=128)
    embedding: str = Field(default_factory=lambda: "text-embedding-ada-002")
    system_context: Optional[str] = None
    chat_history_dict : Optional[Dict[str, ChatEntry]] = None

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

class RetrievalResult(BaseModel):
    answer:str = Field(default="No answer")
    sources: Optional[List[str]]|None =None
    source:str |None= None
    id:str |None= None
    namespace: str
    ids: Optional[List[str]]|None =None
    prompt_token_size: int = Field(default=0)
    success: bool = Field(default=False)
    error_message: Optional[str]|None =None
    chat_history_dict:Optional[Dict[str, ChatEntry]]


class PineconeQueryResult(BaseModel):
    id : str
    metadata_id : str
    metadata_source : str
    metadata_type : str
    text : str

class PineconeItems(BaseModel):
    matches: List[PineconeQueryResult]

class PineconeItemToDelete(BaseModel):
    id : str
    namespace : str


