from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tilellm.models.chat import ChatEntry


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

class RetrievalResult(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
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
    chat_history_dict: Optional[Dict[str, "ChatEntry"]] = None

class RetrievalChunksResult(BaseModel):
    success: bool = Field(default=False)
    namespace: str
    chunks: Optional[List[str]] | None = None
    metadata: Optional[List[dict]] | None = None
    error_message: Optional[str] | None = None
    duration: Optional[float]= Field(default=0)

# Risolvi forward references dopo che ChatEntry è caricato
def rebuild_retrieval_models():
    from tilellm.models.chat import ChatEntry
    RetrievalResult.model_rebuild()
