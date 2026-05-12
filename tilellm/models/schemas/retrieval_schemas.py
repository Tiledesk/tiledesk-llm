from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tilellm.models.chat import ChatEntry


class _LLMCitation(BaseModel):
    """Citation model used exclusively as LLM structured-output target.

    Intentionally omits source_file_name so the LLM cannot hallucinate it.
    The field is populated server-side from document metadata after retrieval.
    """
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    source_name: str = Field(
        ...,
        description="The Article Source as URL (if available) of a SPECIFIC source which justifies the answer.",
    )


class Citation(BaseModel):
    """Citation returned to API consumers — includes server-enriched source_file_name."""
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    source_name: str = Field(
        ...,
        description="The Article Source as URL (if available) of a SPECIFIC source which justifies the answer.",
    )
    source_file_name: Optional[str] = Field(
        default=None,
        description="Human-readable file/page name from document metadata (e.g. 'price-list.pdf', 'Home – Acme Corp'). Use as link label in UX.",
    )

    @classmethod
    def from_llm(cls, llm_cit: "_LLMCitation") -> "Citation":
        return cls(source_id=llm_cit.source_id, source_name=llm_cit.source_name)


class _QuotedAnswer(BaseModel):
    """Structured-output schema sent to the LLM — uses _LLMCitation (no source_file_name)."""
    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[_LLMCitation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


class QuotedAnswer(BaseModel):
    """Answer with citations as returned by the API."""

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
    answer: Union[str, Dict[str, Any], list, "QuotedAnswer"] = Field(default="No answer")
    success: bool = Field(default=False)
    namespace: str
    id: str | None = None
    ids: Optional[List[str]] | None = None
    source: str | None = None
    sources: Optional[List[Union[str, Dict[str,str]]]] | None = None
    citations: Optional[List[Citation]] | None = None
    content_chunks: Optional[List[str]] | None = None
    prompt_token_size: int = Field(default=0)
    error_message: Optional[str] | None = None
    duration: Optional[float]= Field(default=0)
    chat_history_dict: Optional[Dict[str, "ChatEntry"]] = None
    status: Optional[str] = Field(default=None)
    cache_level: Optional[str] = Field(default=None, description="Cache hit level: 'exact' or 'semantic'")
    cache_similarity: Optional[float] = Field(default=None, description="Similarity score for semantic cache hits")

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
