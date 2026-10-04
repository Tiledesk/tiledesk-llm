"""
Request/response models for POST /api/v1/systemone.

Validated strictly at the API boundary: unknown fields, unknown question types, empty
or oversized option sets are rejected with 422 — never silently dropped (an unknown
field silently ignored is how /api/ingestion once lost 112 PDFs). Limits follow the
Jev wire contract: choice 1-255 options, score 2-10 ordered levels.
"""
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from tilellm.modules.system_one.providers import get_provider, list_providers


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulQuestion(_Strict):
    type: Literal["noul"]
    instructions: str = Field(min_length=1)
    criteria: Optional[Dict[Literal["true", "false"], str]] = Field(
        default=None, description="Optional meaning of true/false.")


class ChoiceQuestion(_Strict):
    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    criteria: Dict[str, str] = Field(min_length=1, max_length=255,
                                     description="option -> description")


class ScoreQuestion(_Strict):
    type: Literal["score"]
    instructions: str = Field(min_length=1)
    criteria: List[str] = Field(min_length=2, max_length=10,
                                description="Ordered levels, low to high.")


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneModel(_Strict):
    provider: str = Field(description="Registered provider: jev, laya, clm.")
    url: Optional[str] = Field(default=None, description="Server base url (default: the provider's, if any).")
    api_key: Optional[SecretStr] = None
    name: Optional[str] = Field(default=None, description="Model/checkpoint (default: the provider's).")


class SystemOneRequest(_Strict):
    model: SystemOneModel
    state: Union[str, Dict[str, Any], List[Any]] = Field(description="Text, object or list to decide on.")
    questions: Dict[str, Question] = Field(min_length=1)
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Provider-specific fields forwarded as-is (e.g. Laya: lang, max_len, min_confidence; "
                    "CLM: temperature). The provider validates them.")
    debug: bool = Field(default=False, description="Include the raw provider response.")
    id_project: Optional[str] = None
    request_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_provider(self) -> "SystemOneRequest":
        spec = get_provider(self.model.provider)
        if spec is None:
            known = ", ".join(s.name for s in list_providers())
            raise ValueError(f"provider '{self.model.provider}' sconosciuto: usa uno tra {known}")
        if spec.requires_api_key and not (self.model.api_key and self.model.api_key.get_secret_value()):
            raise ValueError(f"il provider '{spec.name}' richiede model.api_key")
        if not spec.default_url and not self.model.url:
            raise ValueError(f"il provider '{spec.name}' è self-hosted: indica model.url del server")
        return self


class Answer(BaseModel):
    type: Literal["noul", "choice", "score"]
    choice: Optional[str] = None
    score: Optional[float] = None
    noul: Optional[float] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    legend: Optional[Dict[str, str]] = None


class SystemOneResponse(BaseModel):
    provider: str
    model: Optional[str] = Field(default=None, description="Model reported by the provider.")
    answers: Dict[str, Answer]
    usage: Dict[str, int] = Field(default_factory=dict)
    latency_ms: float
    warnings: List[str] = Field(default_factory=list,
                                description="Non-blocking issues (e.g. state truncated by the provider).")
    raw: Optional[Dict[str, Any]] = Field(default=None, description="Raw provider response (debug only).")
