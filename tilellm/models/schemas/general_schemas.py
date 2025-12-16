from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Union, List, Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tilellm.models.chat import ChatEntry

class PromptTokenInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ReasoningConfig(BaseModel):
    """
    Configurazione unificata per reasoning models.
    Supporta parametri specifici per ogni provider:
    - OpenAI (GPT-5): reasoning_effort, reasoning_summary
    - Anthropic (Claude): type, budget_tokens
    - Google (Gemini 2.5): thinkingBudget
    - Google (Gemini 3.0): thinkingLevel
    - DeepSeek: nessun parametro specifico
    """
    # Controllo visibilità thinking nello stream
    show_thinking_stream: bool = Field(
        default=True,
        description="Se True, mostra il thinking content nello stream. Se False, lo nasconde ma lo include nella risposta finale"
    )

    # OpenAI GPT-5 specific
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="OpenAI GPT-5: Effort level for reasoning (low, medium, high)"
    )
    reasoning_summary: Optional[Literal["auto", "always", "never"]] = Field(
        default=None,
        description="OpenAI GPT-5: When to include reasoning summary (auto, always, never)"
    )

    # Anthropic Claude specific
    type: Optional[Literal["enabled", "disabled"]] = Field(
        default=None,
        description="Anthropic Claude: Enable/disable thinking mode"
    )
    budget_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        le=100000,
        description="Anthropic Claude: Token budget for thinking (0-100000)"
    )

    # Google Gemini 2.5 specific
    thinkingBudget: Optional[int] = Field(
        default=None,
        description="Gemini 2.5: Thinking token budget. -1=dynamic, 0=disabled, positive=specific budget (max 32000)"
    )

    # Google Gemini 3.0 specific
    thinkingLevel: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="Gemini 3.0: Thinking level (low, medium, high)"
    )

    @field_validator("thinkingBudget")
    @classmethod
    def validate_thinking_budget(cls, v):
        """Valida il thinkingBudget per Gemini 2.5"""
        if v is not None:
            if v < -1:
                raise ValueError("thinkingBudget must be >= -1")
            if v > 32000:
                raise ValueError("thinkingBudget cannot exceed 32000")
        return v

class SimpleAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    tools_log: Optional[list] = None
    chat_history_dict: Optional[Dict[str, "ChatEntry"]] = None
    prompt_token_info: Optional[PromptTokenInfo] = None

class ReasoningAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    reasoning_content: Union[str, Dict[str, Any], list] = Field(default="No reasoning answer")
    chat_history_dict: Optional[Dict[str, "ChatEntry"]] = None
    prompt_token_info: Optional[PromptTokenInfo] = None

# Risolvi forward references dopo che ChatEntry è caricato
def rebuild_models():
    from tilellm.models.chat import ChatEntry
    SimpleAnswer.model_rebuild()
    ReasoningAnswer.model_rebuild()