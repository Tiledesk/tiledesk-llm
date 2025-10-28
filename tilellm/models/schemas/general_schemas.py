from pydantic import BaseModel, Field
from typing import Dict, Any, Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tilellm.models.chat import ChatEntry

class PromptTokenInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class SimpleAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    tools_log: Optional[list] = None
    chat_history_dict: Optional[Dict[str, "ChatEntry"]] = None
    prompt_token_info: Optional[PromptTokenInfo] = None

class ReasoningAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    reasoning_content: Union[str, Dict[str, Any], list] = Field(default="No reasoningn answer")
    chat_history_dict: Optional[Dict[str, "ChatEntry"]] = None

# Risolvi forward references dopo che ChatEntry è caricato
def rebuild_models():
    from tilellm.models.chat import ChatEntry
    SimpleAnswer.model_rebuild()
    ReasoningAnswer.model_rebuild()