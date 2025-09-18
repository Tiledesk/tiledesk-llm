from pydantic import BaseModel, Field
from typing import Dict, Any, Union, List, Optional
from tilellm.models.chat import ChatEntry


class PromptTokenInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int

class SimpleAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    chat_history_dict: Optional[Dict[str, ChatEntry]]
    prompt_token_info: Optional[PromptTokenInfo] = None


class ReasoningAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    reasoning_content: Union[str, Dict[str, Any], list] = Field(default="No reasoningn answer")
    chat_history_dict: Optional[Dict[str, ChatEntry]]