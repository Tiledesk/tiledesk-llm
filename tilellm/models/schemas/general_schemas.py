from pydantic import BaseModel, Field
from typing import Dict, Any, Union, List, Optional
from tilellm.models.chat import ChatEntry


class SimpleAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    chat_history_dict: Optional[Dict[str, ChatEntry]]

class ReasoningAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    reasoning_content: Union[str, Dict[str, Any], list] = Field(default="No reasoningn answer")
    chat_history_dict: Optional[Dict[str, ChatEntry]]