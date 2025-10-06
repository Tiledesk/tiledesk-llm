from pydantic import BaseModel, ValidationError, Field
from typing import Dict, Optional, Union, List, Any

from tilellm.models.schemas.multimodal_content import MultimodalContent, TextContent, ImageContent, DocumentContent


class ChatEntry(BaseModel):
    question: Union[str, List[MultimodalContent]]
    answer: str
    # metadata: Optional[Dict[str, str]] = None  # Optional field for additional data
    def get_question_text(self) -> str:
        """Estrae il testo dalla question, utile per logging/display"""
        if isinstance(self.question, str):
            return self.question
        else:
            # Estrae solo i contenuti text
            texts = [c.text for c in self.question if isinstance(c, TextContent)]
            return " ".join(texts) if texts else "[Contenuto multimodale]"


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

                # Gestisci la conversione di question multimodale
                if isinstance(entry_data.get('question'), list):
                    # Se è una lista di dict, convertila in MultimodalContent
                    question_list = []
                    for item in entry_data['question']:
                        if isinstance(item, dict):
                            content_type = item.get('type')
                            if content_type == 'text':
                                question_list.append(TextContent(**item))
                            elif content_type == 'image':
                                question_list.append(ImageContent(**item))
                            elif content_type == 'document':
                                question_list.append(DocumentContent(**item))
                        else:
                            question_list.append(item)
                    entry_data['question'] = question_list

                chat_history[key] = ChatEntry(**entry_data)
            except (TypeError, ValueError) as e:
                raise ValidationError(f"Error processing entry '{key}': {str(e)}")
        return cls(chat_history=chat_history)