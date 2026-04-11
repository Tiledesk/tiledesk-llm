"""
Agent middleware for cleaning base64 content from messages before LLM calls.
Prevents context overflow when handling multimodal inputs (images, documents).
"""

import json
import logging
import re
from typing import Any, Dict

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime


logger = logging.getLogger(__name__)


class MessageCleaningMiddleware(AgentMiddleware):
    """
    Middleware that extracts and stores base64 content before LLM calls.

    Instead of sending large base64 strings to the model, this middleware:
    1. Detects large base64 content in messages
    2. Stores them in a reference dictionary
    3. Replaces them with lightweight text references (e.g., <base64_ref_1>)

    This prevents context window overflow while keeping multimodal data accessible
    to tools that need it.
    """

    def __init__(self, base64_storage: Dict[str, Any]):
        """
        Initialize the middleware.

        Args:
            base64_storage: Dictionary to store extracted base64 content.
                           Shared across the agent lifecycle.
        """
        self.base64_storage = base64_storage
        self.base64_counter = 0
        self.logger = logger

    def _clean_message_content(self, content: str) -> tuple[str, bool]:
        """
        Extract and replace large base64 content in a message.

        Args:
            content: Message content string

        Returns:
            Tuple of (cleaned_content, was_cleaned)
        """
        if len(content) < 1000:
            return content, False

        # Try to parse as JSON (typical MCP tool response)
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                was_cleaned = False
                for key in ['images_base64', 'documents_base64', 'file_content']:
                    if key in data:
                        # Handle lists
                        if isinstance(data[key], list):
                            cleaned_list = []
                            for item in data[key]:
                                if isinstance(item, str) and len(item) > 1000:
                                    self.base64_counter += 1
                                    ref_id = f"base64_ref_{self.base64_counter}"
                                    self.base64_storage[ref_id] = {
                                        'data': re.sub(r'[\n\r]', '', item),
                                        'type': 'image' if key == 'images_base64' else 'document',
                                        'media_type': 'image/png'
                                    }
                                    cleaned_list.append(f"<{ref_id}>")
                                    was_cleaned = True
                                else:
                                    cleaned_list.append(item)
                            data[key] = cleaned_list
                        # Handle single strings
                        elif isinstance(data[key], str) and len(data[key]) > 1000:
                            self.base64_counter += 1
                            ref_id = f"base64_ref_{self.base64_counter}"
                            self.base64_storage[ref_id] = {
                                'data': re.sub(r'[\n\r]', '', data[key]),
                                'type': 'image' if key == 'images_base64' else 'document',
                                'media_type': 'image/png'
                            }
                            data[key] = f"<{ref_id}>"
                            was_cleaned = True

                if was_cleaned:
                    return json.dumps(data), True
        except json.JSONDecodeError:
            pass

        # Fallback: raw base64 string without JSON structure
        sample = (content[:500] + content[-500:]).strip()
        if re.fullmatch(r'[A-Za-z0-9+/=]+', re.sub(r'\s+', '', sample)):
            self.base64_counter += 1
            ref_id = f"base64_ref_{self.base64_counter}"
            self.base64_storage[ref_id] = {
                'data': re.sub(r'[\n\r]', '', content),
                'type': 'binary',
                'media_type': 'application/octet-stream'
            }
            return f"[BINARY_REF:{ref_id}:length={len(content)}]", True

        return content, False

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime
    ) -> Dict[str, Any] | None:
        """
        Clean messages before LLM call.

        This hook is called just before the model is invoked.
        It extracts large base64 content and replaces with references.

        Args:
            state: Current agent state containing messages
            runtime: LangGraph runtime context

        Returns:
            Dictionary with updated messages, or None if no changes needed
        """
        messages = state.get("messages", [])

        self.logger.info(f"[MIDDLEWARE] Cleaning {len(messages)} messages before model call")
        cleaned_messages = []

        for idx, msg in enumerate(messages):
            if not hasattr(msg, 'content'):
                cleaned_messages.append(msg)
                continue

            content = msg.content
            was_cleaned = False
            cleaned_content = content

            # Handle string content
            if isinstance(content, str):
                cleaned_content, was_cleaned = self._clean_message_content(content)

            # Handle multimodal content (list of dicts with text/image_url)
            elif isinstance(content, list):
                cleaned_list = []
                for item in content:
                    if isinstance(item, dict):
                        # Clean text items that might contain base64
                        if item.get('type') == 'text':
                            text_content = item.get('text', '')
                            if isinstance(text_content, str) and len(text_content) > 1000:
                                cleaned_text, item_cleaned = self._clean_message_content(text_content)
                                if item_cleaned:
                                    was_cleaned = True
                                    cleaned_list.append({'type': 'text', 'text': cleaned_text})
                                else:
                                    cleaned_list.append(item)
                            else:
                                cleaned_list.append(item)
                        else:
                            # Keep image_url and other types as-is
                            cleaned_list.append(item)
                    else:
                        cleaned_list.append(item)
                cleaned_content = cleaned_list

            if was_cleaned:
                self.logger.debug(f"[MIDDLEWARE] Message {idx} ({msg.type}): cleaned base64")

                # Reconstruct message with cleaned content
                if msg.type == 'tool':
                    cleaned_messages.append(ToolMessage(
                        content=cleaned_content,
                        tool_call_id=msg.tool_call_id,
                        name=getattr(msg, 'name', None),
                        id=getattr(msg, 'id', None)
                    ))
                elif msg.type == 'ai':
                    cleaned_messages.append(AIMessage(
                        content=cleaned_content,
                        tool_calls=getattr(msg, 'tool_calls', None),
                        id=getattr(msg, 'id', None)
                    ))
                elif msg.type == 'human':
                    cleaned_messages.append(HumanMessage(content=cleaned_content))
                elif msg.type == 'system':
                    cleaned_messages.append(SystemMessage(content=cleaned_content))
                else:
                    cleaned_messages.append(msg)
            else:
                cleaned_messages.append(msg)

        self.logger.debug(f"[MIDDLEWARE] Cleaned messages: {len(cleaned_messages)}, "
                         f"storage refs: {len(self.base64_storage)}")

        return {"messages": cleaned_messages}
