from typing import Any


def extract_llm_text(response: Any) -> str:
    """Extract plain string from LangChain LLM response, handling list-based content blocks."""
    content = getattr(response, 'content', response)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif "text" in part:
                    parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts).strip()
    return str(content).strip()
