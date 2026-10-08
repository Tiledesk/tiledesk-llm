"""Identifiers that end up in the Cypher text (labels, relationship types,
property keys) — shared by the async and sync FalkorDB repositories.

They can't be query parameters, so they are interpolated: reject anything
that isn't an identifier when it comes from the API, normalize it when it is
produced by the LLM over user documents.
"""
import re
from typing import Optional


def _cypher_ident(name: str) -> str:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"invalid Cypher identifier: {str(name)[:60]!r}")
    return name


def _llm_type_to_ident(value: Optional[str], default: str) -> str:
    """entity_type / relationship_type produced by the LLM → valid identifier
    ("medical device" → MEDICAL_DEVICE) instead of a broken or injected query."""
    ident = re.sub(r"\W+", "_", (value or "").strip().upper()).strip("_")
    return ident if ident.isidentifier() else default
