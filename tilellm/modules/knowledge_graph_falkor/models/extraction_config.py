"""
Pydantic models for extraction configuration.
"""

from typing import List
from pydantic import BaseModel, Field


class ExtractionConfig(BaseModel):
    """Configuration for a specific knowledge graph domain."""
    
    id: str = Field(..., description="Unique identifier for the configuration (e.g., 'generic', 'debt_recovery')")
    version: str = Field(..., description="Version of the configuration (semantic versioning)")
    domain: str = Field(..., description="Domain name (e.g., 'generic', 'debt_recovery')")
    enabled: bool = Field(default=True, description="Whether this configuration is enabled")
    description: str = Field(default="", description="Human-readable description")
    
    # Entity and relationship types
    entity_types: List[str] = Field(..., description="List of entity type names")
    relationship_types: List[str] = Field(..., description="List of relationship type names")
    
    # Extraction prompt
    extraction_prompt: str = Field(..., description="The prompt template for graph extraction")