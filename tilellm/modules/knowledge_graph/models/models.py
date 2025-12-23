from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class Node(BaseModel):
    """Model representing a node in the knowledge graph"""
    id: Optional[str] = None  # Neo4j generates IDs automatically
    label: str = Field(..., description="The label/type of the node (e.g., 'Person', 'Document')")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Key-value properties of the node")

    class Config:
        json_schema_extra = {
            "example": {
                "label": "Document",
                "properties": {
                    "title": "Introduction to RAG",
                    "content": "RAG stands for Retrieval Augmented Generation...",
                    "embedding": [0.1, 0.2, 0.3]
                }
            }
        }

class NodeUpdate(BaseModel):
    """Model for updating a node's properties"""
    label: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class Relationship(BaseModel):
    """Model representing a relationship between nodes"""
    id: Optional[str] = None
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    type: str = Field(..., description="Type of relationship (e.g., 'RELATES_TO', 'CONTAINS')")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")

    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "123",
                "target_id": "456",
                "type": "REFERENCES",
                "properties": {"weight": 0.8}
            }
        }

class RelationshipUpdate(BaseModel):
    """Model for updating a relationship's properties"""
    type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None