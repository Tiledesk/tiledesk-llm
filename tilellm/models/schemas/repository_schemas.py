import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.documents import Document

from tilellm.models.vector_store import Engine


class RepositoryQueryResult(BaseModel):
    id: str
    metadata_id: str
    metadata_source: str
    metadata_type: str
    date: Optional[str] = Field(default="Date not defined")
    text: Optional[str] | None = None


class RepositoryItems(BaseModel):
    matches: List[RepositoryQueryResult]

class RepositoryEngine(BaseModel):
    engine: Engine

class RepositoryNamespace(BaseModel):
    namespace: str
    engine: Engine


class RepositoryItem(BaseModel):
    id: str
    namespace: str
    engine: Engine


class ScrapeStatusReq(BaseModel):
    id: str
    namespace: str
    namespace_list: Optional[List[str]] | None = None
    engine: Engine


class ScrapeStatusResponse(BaseModel):
    status_message: str = Field(default="Crawling is not started")
    status_code: int = Field(default=0)
    queue_order: int = Field(default=-1)


class IndexingResult(BaseModel):
    id: str | None = None
    chunks: int | None = None
    total_tokens: int | None = None
    cost: str | None = None
    status: int = Field(default=300)
    date: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"))
    error: Optional[str] | None = None


class RepositoryItemNamespaceResult(BaseModel):
    namespace: str
    vector_count: int


class RepositoryIdSummaryResult(BaseModel):
    metadata_id: str
    source: str
    chunks_count: int


class RepositoryNamespaceResult(BaseModel):
    namespaces: Optional[List[RepositoryItemNamespaceResult]]


class RepositoryDescNamespaceResult(BaseModel):
    namespace_desc: RepositoryItemNamespaceResult
    ids: Optional[List[RepositoryIdSummaryResult]]

class MyDocument(Document):
    sparse_values: Optional[dict]