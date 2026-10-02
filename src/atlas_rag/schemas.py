"""Public request and response contracts for the HTTP API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    source_id: str
    filename: str
    page: int | None = None
    chunk_index: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_ms: float
    generation_ms: float
    model: str


class IngestResponse(BaseModel):
    source_id: str
    filename: str
    chunks_added: int
    pages_read: int
    status: str


class DocumentItem(BaseModel):
    source_id: str
    filename: str
    chunks: int
