"""Public request and response contracts for the HTTP API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    use_agentic: bool | None = None


class Source(BaseModel):
    source_id: str
    filename: str
    page: int | None = None
    chunk_index: int
    score: float
    rerank_score: float | None = None
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_ms: float
    generation_ms: float
    model: str
    session_id: str | None = None
    retrieval_passes: int = 1
    grounded: bool = True
    grounding_reason: str = ""


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
