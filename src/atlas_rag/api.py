"""FastAPI entry point. Interactive OpenAPI docs are available at /docs."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from atlas_rag.config import get_settings
from atlas_rag.embeddings import configured_embedder
from atlas_rag.ingestion import MAX_UPLOAD_BYTES, ingest_document
from atlas_rag.retrieval import answer_question
from atlas_rag.schemas import DocumentItem, IngestResponse, QueryRequest, QueryResponse
from atlas_rag.vector_store import configured_store

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Atlas RAG API",
    version="0.1.0",
    description="Document ingestion, semantic retrieval, and cited answers.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {
        "status": "ok",
        "indexed_chunks": configured_store(settings).count(),
        "embedding_provider": settings.embedding_provider,
        "generation_configured": bool(settings.openai_api_key),
    }


@app.post("/documents", response_model=IngestResponse, tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> dict:  # noqa: B008
    filename = file.filename or "upload"
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Files must be 25 MB or smaller.")
    try:
        return ingest_document(
            filename, content, settings, configured_embedder(settings), configured_store(settings)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Document indexing failed. Check server logs."
        ) from exc


@app.get("/documents", response_model=list[DocumentItem], tags=["documents"])
def list_documents() -> list[dict]:
    return configured_store(settings).list_documents()


@app.delete("/documents/{source_id}", tags=["documents"])
def delete_document(source_id: str) -> dict:
    if len(source_id) != 64 or any(char not in "0123456789abcdef" for char in source_id):
        raise HTTPException(status_code=400, detail="source_id must be a SHA-256 document ID.")
    configured_store(settings).delete_source(source_id)
    return {"deleted": source_id}


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(request: QueryRequest) -> dict:
    store = configured_store(settings)
    if store.count() == 0:
        raise HTTPException(
            status_code=409, detail="No documents indexed yet. Upload a document first."
        )
    try:
        return answer_question(
            request.question,
            request.top_k or settings.top_k,
            settings,
            configured_embedder(settings),
            store,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Query failed. Check provider configuration and server logs."
        ) from exc
