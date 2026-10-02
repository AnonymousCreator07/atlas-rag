"""Retrieve ranked chunks and assemble the public response."""

from time import perf_counter

from atlas_rag.config import Settings
from atlas_rag.embeddings import Embedder
from atlas_rag.generation import generate_answer
from atlas_rag.vector_store import ChromaStore


def answer_question(
    question: str, top_k: int, settings: Settings, embedder: Embedder, store: ChromaStore
) -> dict:
    start = perf_counter()
    query_vector = embedder.embed([question])[0]
    rows = store.query(query_vector, top_k)
    retrieval_ms = (perf_counter() - start) * 1000
    answer, generation_ms, model = generate_answer(question, rows, settings)
    sources = [
        {
            "source_id": row["metadata"]["source_id"],
            "filename": row["metadata"]["filename"],
            "page": row["metadata"].get("page") or None,
            "chunk_index": row["metadata"]["chunk_index"],
            "score": row["score"],
            "excerpt": row["document"],
        }
        for row in rows
    ]
    return {
        "answer": answer,
        "sources": sources,
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "model": model,
    }
