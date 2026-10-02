"""Retrieve, rerank, optionally agentically reformulate, and synthesize grounded answers."""

from time import perf_counter

from atlas_rag.agent import retrieve_with_agent
from atlas_rag.config import Settings
from atlas_rag.embeddings import Embedder
from atlas_rag.generation import generate_answer
from atlas_rag.grounding import fallback_answer, verify_grounding
from atlas_rag.memory import ConversationMemory
from atlas_rag.monitoring import log_event, metrics
from atlas_rag.vector_store import ChromaStore


def answer_question(
    question: str,
    top_k: int,
    settings: Settings,
    embedder: Embedder,
    store: ChromaStore,
    memory: ConversationMemory | None = None,
    session_id: str | None = None,
    use_agentic: bool | None = None,
) -> dict:
    started = perf_counter()
    history = memory.get(session_id) if memory and session_id else []

    agentic_original = settings.agentic_rag_enabled
    if use_agentic is not None:
        settings.agentic_rag_enabled = use_agentic
    try:
        retrieval_start = perf_counter()
        effective_question, rows, retrieval_passes = retrieve_with_agent(
            question, top_k, settings, embedder, store
        )
        retrieval_ms = (perf_counter() - retrieval_start) * 1000

        answer, generation_ms, model = generate_answer(
            question, rows, settings, history=history
        )
        grounded, grounding_reason = verify_grounding(answer, rows, settings)
        if not grounded:
            answer = fallback_answer(grounding_reason)

        sources = [
            {
                "source_id": row["metadata"]["source_id"],
                "filename": row["metadata"]["filename"],
                "page": row["metadata"].get("page") or None,
                "chunk_index": row["metadata"]["chunk_index"],
                "score": row["score"],
                "rerank_score": row.get("rerank_score"),
                "excerpt": row["document"],
            }
            for row in rows
        ]

        if memory and session_id:
            memory.add(session_id, "user", question)
            memory.add(session_id, "assistant", answer)

        elapsed = (perf_counter() - started) * 1000
        metrics.record("query", elapsed, success=grounded)
        log_event(
            "query_completed",
            session_id=session_id,
            question_length=len(question),
            retrieved=len(rows),
            retrieval_passes=retrieval_passes,
            effective_query=effective_question,
            grounded=grounded,
            latency_ms=round(elapsed, 2),
        )
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "model": model,
            "session_id": session_id,
            "retrieval_passes": retrieval_passes,
            "grounded": grounded,
            "grounding_reason": grounding_reason,
        }
    except Exception:
        metrics.record("query", (perf_counter() - started) * 1000, success=False)
        raise
    finally:
        settings.agentic_rag_enabled = agentic_original
