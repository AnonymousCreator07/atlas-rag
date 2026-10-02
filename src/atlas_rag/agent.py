"""Bounded agentic retrieval: retrieve, assess, optionally reformulate, retrieve again."""

from typing import Any

from atlas_rag.config import Settings
from atlas_rag.embeddings import Embedder
from atlas_rag.reranking import rerank
from atlas_rag.vector_store import ChromaStore


def retrieve_with_agent(
    question: str,
    top_k: int,
    settings: Settings,
    embedder: Embedder,
    store: ChromaStore,
) -> tuple[str, list[dict[str, Any]], int]:
    """Use a bounded second retrieval pass when the first pass is weak.

    The agent is deliberately constrained to retrieval/query reformulation; it cannot
    execute arbitrary tools or actions.
    """
    def retrieve(query: str) -> list[dict[str, Any]]:
        rows = store.query(embedder.embed([query])[0], settings.retrieval_candidates)
        return rerank(query, rows, settings, top_k)

    rows = retrieve(question)
    if not settings.agentic_rag_enabled or not rows:
        return question, rows, 1

    best = max(row.get("rerank_score", row["score"]) for row in rows)
    if best >= settings.agentic_min_score:
        return question, rows, 1

    if not settings.openai_api_key:
        return question, rows, 1

    from openai import OpenAI

    prompt = (
        "Rewrite the user's question into one concise search query that is likely to "
        "retrieve the needed document passage. Return only the rewritten query.\n\n"
        f"User question: {question}"
    )
    response = OpenAI(api_key=settings.openai_api_key).chat.completions.create(
        model=settings.openai_chat_model,
        messages=[{"role": "system", "content": "You are a retrieval query reformulator."},
                  {"role": "user", "content": prompt}],
        temperature=0,
    )
    rewritten = (response.choices[0].message.content or question).strip()[:1000]
    second = retrieve(rewritten)
    if second:
        return rewritten, second, 2
    return question, rows, 1
