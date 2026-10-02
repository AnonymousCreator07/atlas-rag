"""Optional cross-encoder reranking for retrieved passages."""

from functools import lru_cache
from typing import Any

from atlas_rag.config import Settings


@lru_cache
def _model(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


def rerank(
    question: str, rows: list[dict[str, Any]], settings: Settings, top_k: int
) -> list[dict[str, Any]]:
    """Rerank dense-retrieval candidates while preserving a safe local fallback."""
    if not rows or not settings.reranker_enabled:
        return rows[:top_k]

    model = _model(settings.reranker_model)
    pairs = [(question, row["document"]) for row in rows]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = []
    for row, score in zip(rows, scores, strict=True):
        item = dict(row)
        item["rerank_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_k]
