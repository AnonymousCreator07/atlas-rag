from atlas_rag.config import Settings
from atlas_rag.reranking import rerank


def test_reranking_disabled_preserves_dense_order() -> None:
    settings = Settings(reranker_enabled=False)
    rows = [
        {"document": "first", "score": 0.9},
        {"document": "second", "score": 0.8},
    ]
    assert rerank("question", rows, settings, 1) == rows[:1]
