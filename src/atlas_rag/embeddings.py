"""Embedding provider abstraction shared by indexing and query embedding."""

from functools import lru_cache
from typing import Protocol

from atlas_rag.config import Settings, get_settings


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(result.data, key=lambda item: item.index)]


@lru_cache
def get_embedder(provider: str, model: str, api_key: str | None) -> Embedder:
    settings = get_settings()
    if provider == "openai":
        return OpenAIEmbedder(settings)
    return LocalEmbedder(model)


def configured_embedder(settings: Settings) -> Embedder:
    model = settings.local_embedding_model if settings.embedding_provider == "local" else settings.openai_embedding_model
    return get_embedder(settings.embedding_provider, model, settings.openai_api_key)
