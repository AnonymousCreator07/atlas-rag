"""Persistent Chroma adapter; explicit embeddings keep query and index spaces aligned."""

from functools import lru_cache
from typing import Any

import chromadb

from atlas_rag.config import Settings, get_settings


class ChromaStore:
    def __init__(self, settings: Settings) -> None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine", "description": "Atlas RAG document chunks"},
        )

    def upsert(self, ids: list[str], documents: list[str], embeddings: list[list[float]],
               metadatas: list[dict[str, Any]]) -> None:
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k,
                                       include=["documents", "metadatas", "distances"])
        rows = []
        for doc_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            rows.append({"id": doc_id, "document": document, "metadata": metadata,
                         "distance": float(distance), "score": 1.0 - float(distance)})
        return rows

    def delete_source(self, source_id: str) -> None:
        self.collection.delete(where={"source_id": source_id})

    def list_documents(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = {}
        for metadata in result["metadatas"] or []:
            source_id = metadata["source_id"]
            item = grouped.setdefault(source_id, {"source_id": source_id,
                                                  "filename": metadata["filename"], "chunks": 0})
            item["chunks"] += 1
        return sorted(grouped.values(), key=lambda item: item["filename"].lower())

    def count(self) -> int:
        return self.collection.count()


@lru_cache
def get_store(path: str, collection: str) -> ChromaStore:
    settings = get_settings().model_copy(update={"chroma_path": path, "chroma_collection": collection})
    return ChromaStore(settings)


def configured_store(settings: Settings) -> ChromaStore:
    return get_store(str(settings.chroma_path), settings.chroma_collection)
