"""Typed configuration loaded from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    embedding_provider: str = "local"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: Path = Path(".data/chroma")
    chroma_collection: str = "atlas_documents"
    chunk_size: int = Field(default=900, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    max_context_chars: int = Field(default=12000, ge=1000, le=100000)
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"local", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'openai'")
        return normalized

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info: object) -> int:
        size = info.data.get("chunk_size", 900)  # type: ignore[attr-defined]
        if value >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
