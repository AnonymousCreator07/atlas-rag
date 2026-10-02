"""Ingestion orchestration: extraction -> chunking -> embeddings -> persistent upsert."""

import hashlib
import re
from pathlib import Path

from atlas_rag.chunking import chunk_text
from atlas_rag.config import Settings
from atlas_rag.embeddings import Embedder
from atlas_rag.loaders import extract_pages
from atlas_rag.vector_store import ChromaStore

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def ingest_document(
    filename: str, content: bytes, settings: Settings, embedder: Embedder, store: ChromaStore
) -> dict[str, object]:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Files must be 25 MB or smaller.")
    pages = extract_pages(filename, content)
    safe_filename = Path(filename).name[:255]
    source_id = hashlib.sha256(content).hexdigest()
    clean_name = re.sub(r"[^\w. -]", "_", safe_filename)
    records: list[tuple[str, int | None, int, str]] = []
    for page in pages:
        for index, chunk in enumerate(
            chunk_text(page.text, settings.chunk_size, settings.chunk_overlap)
        ):
            records.append((chunk, page.page, index, clean_name))
    if not records:
        raise ValueError("No extractable text was found. Scanned PDFs need OCR before ingestion.")

    # Bounded batches prevent a large document from creating an oversized embedding request.
    for start in range(0, len(records), 64):
        batch = records[start : start + 64]
        texts = [record[0] for record in batch]
        ids = [f"{source_id}:{start + offset}" for offset in range(len(batch))]
        metadata = [
            {
                "source_id": source_id,
                "filename": record[3],
                "page": record[1] or 0,
                "chunk_index": record[2],
            }
            for record in batch
        ]
        store.upsert(ids, texts, embedder.embed(texts), metadata)
    return {
        "source_id": source_id,
        "filename": clean_name,
        "chunks_added": len(records),
        "pages_read": len(pages),
        "status": "indexed",
    }
