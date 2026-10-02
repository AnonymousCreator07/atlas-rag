"""Deterministic character-window chunking with overlap and whitespace-aware boundaries."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split normalized text into bounded chunks, preferring whitespace near each boundary."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Avoid tiny chunks when there is no nearby whitespace (e.g. a long identifier).
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        # Keep the next window aligned to a word boundary where possible.
        space = text.find(" ", next_start, min(next_start + 80, len(text)))
        start = space + 1 if space >= 0 else next_start
    return chunks
