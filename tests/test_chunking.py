from atlas_rag.chunking import chunk_text


def test_chunking_is_empty_for_blank_text() -> None:
    assert chunk_text("  \n", chunk_size=20, overlap=3) == []


def test_chunking_respects_limit_and_preserves_content() -> None:
    text = "Alpha is first. Beta is second. Gamma is third. Delta is fourth."
    chunks = chunk_text(text, chunk_size=28, overlap=5)
    assert chunks
    assert all(len(chunk) <= 28 for chunk in chunks)
    for word in ("Alpha", "Beta", "Gamma", "Delta"):
        assert any(word in chunk for chunk in chunks)


def test_chunking_rejects_invalid_overlap() -> None:
    try:
        chunk_text("text", chunk_size=10, overlap=10)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("Expected invalid overlap to fail")
