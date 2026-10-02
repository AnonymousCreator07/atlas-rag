import pytest

from atlas_rag.loaders import extract_pages


def test_text_loader_decodes_utf8_bom() -> None:
    pages = extract_pages("notes.txt", b"\xef\xbb\xbfHello RAG")
    assert pages[0].text == "Hello RAG"


def test_reject_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        extract_pages("payload.exe", b"bad")


def test_reject_empty_file() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_pages("notes.txt", b"")
