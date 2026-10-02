from atlas_rag.grounding import validate_citations


def test_valid_citations() -> None:
    assert validate_citations("Answer [S1] and [S2].", 2) == (True, "")


def test_invalid_citation_number() -> None:
    valid, reason = validate_citations("Answer [S3].", 2)
    assert not valid
    assert "Invalid" in reason


def test_missing_citation() -> None:
    valid, reason = validate_citations("Unsupported answer.", 2)
    assert not valid
    assert "citation" in reason
