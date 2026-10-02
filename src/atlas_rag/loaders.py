"""Document text extraction; parsers are selected by extension, never by client MIME type."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@dataclass(frozen=True)
class PageText:
    page: int | None
    text: str


def extract_pages(filename: str, content: bytes) -> list[PageText]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{extension}'. Use TXT, MD, PDF, or DOCX.")
    if not content:
        raise ValueError("The uploaded file is empty.")
    if extension in {".txt", ".md"}:
        text = content.decode("utf-8-sig", errors="replace").strip()
        return [PageText(page=None, text=text)] if text else []
    if extension == ".pdf":
        reader = PdfReader(BytesIO(content), strict=False)
        return [
            PageText(page=i + 1, text=(page.extract_text() or "").strip())
            for i, page in enumerate(reader.pages)
        ]
    document = Document(BytesIO(content))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()
    return [PageText(page=None, text=text)] if text else []
