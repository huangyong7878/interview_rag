"""PDF loader: type detection and image extraction via PyMuPDF."""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from src.config import settings


@dataclass
class PageImage:
    """A single page extracted as an image."""
    page_num: int       # 1-indexed
    image: Image.Image
    width: int
    height: int


@dataclass
class PDFInfo:
    """Metadata about the loaded PDF."""
    file_path: Path
    page_count: int
    is_scanned: bool
    pages: list[PageImage]


def detect_pdf_type(doc: fitz.Document) -> bool:
    """Check if PDF is scanned (image-only, no text layer).

    Returns True if scanned, False if text-based.
    """
    text_chars = 0
    for page in doc:
        text = page.get_text()
        text_chars += len(text.strip())
    # If total extracted text across all pages is under 50 chars, it's scanned
    return text_chars < 50


def extract_page_images(doc: fitz.Document) -> list[PageImage]:
    """Extract each page as a PIL Image at configured DPI."""
    pages = []
    dpi = settings.image_dpi

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(PageImage(
            page_num=i + 1,
            image=img,
            width=pix.width,
            height=pix.height,
        ))

    return pages


def load_pdf(file_path: str | Path) -> PDFInfo:
    """Load a PDF, detect type, and extract page images.

    For scanned PDFs, extracts page images for OCR/VLM processing.
    For text-based PDFs, could skip image extraction, but for consistency
    we always extract images (VLM can also process text-based pages).

    TODO: is_scanned flag is detected but never consumed by the parser chain.
    For text-based PDFs (is_scanned=False), PyMuPDF can extract structured
    Markdown directly via page.get_text("markdown"), which preserves tables
    and headings without OCR. This would be faster AND higher quality than
    re-rendering pages as images and running PaddleOCR on them.
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(str(file_path))
    is_scanned = detect_pdf_type(doc)
    pages = extract_page_images(doc)
    doc.close()

    return PDFInfo(
        file_path=file_path,
        page_count=len(pages),
        is_scanned=is_scanned,
        pages=pages,
    )
