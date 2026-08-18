"""Document ingestion and page extraction."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from src.ingestion.exceptions import DocumentValidationError
from src.ingestion.models import DocumentMetadata, DocumentPage
from src.ingestion.validator import (
    get_file_type,
    validate_document_content,
    validate_document_path,
)

PDF_RENDER_SCALE = 2.0


def ingest_document(path: str | Path) -> DocumentMetadata:
    """Validate a document and return its metadata."""
    resolved = validate_document_path(path)
    file_type = get_file_type(resolved)
    validate_document_content(resolved, file_type)

    return DocumentMetadata(
        document_id=str(uuid4()),
        filename=resolved.name,
        file_type=file_type,
        file_size_bytes=resolved.stat().st_size,
        page_count=_get_page_count(resolved, file_type),
        source_path=str(resolved),
    )


def load_document_pages(path: str | Path) -> list[DocumentPage]:
    """Load all pages from a validated document as RGB numpy arrays."""
    resolved = validate_document_path(path)
    file_type = get_file_type(resolved)
    validate_document_content(resolved, file_type)

    if file_type == "pdf":
        return _load_pdf_pages(resolved)

    return [
        DocumentPage(
            page_number=1,
            image=_load_image_array(resolved),
            metadata={"source_path": str(resolved)},
        )
    ]


def _get_page_count(path: Path, file_type: str) -> int:
    if file_type != "pdf":
        return 1

    try:
        import fitz
    except ImportError as exc:
        raise DocumentValidationError(
            "PDF page counting requires PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    with fitz.open(path) as document:
        return document.page_count


def _load_pdf_pages(path: Path) -> list[DocumentPage]:
    try:
        import fitz
    except ImportError as exc:
        raise DocumentValidationError(
            "PDF loading requires PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    pages: list[DocumentPage] = []
    matrix = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)

    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            channels = pixmap.n
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height,
                pixmap.width,
                channels,
            )
            pages.append(
                DocumentPage(
                    page_number=index,
                    image=image,
                    metadata={
                        "source_path": str(path),
                        "render_scale": PDF_RENDER_SCALE,
                    },
                )
            )

    return pages


def _load_image_array(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"))

    return rgb
