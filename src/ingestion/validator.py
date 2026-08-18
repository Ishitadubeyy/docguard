"""File type and content validation for supported document formats."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.exceptions import (
    DocumentNotFoundError,
    DocumentValidationError,
    UnsupportedDocumentTypeError,
)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpeg",
}


def validate_document_path(path: str | Path) -> Path:
    """Validate that a path exists and has a supported extension."""
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise DocumentNotFoundError(f"Document not found: {resolved}")

    if not resolved.is_file():
        raise DocumentValidationError(f"Path is not a file: {resolved}")

    extension = resolved.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentTypeError(
            f"Unsupported file type '{extension or '(none)'}'. "
            f"Supported extensions: {supported}"
        )

    return resolved


def get_file_type(path: Path) -> str:
    """Return the normalized file type for a validated path."""
    return SUPPORTED_EXTENSIONS[path.suffix.lower()]


def validate_document_content(path: Path, file_type: str) -> None:
    """Validate that the file content matches its declared type."""
    if file_type == "pdf":
        _validate_pdf_content(path)
        return

    _validate_image_content(path)


def _validate_pdf_content(path: Path) -> None:
    try:
        import fitz
    except ImportError as exc:
        raise DocumentValidationError(
            "PDF validation requires PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    try:
        with fitz.open(path) as document:
            if document.page_count < 1:
                raise DocumentValidationError(f"PDF has no pages: {path}")
    except DocumentValidationError:
        raise
    except Exception as exc:
        raise DocumentValidationError(f"Invalid or corrupted PDF: {path}") from exc


def _validate_image_content(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DocumentValidationError(
            "Image validation requires Pillow. Install with: pip install Pillow"
        ) from exc

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise DocumentValidationError(f"Invalid or corrupted image: {path}") from exc
