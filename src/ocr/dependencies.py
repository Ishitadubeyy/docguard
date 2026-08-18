"""Phase 1 dependency checks for OCR and preprocessing."""

from __future__ import annotations

import shutil
from importlib.util import find_spec


def get_missing_dependencies() -> list[str]:
    """Return human-readable install instructions for missing dependencies."""
    missing: list[str] = []

    if find_spec("cv2") is None:
        missing.append("opencv-python (pip install opencv-python)")

    if find_spec("fitz") is None:
        missing.append("pymupdf (pip install pymupdf)")

    if find_spec("pytesseract") is None:
        missing.append("pytesseract (pip install pytesseract)")
    elif shutil.which("tesseract") is None:
        missing.append(
            "Tesseract OCR binary (install separately and ensure `tesseract` is on PATH; "
            "Windows: winget install UB-Mannheim.TesseractOCR)"
        )

    return missing


def require_ocr_dependencies() -> None:
    """Raise ImportError when OCR dependencies are missing."""
    missing = get_missing_dependencies()
    if not missing:
        return

    details = "\n".join(f"  - {item}" for item in missing)
    raise ImportError(
        "Missing dependencies required for OCR processing:\n"
        f"{details}"
    )
