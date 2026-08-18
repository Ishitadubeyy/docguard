"""OCR pipeline and structured output generation."""

from src.ocr.dependencies import (
    get_missing_dependencies,
    require_ocr_dependencies,
)
from src.ocr.engine import TesseractOCREngine
from src.ocr.models import OCRBlock, OCRPageResult, OCRPipelineResult
from src.ocr.pipeline import process_document

__all__ = [
    "OCRBlock",
    "OCRPageResult",
    "OCRPipelineResult",
    "TesseractOCREngine",
    "get_missing_dependencies",
    "process_document",
    "require_ocr_dependencies",
]
