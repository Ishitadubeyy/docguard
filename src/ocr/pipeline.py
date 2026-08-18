"""End-to-end document OCR pipeline."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.loader import ingest_document, load_document_pages
from src.ocr.engine import TesseractOCREngine
from src.ocr.models import OCRPipelineResult
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipeline import preprocess_for_ocr


def process_document(
    path: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
    ocr_engine: TesseractOCREngine | None = None,
) -> OCRPipelineResult:
    """Ingest, preprocess, and OCR a supported document."""
    metadata = ingest_document(path)
    pages = load_document_pages(path)
    engine = ocr_engine or TesseractOCREngine()

    ocr_pages = []
    for page in pages:
        preprocessed = preprocess_for_ocr(page.image, preprocessing_config)
        ocr_pages.append(engine.recognize(preprocessed, page_number=page.page_number))

    return OCRPipelineResult(document_id=metadata.document_id, pages=ocr_pages)
