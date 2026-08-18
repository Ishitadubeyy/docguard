"""End-to-end OCR + VLM document understanding pipeline."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.document_understanding.config import TaskType, VLMConfig
from src.document_understanding.inference import run_inference
from src.document_understanding.models import Document, DocumentPage, DocumentUnderstandingResult
from src.document_understanding.vlm_interface import VLMModel
from src.document_understanding.vlm_loader import create_vlm_model
from src.ingestion.loader import ingest_document, load_document_pages
from src.ocr.engine import TesseractOCREngine
from src.ocr.models import OCRPipelineResult
from src.ocr.pipeline import process_document
from src.preprocessing.config import PreprocessingConfig

logger = logging.getLogger(__name__)


def build_document_from_ocr(
    path: str | Path,
    ocr_result: OCRPipelineResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Combine ingested page images with OCR output."""
    pages = load_document_pages(path)
    ocr_by_page = {page.page_number: page for page in ocr_result.pages}

    document_pages: list[DocumentPage] = []
    for page in pages:
        ocr_page = ocr_by_page.get(page.page_number)
        document_pages.append(
            DocumentPage(
                page_number=page.page_number,
                image=page.image,
                ocr_text=ocr_page.text if ocr_page else None,
                ocr_blocks=list(ocr_page.blocks) if ocr_page else [],
            )
        )

    return Document(
        document_id=ocr_result.document_id,
        pages=document_pages,
        metadata=metadata or {},
    )


def run_phase1_ocr(
    path: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
    ocr_engine: TesseractOCREngine | None = None,
) -> OCRPipelineResult:
    """Run the existing Phase 1 ingestion, preprocessing, and OCR pipeline."""
    return process_document(
        path,
        preprocessing_config=preprocessing_config,
        ocr_engine=ocr_engine,
    )


def run_document_understanding_pipeline(
    path: str | Path,
    task: TaskType,
    *,
    model: VLMModel | None = None,
    vlm_config: VLMConfig | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    ocr_engine: TesseractOCREngine | None = None,
    page_number: int | None = None,
) -> dict[str, Any]:
    """Run Phase 1 OCR and Phase 2 VLM document understanding."""
    metadata = ingest_document(path)
    logger.info("Starting document understanding for '%s'", metadata.filename)

    ocr_result = run_phase1_ocr(
        path,
        preprocessing_config=preprocessing_config,
        ocr_engine=ocr_engine,
    )
    document = build_document_from_ocr(
        path,
        ocr_result,
        metadata={
            "filename": metadata.filename,
            "file_type": metadata.file_type,
            "page_count": metadata.page_count,
            "source_path": metadata.source_path,
        },
    )

    vlm_model = model or create_vlm_model(vlm_config)
    results, inference_logs = run_inference(
        vlm_model,
        document,
        task,
        page_number=page_number,
    )

    return {
        "document_id": document.document_id,
        "task": task.value,
        "model_id": vlm_model.model_id,
        "metadata": document.metadata,
        "ocr": ocr_result.to_dict(),
        "results": [result.to_dict() for result in results],
        "inference_logs": [log.to_dict() for log in inference_logs],
    }


def serialize_pipeline_result(result: dict[str, Any]) -> dict[str, Any]:
    """Ensure pipeline output is JSON serializable."""
    return asdict(result) if hasattr(result, "__dataclass_fields__") else result
