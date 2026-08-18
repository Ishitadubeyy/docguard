"""VLM inference orchestration."""

from __future__ import annotations

import logging

from src.document_understanding.config import TaskType
from src.document_understanding.eval_logging import InferenceLogRecord, build_inference_log
from src.document_understanding.models import Document, DocumentUnderstandingResult
from src.document_understanding.parser import parse_structured_output
from src.document_understanding.prompts import build_prompt
from src.document_understanding.vlm_interface import VLMModel, VLMModelError

logger = logging.getLogger(__name__)


def run_inference(
    model: VLMModel,
    document: Document,
    task: TaskType,
    *,
    page_number: int | None = None,
) -> tuple[list[DocumentUnderstandingResult], list[InferenceLogRecord]]:
    """Run VLM inference for one or all pages in a document."""
    if not document.pages:
        raise ValueError("Document has no pages to process")

    if page_number is not None:
        pages = [page for page in document.pages if page.page_number == page_number]
        if not pages:
            raise ValueError(f"Page {page_number} not found in document")
    else:
        pages = document.pages

    if not model.is_loaded:
        logger.info("VLM model not loaded; loading '%s'", model.model_id)
        model.load()

    results: list[DocumentUnderstandingResult] = []
    logs: list[InferenceLogRecord] = []

    for page in pages:
        if page.image is None:
            raise ValueError(f"Page {page.page_number} is missing an image")

        prompt = build_prompt(
            task=task,
            ocr_text=page.ocr_text,
            ocr_blocks=page.ocr_blocks,
        )
        logger.info(
            "Running VLM inference for document '%s' page %s task '%s'",
            document.document_id,
            page.page_number,
            task.value,
        )

        try:
            generation = model.generate(page.image, prompt)
        except VLMModelError:
            raise
        except Exception as exc:
            raise VLMModelError(
                f"Inference failed for page {page.page_number}: {exc}"
            ) from exc

        parsed = parse_structured_output(
            generation.text,
            page_number=page.page_number,
        )
        results.append(parsed)

        log_record = build_inference_log(
            model_id=model.model_id,
            task=task.value,
            latency_seconds=generation.latency_seconds,
            result=parsed,
            ocr_text_length=len(page.ocr_text or ""),
            page_number=page.page_number,
        )
        logs.append(log_record)
        logger.info(
            "Inference log: model=%s task=%s latency=%.2fs parse_success=%s "
            "schema_valid=%s schema_normalized=%s document_type=%s fields=%s ocr_text_length=%s",
            log_record.model_id,
            log_record.task,
            log_record.latency_seconds,
            log_record.parse_success,
            log_record.schema_valid,
            log_record.schema_normalized,
            log_record.document_type,
            log_record.number_of_extracted_fields,
            log_record.ocr_text_length,
        )

    return results, logs
