"""Document understanding task layer built on the baseline VLM inference layer.

Architecture::

    VLM Loader -> VLM Inference -> Document Analyzer -> Task-specific output

Model loading is owned by ``vlm_loader``; this module only selects a prompt,
delegates generation to ``vlm_inference`` and parses the result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.document_understanding.extraction_schema import (
    AnalysisTask,
    DocumentType,
    TaskResult,
    parse_classification,
    parse_key_value,
    parse_question_answer,
    parse_summary,
    parse_table,
)
from src.document_understanding.task_prompts import build_task_prompt
from src.document_understanding.vlm_config import BaselineVLMConfig
from src.document_understanding.vlm_inference import ImageInput, run_vlm_inference
from src.document_understanding.vlm_loader import LoadedVLM
from src.ocr.models import OCRPageResult, OCRPipelineResult

logger = logging.getLogger(__name__)

InferenceFn = Callable[..., dict[str, Any]]

SUPPORTED_TASKS = tuple(task.value for task in AnalysisTask)


def extract_ocr_text(ocr_result: Any) -> str | None:
    """Read plain text out of OCR output without modifying the OCR layer."""
    if ocr_result is None:
        return None
    if isinstance(ocr_result, str):
        return ocr_result or None
    if isinstance(ocr_result, OCRPageResult):
        return ocr_result.text or None
    if isinstance(ocr_result, OCRPipelineResult):
        text = "\n".join(page.text for page in ocr_result.pages if page.text)
        return text or None
    raise TypeError(
        "ocr_result must be a string, OCRPageResult, or OCRPipelineResult, "
        f"got {type(ocr_result).__name__}"
    )


class DocumentAnalyzer:
    """Run document understanding tasks against a document image."""

    def __init__(
        self,
        config: BaselineVLMConfig | None = None,
        *,
        loaded_model: LoadedVLM | None = None,
        inference_fn: InferenceFn = run_vlm_inference,
    ) -> None:
        self.config = config
        self.loaded_model = loaded_model
        self._inference_fn = inference_fn

    def analyze(
        self,
        image_path: ImageInput | None = None,
        task: AnalysisTask | str = AnalysisTask.CLASSIFICATION,
        *,
        image: ImageInput | None = None,
        question: str | None = None,
        ocr_result: Any = None,
        document_type: DocumentType | str | None = None,
    ) -> dict[str, Any]:
        """Analyze a document image for a single task.

        ``document_type`` narrows key-value extraction to that type's schema; it
        is a schema hint only and never becomes an extracted value.
        """
        resolved_task = AnalysisTask.from_string(task)
        source_image = image if image is not None else image_path
        if source_image is None:
            raise ValueError("An image_path or image is required")

        if resolved_task is AnalysisTask.QUESTION_ANSWERING and not (question or "").strip():
            raise ValueError("question is required for the question_answering task")

        ocr_text = extract_ocr_text(ocr_result)
        prompt = build_task_prompt(
            resolved_task,
            question=question,
            document_type=document_type,
            ocr_text=ocr_text,
        )

        inference = self._inference_fn(
            source_image,
            prompt,
            self.config,
            loaded_model=self.loaded_model,
        )
        raw_response = str(inference.get("response", ""))
        result = self._parse(resolved_task, raw_response, question, document_type)

        payload = result.to_dict()
        payload.update(
            {
                "prompt": prompt,
                "ocr_context_used": ocr_text is not None,
                "model_name": inference.get("model_name"),
                "device": inference.get("device"),
                "inference_time": inference.get("inference_time"),
                "memory": inference.get("memory", {}),
            }
        )
        logger.info(
            "Analyzed document: task=%s parse_success=%s time=%s",
            resolved_task.value,
            result.parse_success,
            inference.get("inference_time"),
        )
        return payload

    @staticmethod
    def _parse(
        task: AnalysisTask,
        raw_response: str,
        question: str | None,
        document_type: DocumentType | str | None,
    ) -> TaskResult:
        if task is AnalysisTask.CLASSIFICATION:
            return parse_classification(raw_response)
        if task is AnalysisTask.KEY_VALUE_EXTRACTION:
            return parse_key_value(raw_response, document_type)
        if task is AnalysisTask.TABLE_EXTRACTION:
            return parse_table(raw_response)
        if task is AnalysisTask.SUMMARIZATION:
            return parse_summary(raw_response)
        return parse_question_answer(raw_response, question or "")
