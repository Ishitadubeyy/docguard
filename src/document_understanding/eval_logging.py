"""Evaluation logging for VLM inference runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.document_understanding.models import DocumentUnderstandingResult


@dataclass
class InferenceLogRecord:
    """Structured log entry for a single VLM inference."""

    model_id: str
    task: str
    latency_seconds: float
    parse_success: bool
    schema_valid: bool
    schema_normalized: bool
    document_type: str | None
    number_of_extracted_fields: int
    ocr_text_length: int
    timestamp: str
    page_number: int | None = None
    parse_errors: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_inference_log(
    *,
    model_id: str,
    task: str,
    latency_seconds: float,
    result: DocumentUnderstandingResult,
    ocr_text_length: int,
    page_number: int | None = None,
    timestamp: datetime | None = None,
) -> InferenceLogRecord:
    """Build an inference log record from a parsed result."""
    event_time = timestamp or datetime.now(timezone.utc)
    non_null_fields = sum(1 for value in result.fields.values() if value is not None and value != "")

    return InferenceLogRecord(
        model_id=model_id,
        task=task,
        latency_seconds=latency_seconds,
        parse_success=result.parse_success,
        schema_valid=result.schema_valid,
        schema_normalized=result.schema_normalized,
        document_type=result.document_type,
        number_of_extracted_fields=non_null_fields,
        ocr_text_length=ocr_text_length,
        timestamp=event_time.isoformat(),
        page_number=page_number,
        parse_errors=list(result.parse_errors),
        schema_errors=list(result.schema_errors),
    )
