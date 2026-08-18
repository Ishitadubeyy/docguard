"""Metric helpers for document understanding evaluation."""

from __future__ import annotations

from typing import Any

from src.document_understanding.models import DocumentUnderstandingResult
from src.document_understanding.parser import (
    _is_schema_valid,
    normalize_schema_structure,
    parse_structured_output,
    validate_extraction_result,
)
from evaluation.benchmarks.document_understanding.schema import BenchmarkSample, EvaluationResult

_PREDICTION_META_KEYS = frozenset(
    {
        "raw_response",
        "page_number",
        "parse_errors",
        "schema_errors",
        "parse_success",
        "schema_valid",
        "schema_normalized",
    }
)


def field_extraction_accuracy(
    predicted_fields: dict[str, Any],
    expected_fields: dict[str, Any],
) -> float | None:
    """Compute exact-match field accuracy between predicted and expected fields."""
    if not expected_fields:
        return None

    matches = sum(
        1 for key, expected_value in expected_fields.items()
        if str(predicted_fields.get(key, "")).strip() == str(expected_value).strip()
    )
    return matches / len(expected_fields)


def document_classification_accuracy(
    predicted_type: str | None,
    expected_type: str | None,
) -> float | None:
    """Return 1.0 for an exact document type match, else 0.0."""
    if expected_type is None:
        return None
    if predicted_type is None:
        return 0.0
    return 1.0 if predicted_type.strip().lower() == expected_type.strip().lower() else 0.0


def ocr_correction_accuracy(predicted_text: str | None, expected_text: str | None) -> float | None:
    """Compute normalized exact-match accuracy for OCR correction."""
    if expected_text is None:
        return None
    if predicted_text is None:
        return 0.0
    return 1.0 if predicted_text.strip() == expected_text.strip() else 0.0


def json_parse_success_score(raw_output: str) -> float:
    """Return 1.0 when model output parses as valid JSON, else 0.0."""
    result = parse_structured_output(raw_output)
    return 1.0 if result.parse_success else 0.0


def json_validity_score(raw_output: str) -> float:
    """Backward-compatible alias for JSON parse success."""
    return json_parse_success_score(raw_output)


def schema_compliance_score(raw_output: str) -> float:
    """Return 1.0 when model output already matched the expected schema, else 0.0."""
    result = parse_structured_output(raw_output)
    return 1.0 if result.schema_valid else 0.0


def schema_normalization_rate(raw_output: str) -> float:
    """Return 1.0 when output required structural schema normalization, else 0.0."""
    result = parse_structured_output(raw_output)
    return 1.0 if result.schema_normalized else 0.0


def field_extraction_count_from_result(result: DocumentUnderstandingResult) -> float | None:
    """Return the number of non-empty extracted fields from a parsed result."""
    if not result.parse_success:
        return None
    return float(
        sum(1 for value in result.fields.values() if value is not None and value != "")
    )


def evaluate_sample(
    *,
    model: str,
    dataset: str,
    task: str,
    sample: BenchmarkSample,
    prediction: dict[str, Any] | str,
    latency_seconds: float | None = None,
    memory_mb: float | None = None,
) -> list[EvaluationResult]:
    """Evaluate one sample and return metric results without inventing scores."""
    if isinstance(prediction, str):
        parsed = parse_structured_output(prediction)
    else:
        payload = {key: value for key, value in prediction.items() if key not in _PREDICTION_META_KEYS}
        schema_valid = _is_schema_valid(payload)
        normalized_payload, schema_normalized, normalization_errors = normalize_schema_structure(payload)
        parsed = validate_extraction_result(
            normalized_payload,
            parse_success=True,
            schema_valid=schema_valid,
            schema_normalized=schema_normalized,
            initial_schema_errors=normalization_errors,
        )

    predicted_fields = parsed.fields
    predicted_type = parsed.document_type
    predicted_summary = parsed.summary

    metrics: list[tuple[str, float | None]] = [
        ("field_extraction_accuracy", field_extraction_accuracy(predicted_fields, sample.fields)),
        (
            "document_classification_accuracy",
            document_classification_accuracy(predicted_type, sample.document_type),
        ),
        (
            "ocr_correction_accuracy",
            ocr_correction_accuracy(predicted_summary, sample.corrected_text),
        ),
        ("json_parse_success", 1.0 if parsed.parse_success else 0.0),
        ("schema_compliance", 1.0 if parsed.schema_valid else 0.0),
        ("schema_normalization_rate", 1.0 if parsed.schema_normalized else 0.0),
        ("field_extraction_count", field_extraction_count_from_result(parsed)),
    ]

    return [
        EvaluationResult(
            model=model,
            dataset=dataset,
            task=task,
            metric=metric_name,
            score=score,
            latency_seconds=latency_seconds,
            memory_mb=memory_mb,
            metadata={"sample_id": sample.sample_id},
        )
        for metric_name, score in metrics
        if score is not None
    ]
