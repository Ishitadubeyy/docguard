"""Parse and validate structured VLM output."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.document_understanding.models import DocumentUnderstandingResult

logger = logging.getLogger(__name__)

_REQUIRED_TOP_LEVEL_KEYS = ("document_type", "fields", "tables", "summary")
_KNOWN_FLAT_FIELD_NAMES = (
    "invoice_number",
    "date",
    "customer_name",
    "company",
    "total_amount",
)
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_json_fences(text: str) -> list[str]:
    """Return inner text from markdown JSON code fences."""
    return [match.group(1).strip() for match in _JSON_FENCE_PATTERN.finditer(text) if match.group(1).strip()]


def _iter_json_object_candidates(text: str) -> list[str]:
    """Yield candidate JSON object substrings using brace matching."""
    candidates: list[str] = []
    length = len(text)
    index = 0

    while index < length:
        if text[index] != "{":
            index += 1
            continue

        depth = 0
        in_string = False
        escape = False

        for end in range(index, length):
            char = text[end]

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[index : end + 1].strip())
                    index = end + 1
                    break
        else:
            index += 1

    return candidates


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def iter_json_candidates(raw_text: str) -> list[str]:
    """Collect candidate JSON substrings from model output."""
    text = raw_text.strip()
    if not text:
        return []

    candidates: list[str] = []
    candidates.extend(_strip_json_fences(text))

    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)

    candidates.extend(_iter_json_object_candidates(text))
    return _dedupe_preserve_order([candidate for candidate in candidates if candidate])


def extract_json_text(raw_text: str) -> str:
    """Extract a JSON object string from free-form model output."""
    candidates = iter_json_candidates(raw_text)
    if not candidates:
        raise ValueError("No JSON object found in model output")

    errors: list[str] = []
    for candidate in candidates:
        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        return candidate

    detail = errors[0] if errors else "unknown parse error"
    raise ValueError(f"Malformed JSON in model output: {detail}")


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """Parse JSON from model output, raising ValueError on failure."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Model output is empty")

    json_text = extract_json_text(raw_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in model output: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Model JSON output must be an object")

    return payload


def _unwrap_nested_text_object(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Detect and unwrap a single nested {\"text\": {...}} wrapper."""
    errors: list[str] = []
    if set(payload) == {"text"} and isinstance(payload.get("text"), dict):
        errors.append('Unwrapped unexpected nested "text" object')
        return payload["text"], errors
    return payload, errors


def _is_schema_valid(payload: dict[str, Any]) -> bool:
    """Return True when the payload already matches the expected nested schema."""
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload:
            return False

    for key in _KNOWN_FLAT_FIELD_NAMES:
        if key in payload:
            return False

    document_type = payload.get("document_type")
    if document_type is not None and not isinstance(document_type, str):
        return False

    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return False
    if any(not isinstance(key, str) for key in fields):
        return False

    tables = payload.get("tables")
    if not isinstance(tables, list):
        return False

    summary = payload.get("summary")
    if summary is not None and not isinstance(summary, str):
        return False

    confidence = payload.get("confidence")
    if confidence is not None and _normalize_confidence(confidence) is None:
        return False

    unexpected_keys = set(payload) - set(_REQUIRED_TOP_LEVEL_KEYS) - {"confidence"}
    return not unexpected_keys


def normalize_schema_structure(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool, list[str]]:
    """Move known flat document fields into the nested fields object.

    This is an evaluation/compatibility layer. It does not invent values and
    records schema violations for each relocated top-level field.
    """
    schema_errors: list[str] = []
    normalized = dict(payload)
    existing_fields = normalized.get("fields")
    fields: dict[str, Any] = dict(existing_fields) if isinstance(existing_fields, dict) else {}
    schema_normalized = False

    for key in _KNOWN_FLAT_FIELD_NAMES:
        if key not in normalized:
            continue
        schema_errors.append(f"unexpected top-level field: {key}")
        if key not in fields:
            fields[key] = normalized.pop(key)
        else:
            normalized.pop(key)
        schema_normalized = True

    if schema_normalized:
        normalized["fields"] = fields

    return normalized, schema_normalized, schema_errors


def _normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def validate_extraction_result(
    payload: dict[str, Any],
    *,
    raw_response: str | None = None,
    page_number: int | None = None,
    parse_success: bool = True,
    schema_valid: bool = False,
    schema_normalized: bool = False,
    initial_schema_errors: list[str] | None = None,
) -> DocumentUnderstandingResult:
    """Validate and normalize a structured extraction payload."""
    schema_errors: list[str] = list(initial_schema_errors or [])

    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload:
            schema_errors.append(f"missing required field: {key}")

    document_type = payload.get("document_type")
    if document_type is not None and not isinstance(document_type, str):
        schema_errors.append("document_type must be a string or null")
        document_type = None

    fields = payload.get("fields", {})
    if "fields" not in payload:
        fields = {}
    elif not isinstance(fields, dict):
        schema_errors.append("fields must be an object")
        fields = {}
    elif any(not isinstance(key, str) for key in fields):
        schema_errors.append("fields keys must be strings")

    tables = payload.get("tables", [])
    if "tables" not in payload:
        tables = []
    elif not isinstance(tables, list):
        schema_errors.append("tables must be a list")
        tables = []

    summary = payload.get("summary")
    if "summary" not in payload:
        summary = None
    elif summary is not None and not isinstance(summary, str):
        schema_errors.append("summary must be a string or null")
        summary = None

    confidence = payload.get("confidence")
    normalized_confidence = _normalize_confidence(confidence)
    if confidence is not None and normalized_confidence is None:
        schema_errors.append("confidence must be a number or null")

    unexpected_keys = sorted(set(payload) - set(_REQUIRED_TOP_LEVEL_KEYS) - {"confidence"})
    for key in unexpected_keys:
        schema_errors.append(f"unexpected top-level field: {key}")

    return DocumentUnderstandingResult(
        document_type=document_type,
        fields=fields,
        tables=tables,
        summary=summary,
        confidence=normalized_confidence,
        raw_response=raw_response,
        page_number=page_number,
        parse_errors=[],
        schema_errors=schema_errors,
        parse_success=parse_success,
        schema_valid=schema_valid,
        schema_normalized=schema_normalized,
    )


def parse_structured_output(
    raw_text: str,
    *,
    page_number: int | None = None,
) -> DocumentUnderstandingResult:
    """Parse and validate structured JSON from raw VLM text output."""
    if not raw_text or not raw_text.strip():
        return DocumentUnderstandingResult(
            raw_response=raw_text,
            page_number=page_number,
            parse_errors=["Model output is empty"],
            parse_success=False,
            schema_valid=False,
            schema_normalized=False,
        )

    try:
        payload = parse_json_response(raw_text)
    except ValueError as exc:
        logger.warning("Failed to parse structured VLM output: %s", exc)
        return DocumentUnderstandingResult(
            raw_response=raw_text,
            page_number=page_number,
            parse_errors=[str(exc)],
            parse_success=False,
            schema_valid=False,
            schema_normalized=False,
        )

    payload, unwrap_errors = _unwrap_nested_text_object(payload)
    schema_valid = _is_schema_valid(payload)
    normalized_payload, schema_normalized, normalization_errors = normalize_schema_structure(payload)

    return validate_extraction_result(
        normalized_payload,
        raw_response=raw_text,
        page_number=page_number,
        parse_success=True,
        schema_valid=schema_valid,
        schema_normalized=schema_normalized,
        initial_schema_errors=[*unwrap_errors, *normalization_errors],
    )
