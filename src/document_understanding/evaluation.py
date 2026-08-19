"""Per-task evaluation metrics for the document understanding layer.

Metrics are deliberately reported per task and never combined into a single
score. Summarization has no numerical score in this phase.
"""

from __future__ import annotations

import re
from typing import Any

_PUNCTUATION = re.compile(r"[^\w./-]+")


def normalize_for_match(value: Any) -> str | None:
    """Case-fold and strip formatting so values can be compared exactly."""
    if value is None:
        return None
    text = _PUNCTUATION.sub(" ", str(value).strip().lower())
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("./- ")
    return text or None


def exact_match(predicted: Any, expected: Any) -> bool:
    """Exact match after normalization; ``None`` only matches ``None``."""
    return normalize_for_match(predicted) == normalize_for_match(expected)


def classification_accuracy(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    """Accuracy over (predicted, expected) document type pairs."""
    total = len(pairs)
    correct = sum(1 for predicted, expected in pairs if exact_match(predicted, expected))
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
    }


def field_accuracy(
    predicted_fields: dict[str, Any],
    expected_fields: dict[str, Any],
) -> dict[str, Any]:
    """Field-level exact accuracy against ground truth."""
    per_field = {
        name: exact_match(predicted_fields.get(name), expected)
        for name, expected in expected_fields.items()
    }
    total = len(per_field)
    correct = sum(1 for is_correct in per_field.values() if is_correct)
    return {
        "total_fields": total,
        "correct_fields": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "per_field": per_field,
    }


def table_accuracy(
    predicted_table: dict[str, Any] | None,
    expected_rows: list[list[Any]],
) -> dict[str, Any]:
    """Row and cell accuracy for extracted tables.

    A predicted row counts as correct only when every cell matches the expected
    row at the same index.
    """
    predicted_rows = list((predicted_table or {}).get("rows") or [])
    expected_count = len(expected_rows)

    correct_rows = 0
    correct_cells = 0
    total_cells = sum(len(row) for row in expected_rows)

    for index, expected_row in enumerate(expected_rows):
        predicted_row = predicted_rows[index] if index < len(predicted_rows) else []
        matches = [
            exact_match(predicted_row[cell] if cell < len(predicted_row) else None, expected)
            for cell, expected in enumerate(expected_row)
        ]
        correct_cells += sum(1 for match in matches if match)
        if matches and all(matches):
            correct_rows += 1

    return {
        "expected_rows": expected_count,
        "predicted_rows": len(predicted_rows),
        "correct_rows": correct_rows,
        "row_accuracy": round(correct_rows / expected_count, 4) if expected_count else None,
        "total_cells": total_cells,
        "correct_cells": correct_cells,
        "cell_accuracy": round(correct_cells / total_cells, 4) if total_cells else None,
    }


def qa_exact_match(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    """Exact match over (predicted answer, expected answer) pairs."""
    total = len(pairs)
    correct = sum(1 for predicted, expected in pairs if exact_match(predicted, expected))
    return {
        "total": total,
        "correct": correct,
        "exact_match": round(correct / total, 4) if total else None,
    }
