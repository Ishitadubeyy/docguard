"""Modular prompt templates for document understanding tasks."""

from __future__ import annotations

import json
from typing import Any

from src.document_understanding.config import TaskType
from src.ocr.models import OCRBlock

DOCUMENT_UNDERSTANDING_SCHEMA = """{
  "document_type": "...",
  "fields": {
    "invoice_number": "...",
    "date": "...",
    "customer_name": "...",
    "company": "...",
    "total_amount": "..."
  },
  "tables": [],
  "summary": "..."
}"""

GENERIC_OUTPUT_SCHEMA = """{
  "document_type": "...",
  "fields": {"<field_name>": "<value>"},
  "tables": [],
  "summary": "..."
}"""

# Backward-compatible alias used by tests and external callers.
OUTPUT_SCHEMA = GENERIC_OUTPUT_SCHEMA

_BASE_INSTRUCTIONS = (
    "You are a document understanding assistant. "
    "The document image is the primary source of truth. "
    "Use the OCR text and bounding boxes below as supporting context only. "
    "Do not replace the image with OCR text. "
    "Correct obvious OCR errors only when the image clearly supports the correction. "
    "Do not invent missing information. "
    "Respond with a single JSON object only. "
    "Do not include Markdown, code fences, explanations, or any text outside the JSON object. "
    "Do not wrap the JSON in nested objects such as {\"text\": {...}}."
)

_JSON_OUTPUT_RULES = (
    "Output rules:\n"
    "- Return valid JSON only.\n"
    "- No Markdown.\n"
    "- No explanations before or after the JSON.\n"
    "- No duplicate nested \"text\" objects.\n"
    "- Use null for unknown scalar values; use an empty object for fields when no "
    "key-value pairs are visible.\n"
    "- Do not include a confidence score unless you have a calibrated one."
)

_TASK_INSTRUCTIONS: dict[TaskType, str] = {
    TaskType.DOCUMENT_UNDERSTANDING: (
        "Analyze the document image together with the OCR context. "
        "Extract the document type, invoice-related fields when present, any tables, "
        "and a concise summary. "
        "Populate fields only with values visible in the image or supported by OCR context. "
        "Leave individual field values as null when not found."
    ),
    TaskType.KEY_VALUE_EXTRACTION: (
        "Extract all visible key-value pairs from the document. "
        "Populate the fields object with normalized field names."
    ),
    TaskType.DOCUMENT_CLASSIFICATION: (
        "Classify the document type. Set document_type to the best label "
        "and include supporting evidence in summary."
    ),
    TaskType.TABLE_UNDERSTANDING: (
        "Identify and extract tables from the document. "
        "Represent each table as a list of row objects in tables."
    ),
    TaskType.OCR_CORRECTION: (
        "Optional OCR correction task: compare the OCR text against the document image "
        "and propose corrections supported by visual evidence. "
        "Put corrected key values in fields under names such as corrected_text or "
        "corrected_fields. Note major fixes in summary. "
        "Do not modify the original OCR layer; only report proposed corrections here."
    ),
}

_TASK_SCHEMAS: dict[TaskType, str] = {
    TaskType.DOCUMENT_UNDERSTANDING: DOCUMENT_UNDERSTANDING_SCHEMA,
}


def get_output_schema(task: TaskType) -> str:
    """Return the expected JSON schema string for a task."""
    return _TASK_SCHEMAS.get(task, GENERIC_OUTPUT_SCHEMA)


def format_ocr_blocks(blocks: list[OCRBlock] | None) -> str:
    """Serialize OCR blocks for inclusion in a prompt."""
    if not blocks:
        return "No OCR bounding boxes available."

    payload = [
        {
            "text": block.text,
            "bbox": block.bbox,
            "confidence": block.confidence,
        }
        for block in blocks
    ]
    return json.dumps(payload, indent=2)


def build_prompt(
    task: TaskType,
    ocr_text: str | None = None,
    ocr_blocks: list[OCRBlock] | None = None,
    extra_instructions: str | None = None,
) -> str:
    """Build a multimodal text prompt for the given task.

    The VLM receives the page image separately. This prompt supplies OCR context,
    task instructions, and the expected JSON schema.
    """
    task_instruction = _TASK_INSTRUCTIONS[task]
    schema = get_output_schema(task)
    ocr_section = ocr_text.strip() if ocr_text and ocr_text.strip() else "No OCR text available."
    blocks_section = format_ocr_blocks(ocr_blocks)

    sections = [
        _BASE_INSTRUCTIONS,
        f"Task: {task.value}",
        f"OCR text (supporting context, may contain errors):\n{ocr_section}",
        f"OCR blocks (supporting context):\n{blocks_section}",
        f"Task instructions: {task_instruction}",
        f"Expected JSON schema:\n{schema}",
        _JSON_OUTPUT_RULES,
    ]

    if extra_instructions:
        sections.append(f"Additional instructions:\n{extra_instructions.strip()}")

    sections.append("Respond with JSON only.")
    return "\n\n".join(sections)


def get_task_instruction(task: TaskType) -> str:
    """Return the task-specific instruction string."""
    return _TASK_INSTRUCTIONS[task]
