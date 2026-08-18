"""Tests for document understanding prompt generation."""

from __future__ import annotations

import json

from src.document_understanding.config import TaskType
from src.document_understanding.prompts import (
    DOCUMENT_UNDERSTANDING_SCHEMA,
    build_prompt,
    format_ocr_blocks,
    get_task_instruction,
)
from src.ocr.models import OCRBlock


def test_build_prompt_includes_task_and_schema() -> None:
    prompt = build_prompt(
        TaskType.DOCUMENT_UNDERSTANDING,
        ocr_text="Invoice 123",
        ocr_blocks=[OCRBlock(text="Invoice 123", bbox=[1, 2, 3, 4], confidence=0.9)],
    )

    assert "document_understanding" in prompt
    assert '"invoice_number"' in prompt
    assert "Invoice 123" in prompt
    assert "Respond with JSON only." in prompt
    assert "No Markdown" in prompt
    assert "Do not invent missing information" in prompt


def test_build_prompt_includes_ocr_context_before_task_instructions() -> None:
    prompt = build_prompt(
        TaskType.DOCUMENT_UNDERSTANDING,
        ocr_text="Al Document Processing",
    )
    ocr_index = prompt.index("OCR text (supporting context")
    task_index = prompt.index("Task instructions:")
    schema_index = prompt.index("Expected JSON schema:")
    assert ocr_index < task_index < schema_index
    assert "Al Document Processing" in prompt
    assert DOCUMENT_UNDERSTANDING_SCHEMA in prompt


def test_build_prompt_handles_missing_ocr_data() -> None:
    prompt = build_prompt(TaskType.OCR_CORRECTION, ocr_text=None, ocr_blocks=None)

    assert "No OCR text available." in prompt
    assert "No OCR bounding boxes available." in prompt
    assert "ocr_correction" in prompt
    assert "Optional OCR correction task" in prompt


def test_format_ocr_blocks_serializes_bbox_and_confidence() -> None:
    blocks = [OCRBlock(text="Line", bbox=[0, 0, 10, 10], confidence=0.75)]
    payload = json.loads(format_ocr_blocks(blocks))

    assert payload[0]["text"] == "Line"
    assert payload[0]["bbox"] == [0, 0, 10, 10]
    assert payload[0]["confidence"] == 0.75


def test_all_tasks_have_instructions() -> None:
    for task in TaskType:
        instruction = get_task_instruction(task)
        assert instruction
        prompt = build_prompt(task, ocr_text="sample")
        assert instruction in prompt
