"""Tests for OCR pipeline and structured output."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ocr.models import OCRBlock, OCRPageResult, OCRPipelineResult
from src.ocr.pipeline import process_document
from tests.conftest import requires_ocr


def test_ocr_result_schema() -> None:
    result = OCRPipelineResult(
        document_id="test-doc-id",
        pages=[
            OCRPageResult(
                page_number=1,
                text="Hello",
                blocks=[
                    OCRBlock(text="Hello", bbox=[10, 20, 80, 40], confidence=0.95),
                ],
            )
        ],
    )
    payload = result.to_dict()

    assert payload["document_id"] == "test-doc-id"
    assert len(payload["pages"]) == 1
    page = payload["pages"][0]
    assert page["page_number"] == 1
    assert page["text"] == "Hello"
    assert len(page["blocks"]) == 1
    block = page["blocks"][0]
    assert block["text"] == "Hello"
    assert block["bbox"] == [10, 20, 80, 40]
    assert block["confidence"] == pytest.approx(0.95)


@requires_ocr
def test_process_document_output_schema(sample_image_path: Path) -> None:
    result = process_document(sample_image_path)
    payload = result.to_dict()

    assert "document_id" in payload
    assert isinstance(payload["document_id"], str)
    assert payload["document_id"]

    assert "pages" in payload
    assert isinstance(payload["pages"], list)
    assert len(payload["pages"]) == 1

    page = payload["pages"][0]
    assert page["page_number"] == 1
    assert isinstance(page["text"], str)
    assert isinstance(page["blocks"], list)

    if page["blocks"]:
        block = page["blocks"][0]
        assert isinstance(block["text"], str)
        assert isinstance(block["bbox"], list)
        assert len(block["bbox"]) == 4
        assert isinstance(block["confidence"], float)
        assert 0.0 <= block["confidence"] <= 1.0

    assert page["text"].strip()
