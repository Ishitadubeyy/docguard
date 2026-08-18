"""Unit tests for the DocumentAnalyzer task layer (no model downloads).

The analyzer is exercised with a stub inference function so prompt selection,
task validation, and parsing are tested without asserting model quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.document_understanding.document_analyzer import DocumentAnalyzer, extract_ocr_text
from src.document_understanding.extraction_schema import AnalysisTask
from src.document_understanding.vlm_inference import InvalidImageError
from src.ocr.models import OCRPageResult, OCRPipelineResult


class StubInference:
    """Record prompts and return a canned raw model response."""

    def __init__(self, response: str = "invoice") -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, image, prompt, config=None, *, loaded_model=None):
        self.prompts.append(prompt)
        return {
            "response": self.response,
            "model_name": "stub-model",
            "device": "cpu",
            "inference_time": 0.5,
            "memory": {"process_rss_mb": 1.0},
        }


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "doc.png"
    Image.new("RGB", (64, 64), "white").save(path)
    return path


class TestTaskValidation:
    def test_all_supported_tasks_run(self, image_path):
        analyzer = DocumentAnalyzer(inference_fn=StubInference("invoice"))
        for task in AnalysisTask:
            payload = analyzer.analyze(
                image_path=image_path,
                task=task,
                question="What is the invoice number?",
            )
            assert payload["task"] == task.value

    def test_unsupported_task_rejected(self, image_path):
        analyzer = DocumentAnalyzer(inference_fn=StubInference())
        with pytest.raises(ValueError, match="Unsupported task"):
            analyzer.analyze(image_path=image_path, task="forgery_detection")

    def test_missing_image_argument_rejected(self):
        analyzer = DocumentAnalyzer(inference_fn=StubInference())
        with pytest.raises(ValueError, match="image_path or image is required"):
            analyzer.analyze(task="classification")

    def test_missing_question_rejected(self, image_path):
        analyzer = DocumentAnalyzer(inference_fn=StubInference())
        with pytest.raises(ValueError, match="question is required"):
            analyzer.analyze(image_path=image_path, task="question_answering")


class TestResponseParsing:
    def test_classification_result(self, image_path):
        analyzer = DocumentAnalyzer(inference_fn=StubInference("This is a bank statement."))
        payload = analyzer.analyze(image_path=image_path, task="classification")
        assert payload["document_type"] == "bank_statement"
        assert payload["raw_response"] == "This is a bank statement."
        assert payload["confidence"] is None

    def test_key_value_result_uses_document_schema(self, image_path):
        stub = StubInference('{"fields": {"invoice_number": "INV-001"}}')
        payload = DocumentAnalyzer(inference_fn=stub).analyze(
            image_path=image_path,
            task="key_value_extraction",
            document_type="invoice",
        )
        assert payload["fields"]["invoice_number"] == "INV-001"
        assert payload["fields"]["customer"] is None
        assert "invoice_number" in stub.prompts[0]

    def test_table_result(self, image_path):
        stub = StubInference('{"columns": ["Date"], "rows": [["2026-02-03"]]}')
        payload = DocumentAnalyzer(inference_fn=stub).analyze(
            image_path=image_path, task="table_extraction"
        )
        assert payload["table"]["rows"] == [["2026-02-03"]]

    def test_summarization_result(self, image_path):
        stub = StubInference("An invoice for one laptop.")
        payload = DocumentAnalyzer(inference_fn=stub).analyze(
            image_path=image_path, task="summarization"
        )
        assert payload["summary"] == "An invoice for one laptop"

    def test_question_answering_result(self, image_path):
        stub = StubInference("INV-001")
        payload = DocumentAnalyzer(inference_fn=stub).analyze(
            image_path=image_path,
            task="question_answering",
            question="What is the invoice number?",
        )
        assert payload["question"] == "What is the invoice number?"
        assert payload["answer"] == "INV-001"

    def test_malformed_output_is_reported(self, image_path):
        stub = StubInference("???")
        payload = DocumentAnalyzer(inference_fn=stub).analyze(
            image_path=image_path, task="table_extraction"
        )
        assert payload["parse_success"] is False
        assert payload["parse_errors"]
        assert payload["raw_response"] == "???"

    def test_result_carries_inference_metadata(self, image_path):
        payload = DocumentAnalyzer(inference_fn=StubInference()).analyze(
            image_path=image_path, task="classification"
        )
        assert payload["model_name"] == "stub-model"
        assert payload["device"] == "cpu"
        assert payload["inference_time"] == 0.5
        assert payload["memory"] == {"process_rss_mb": 1.0}


class TestImageHandling:
    """Image validation happens before any model load, so no weights are needed."""

    def test_missing_image_file_raises(self, tmp_path):
        with pytest.raises(InvalidImageError, match="Image file not found"):
            DocumentAnalyzer().analyze(
                image_path=tmp_path / "missing.png",
                task="classification",
            )

    def test_corrupt_image_raises(self, tmp_path):
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        with pytest.raises(InvalidImageError, match="Could not read image"):
            DocumentAnalyzer().analyze(image_path=corrupt, task="classification")


class TestOCRContext:
    def test_ocr_text_from_supported_types(self):
        assert extract_ocr_text("INVOICE") == "INVOICE"
        assert extract_ocr_text(OCRPageResult(page_number=1, text="INV-001")) == "INV-001"
        pipeline = OCRPipelineResult(
            document_id="doc",
            pages=[
                OCRPageResult(page_number=1, text="page one"),
                OCRPageResult(page_number=2, text="page two"),
            ],
        )
        assert extract_ocr_text(pipeline) == "page one\npage two"
        assert extract_ocr_text(None) is None

    def test_unsupported_ocr_type_rejected(self):
        with pytest.raises(TypeError, match="ocr_result must be"):
            extract_ocr_text(42)

    def test_ocr_context_is_optional_and_flagged(self, image_path):
        stub = StubInference()
        analyzer = DocumentAnalyzer(inference_fn=stub)

        without = analyzer.analyze(image_path=image_path, task="classification")
        with_ocr = analyzer.analyze(
            image_path=image_path,
            task="classification",
            ocr_result="INVOICE INV-001",
        )

        assert without["ocr_context_used"] is False
        assert with_ocr["ocr_context_used"] is True
        assert "INV-001" in stub.prompts[1]
