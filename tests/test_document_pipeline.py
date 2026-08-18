"""Tests for OCR + VLM document understanding pipeline integration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from src.document_understanding.config import TaskType, VLMConfig
from src.document_understanding.inference import run_inference
from src.document_understanding.models import Document, DocumentPage
from src.document_understanding.pipeline import build_document_from_ocr
from src.document_understanding.vlm_interface import VLMGenerationResult, VLMModel, VLMModelError
from src.ocr.models import OCRBlock, OCRPageResult, OCRPipelineResult

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

from tests.conftest import requires_ocr, requires_vlm


class MockVLMModel(VLMModel):
    """Deterministic VLM stub for unit tests."""

    def __init__(self, response_text: str, *, model_id: str = "mock/vlm") -> None:
        self._model_id = model_id
        self._response_text = response_text
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def generate(self, image, prompt: str) -> VLMGenerationResult:
        if image is None:
            raise VLMModelError("Image is required")
        if not prompt:
            raise VLMModelError("Prompt is required")
        return VLMGenerationResult(text=self._response_text, latency_seconds=0.01)

    def unload(self) -> None:
        self._loaded = False


@pytest.fixture
def sample_document(sample_array) -> Document:
    return Document(
        document_id="doc-123",
        pages=[
            DocumentPage(
                page_number=1,
                image=sample_array,
                ocr_text="Invoice 001",
                ocr_blocks=[OCRBlock(text="Invoice 001", bbox=[1, 2, 3, 4], confidence=0.9)],
            )
        ],
        metadata={"filename": "invoice.png"},
    )


def test_mock_vlm_model_interface(sample_document) -> None:
    model = MockVLMModel(
        response_text='{"document_type":"invoice","fields":{"number":"001"},"tables":[],"summary":"Invoice","confidence":null}'
    )
    results, logs = run_inference(model, sample_document, TaskType.DOCUMENT_UNDERSTANDING)

    assert model.is_loaded
    assert len(results) == 1
    assert len(logs) == 1
    assert results[0].document_type == "invoice"
    assert results[0].fields["number"] == "001"
    assert results[0].confidence is None
    assert logs[0].parse_success is True


def test_run_inference_requires_image(sample_document) -> None:
    sample_document.pages[0].image = None
    model = MockVLMModel(response_text="{}")

    with pytest.raises(ValueError, match="missing an image"):
        run_inference(model, sample_document, TaskType.DOCUMENT_UNDERSTANDING)


def test_run_inference_requires_pages() -> None:
    document = Document(document_id="empty", pages=[])
    model = MockVLMModel(response_text="{}")

    with pytest.raises(ValueError, match="no pages"):
        run_inference(model, document, TaskType.KEY_VALUE_EXTRACTION)


def test_build_document_from_ocr_combines_image_and_text(tmp_path: Path) -> None:
    image_path = tmp_path / "doc.png"
    Image.new("RGB", (100, 50), color="white").save(image_path)

    ocr_result = OCRPipelineResult(
        document_id="doc-abc",
        pages=[
            OCRPageResult(
                page_number=1,
                text="Hello",
                blocks=[OCRBlock(text="Hello", bbox=[0, 0, 10, 10], confidence=0.8)],
            )
        ],
    )

    document = build_document_from_ocr(image_path, ocr_result, metadata={"source": "test"})
    assert document.document_id == "doc-abc"
    assert document.pages[0].ocr_text == "Hello"
    assert isinstance(document.pages[0].image, np.ndarray)
    assert document.metadata["source"] == "test"


@requires_ocr
def test_pipeline_integration_with_mock_vlm(sample_image_path: Path, monkeypatch: MonkeyPatch) -> None:
    from src.document_understanding import pipeline as pipeline_module

    mock_response = (
        '{"document_type":"test_document","fields":{"text":"DocGuard OCR Test"},'
        '"tables":[],"summary":"Test page","confidence":null}'
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_vlm_model",
        lambda config=None: MockVLMModel(mock_response),
    )

    payload = pipeline_module.run_document_understanding_pipeline(
        sample_image_path,
        TaskType.DOCUMENT_UNDERSTANDING,
    )

    assert payload["document_id"]
    assert payload["task"] == "document_understanding"
    assert payload["ocr"]["pages"]
    assert payload["results"][0]["document_type"] == "test_document"
    assert payload["inference_logs"][0]["parse_success"] is True


def test_vlm_config_local_files_only_blocks_missing_model() -> None:
    config = VLMConfig(model_id="definitely/missing-model", local_files_only=True)
    from src.document_understanding.vlm_loader import HuggingFaceVLMModel

    model = HuggingFaceVLMModel(config=config)
    with pytest.raises(VLMModelError, match="Failed to load VLM model"):
        model.load()


@pytest.mark.integration
@requires_vlm
def test_real_vlm_inference_when_model_available() -> None:
    """Integration test: requires a locally cached VLM checkpoint."""
    from src.document_understanding.vlm_loader import HuggingFaceVLMModel

    config = VLMConfig.from_env()
    model = HuggingFaceVLMModel(config=config)
    try:
        model.load()
    except Exception as exc:
        pytest.skip(f"VLM model unavailable for integration test: {exc}")

    image = Image.new("RGB", (200, 100), color="white")
    result = model.generate(image, "Return JSON with document_type, fields, tables, summary, confidence.")
    assert result.text
    model.unload()
