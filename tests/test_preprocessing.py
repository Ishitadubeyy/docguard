"""Tests for document ingestion and image preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ingestion.exceptions import (
    DocumentNotFoundError,
    UnsupportedDocumentTypeError,
)
from src.ingestion.loader import ingest_document
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipeline import preprocess_for_ocr
from tests.conftest import requires_opencv


def test_ingest_valid_image(sample_image_path: Path) -> None:
    metadata = ingest_document(sample_image_path)

    assert metadata.filename == "sample.png"
    assert metadata.file_type == "png"
    assert metadata.file_size_bytes > 0
    assert metadata.page_count == 1
    assert len(metadata.document_id) == 36


def test_ingest_invalid_path() -> None:
    with pytest.raises(DocumentNotFoundError):
        ingest_document("this/path/does/not/exist.png")


def test_ingest_unsupported_file_type(tmp_path: Path) -> None:
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("plain text", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        ingest_document(unsupported)


@requires_opencv
def test_preprocessing_output_shape_and_dtype(sample_array: np.ndarray) -> None:
    config = PreprocessingConfig(
        grayscale=True,
        resize=False,
        contrast_enhancement=False,
        denoise=False,
        threshold=False,
        deskew=False,
    )
    result = preprocess_for_ocr(sample_array, config)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.dtype == np.uint8
    assert result.shape[0] > 0 and result.shape[1] > 0


@requires_opencv
def test_preprocessing_respects_disabled_steps(sample_array: np.ndarray) -> None:
    config = PreprocessingConfig(
        grayscale=False,
        resize=False,
        contrast_enhancement=False,
        denoise=False,
        threshold=False,
        deskew=False,
    )
    result = preprocess_for_ocr(sample_array, config)

    assert result.ndim == 3
    assert result.shape[2] == 3
