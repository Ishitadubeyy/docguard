"""Shared pytest fixtures and dependency helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def has_opencv() -> bool:
    from importlib.util import find_spec

    return find_spec("cv2") is not None


def has_ocr_stack() -> bool:
    import shutil
    from importlib.util import find_spec

    return find_spec("pytesseract") is not None and shutil.which("tesseract") is not None


requires_opencv = pytest.mark.skipif(
    not has_opencv(),
    reason="opencv-python is not installed",
)

requires_ocr = pytest.mark.skipif(
    not has_ocr_stack(),
    reason="pytesseract and/or the Tesseract binary are not installed",
)


def has_vlm_stack() -> bool:
    from importlib.util import find_spec

    return find_spec("torch") is not None and find_spec("transformers") is not None


requires_vlm = pytest.mark.skipif(
    not has_vlm_stack(),
    reason="torch and/or transformers are not installed",
)


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Create a simple PNG with text for ingestion and OCR tests."""
    image = Image.new("RGB", (320, 120), color="white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 40), "DocGuard OCR Test", fill="black", font=font)

    path = tmp_path / "sample.png"
    image.save(path, format="PNG")
    return path


@pytest.fixture
def sample_array() -> np.ndarray:
    """Return a small RGB array suitable for preprocessing tests."""
    image = Image.new("RGB", (200, 100), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 180, 80), outline=(20, 20, 20), width=2)
    return np.array(image)
