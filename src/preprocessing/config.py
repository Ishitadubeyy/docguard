"""Configuration for the OCR preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configurable preprocessing steps applied before OCR."""

    grayscale: bool = True
    resize: bool = True
    max_dimension: int = 3000
    contrast_enhancement: bool = True
    denoise: bool = True
    threshold: bool = False
    threshold_method: str = "adaptive"
    deskew: bool = True
    deskew_min_angle: float = 0.5
