"""Abstract interface for vision-language models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image


class VLMModelError(RuntimeError):
    """Raised when VLM loading or inference fails."""


@dataclass
class VLMGenerationResult:
    """Raw generation output from a VLM."""

    text: str
    latency_seconds: float | None = None


class VLMModel(ABC):
    """Model abstraction so the pipeline can swap VLMs without code changes."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the configured model identifier."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether model weights are currently loaded."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processors."""

    @abstractmethod
    def generate(self, image: Image.Image | np.ndarray, prompt: str) -> VLMGenerationResult:
        """Run multimodal generation for a single image and text prompt."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
