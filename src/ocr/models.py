"""Structured OCR result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OCRBlock:
    """A text block detected on a page."""

    text: str
    bbox: list[int]
    confidence: float


@dataclass
class OCRPageResult:
    """OCR output for a single page."""

    page_number: int
    text: str
    blocks: list[OCRBlock] = field(default_factory=list)


@dataclass
class OCRPipelineResult:
    """Structured OCR output for a full document."""

    document_id: str
    pages: list[OCRPageResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return asdict(self)
