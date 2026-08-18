"""Document representation and structured understanding output models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.ocr.models import OCRBlock


@dataclass
class DocumentPage:
    """A document page with image and optional OCR context."""

    page_number: int
    image: np.ndarray
    ocr_text: str | None = None
    ocr_blocks: list[OCRBlock] = field(default_factory=list)


@dataclass
class Document:
    """Multimodal document passed to the VLM pipeline."""

    document_id: str
    pages: list[DocumentPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentUnderstandingResult:
    """Validated structured output from document understanding."""

    document_type: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    tables: list[Any] = field(default_factory=list)
    summary: str | None = None
    confidence: float | None = None
    raw_response: str | None = None
    page_number: int | None = None
    parse_errors: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    parse_success: bool = False
    schema_valid: bool = False
    schema_normalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)
