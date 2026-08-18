"""Data models for ingested documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata captured during document ingestion."""

    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    source_path: str


@dataclass
class DocumentPage:
    """A single page extracted from a document."""

    page_number: int
    image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
