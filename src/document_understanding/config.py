"""Configuration for document understanding and VLM inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    """Supported document understanding tasks."""

    DOCUMENT_UNDERSTANDING = "document_understanding"
    KEY_VALUE_EXTRACTION = "key_value_extraction"
    DOCUMENT_CLASSIFICATION = "document_classification"
    TABLE_UNDERSTANDING = "table_understanding"
    OCR_CORRECTION = "ocr_correction"

    @classmethod
    def from_string(cls, value: str) -> TaskType:
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(task.value for task in cls)
            raise ValueError(f"Unsupported task '{value}'. Supported tasks: {supported}") from exc


# Smallest SmolVLM instruct checkpoint; suitable for CPU-only experimentation.
DEFAULT_VLM_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


@dataclass(frozen=True)
class VLMConfig:
    """Runtime configuration for VLM loading and inference."""

    model_id: str = DEFAULT_VLM_MODEL_ID
    device: str = "cpu"
    max_new_tokens: int = 512
    dtype: str = "float32"
    trust_remote_code: bool = True
    local_files_only: bool = False

    @classmethod
    def from_env(cls) -> VLMConfig:
        """Build configuration from environment variables."""
        return cls(
            model_id=os.getenv("VLM_MODEL_ID", DEFAULT_VLM_MODEL_ID),
            device=os.getenv("VLM_DEVICE", "cpu"),
            max_new_tokens=int(os.getenv("VLM_MAX_NEW_TOKENS", "512")),
            dtype=os.getenv("VLM_DTYPE", "float32"),
            trust_remote_code=os.getenv("VLM_TRUST_REMOTE_CODE", "true").lower()
            in {"1", "true", "yes"},
            local_files_only=os.getenv("VLM_LOCAL_FILES_ONLY", "false").lower()
            in {"1", "true", "yes"},
        )
