"""Configuration for the baseline VLM (Phase 2 smoke-test layer).

This configuration is intentionally independent of the OCR pipeline: it only
describes how to load and run a vision-language model on a single image and a
single free-form prompt.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

# Smallest SmolVLM instruct checkpoint (~256M params); the model id is
# configurable and must not be hard-coded outside this module.
DEFAULT_BASELINE_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def detect_device(requested: str | None = None) -> str:
    """Return the device to run on, auto-detecting CUDA when available."""
    if requested and requested.strip().lower() != "auto":
        return requested.strip().lower()

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def default_dtype_for(device: str) -> str:
    """Return a dtype that is safe for the given device.

    CPU kernels for float16 are slow or unimplemented for many operators, so
    float32 is used on CPU and float16 on CUDA.
    """
    return "float16" if device.startswith("cuda") else "float32"


@dataclass(frozen=True)
class BaselineVLMConfig:
    """Runtime configuration for baseline VLM loading and generation."""

    model_id: str = DEFAULT_BASELINE_MODEL_ID
    device: str = field(default_factory=lambda: detect_device("auto"))
    dtype: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.0
    cache_dir: str | None = None
    trust_remote_code: bool = True
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")

        object.__setattr__(self, "device", detect_device(self.device))
        if not self.dtype:
            object.__setattr__(self, "dtype", default_dtype_for(self.device))
        object.__setattr__(self, "dtype", self.dtype.strip().lower())

        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}'. Supported values: {', '.join(SUPPORTED_DTYPES)}"
            )

    @property
    def do_sample(self) -> bool:
        """Greedy decoding when temperature is zero."""
        return self.temperature > 0

    @classmethod
    def from_env(cls) -> BaselineVLMConfig:
        """Build configuration from environment variables."""
        return cls(
            model_id=os.getenv("VLM_MODEL_ID", DEFAULT_BASELINE_MODEL_ID),
            device=os.getenv("VLM_DEVICE", "auto"),
            dtype=os.getenv("VLM_DTYPE", ""),
            max_new_tokens=int(os.getenv("VLM_MAX_NEW_TOKENS", "256")),
            temperature=float(os.getenv("VLM_TEMPERATURE", "0.0")),
            cache_dir=os.getenv("VLM_CACHE_DIR") or None,
            trust_remote_code=_env_flag("VLM_TRUST_REMOTE_CODE", "true"),
            local_files_only=_env_flag("VLM_LOCAL_FILES_ONLY", "false"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
