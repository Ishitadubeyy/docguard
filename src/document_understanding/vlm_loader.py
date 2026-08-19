"""VLM model loading utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np
from PIL import Image

from src.document_understanding.config import VLMConfig
from src.document_understanding.vlm_config import BaselineVLMConfig
from src.document_understanding.vlm_interface import VLMGenerationResult, VLMModel, VLMModelError

logger = logging.getLogger(__name__)


def get_missing_vlm_dependencies() -> list[str]:
    """Return missing Python packages required for Hugging Face VLM inference."""
    missing: list[str] = []
    if find_spec("torch") is None:
        missing.append("torch")
    if find_spec("transformers") is None:
        missing.append("transformers")
    return missing


def _to_pil_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        return Image.fromarray(array[:, :, :3]).convert("RGB")

    raise VLMModelError(f"Unsupported image shape for VLM input: {array.shape}")


def _resolve_dtype(dtype_name: str):
    import torch

    normalized = dtype_name.strip().lower()
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise VLMModelError(
            f"Unsupported VLM dtype '{dtype_name}'. Supported values: {', '.join(mapping)}"
        )
    return mapping[normalized]


class HuggingFaceVLMModel(VLMModel):
    """Hugging Face Transformers implementation for image-text-to-text VLMs."""

    def __init__(self, config: VLMConfig | None = None) -> None:
        self._config = config or VLMConfig.from_env()
        self._processor = None
        self._model = None

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.is_loaded:
            return

        missing = get_missing_vlm_dependencies()
        if missing:
            raise VLMModelError(
                "Missing dependencies required for VLM inference: "
                + ", ".join(missing)
                + ". Install with: pip install torch transformers"
            )

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading VLM model '%s' on device '%s'", self._config.model_id, self._config.device)

        try:
            self._processor = AutoProcessor.from_pretrained(
                self._config.model_id,
                trust_remote_code=self._config.trust_remote_code,
                local_files_only=self._config.local_files_only,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._config.model_id,
                dtype=_resolve_dtype(self._config.dtype),
                trust_remote_code=self._config.trust_remote_code,
                local_files_only=self._config.local_files_only,
            )
        except OSError as exc:
            raise VLMModelError(
                f"Failed to load VLM model '{self._config.model_id}'. "
                "Download the model first or set VLM_MODEL_ID to a locally available checkpoint. "
                f"Original error: {exc}"
            ) from exc
        except Exception as exc:
            raise VLMModelError(
                f"Failed to initialize VLM model '{self._config.model_id}': {exc}"
            ) from exc

        self._model.to(self._config.device)
        self._model.eval()
        logger.info("VLM model '%s' loaded successfully", self._config.model_id)

    def generate(self, image: Image.Image | np.ndarray, prompt: str) -> VLMGenerationResult:
        if not self.is_loaded:
            raise VLMModelError("VLM model is not loaded. Call load() before generate().")

        import time

        import torch

        pil_image = _to_pil_image(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(text=text, images=pil_image, return_tensors="pt")
            inputs = {key: value.to(self._config.device) for key, value in inputs.items()}

            start = time.perf_counter()
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    do_sample=False,
                )
            latency_seconds = time.perf_counter() - start

            decoded = self._processor.decode(output_ids[0], skip_special_tokens=True)
            response_text = _extract_assistant_response(decoded)
            logger.info(
                "VLM inference completed in %.2fs for model '%s'",
                latency_seconds,
                self._config.model_id,
            )
            return VLMGenerationResult(text=response_text, latency_seconds=latency_seconds)
        except Exception as exc:
            raise VLMModelError(f"VLM inference failed: {exc}") from exc

    def unload(self) -> None:
        if self._model is not None:
            logger.info("Unloading VLM model '%s'", self._config.model_id)
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _extract_assistant_response(decoded: str) -> str:
    """Return only the assistant portion of a chat-formatted decode."""
    marker = "Assistant:"
    if marker in decoded:
        return decoded.rsplit(marker, maxsplit=1)[-1].strip()
    return decoded.strip()


def create_vlm_model(config: VLMConfig | None = None) -> VLMModel:
    """Create the default Hugging Face VLM implementation."""
    return HuggingFaceVLMModel(config=config)


@dataclass
class LoadedVLM:
    """A loaded baseline VLM together with its processor and configuration."""

    model: Any
    processor: Any
    config: BaselineVLMConfig


_LOADED_MODELS: dict[tuple[str, str, str, str | None, bool], LoadedVLM] = {}


def _cache_key(config: BaselineVLMConfig) -> tuple[str, str, str, str | None, bool]:
    return (
        config.model_id,
        config.device,
        config.dtype,
        config.cache_dir,
        config.local_files_only,
    )


def load_baseline_vlm(config: BaselineVLMConfig | None = None) -> LoadedVLM:
    """Load (or reuse) a baseline VLM and its processor.

    Loading is cached per (model id, device, dtype, cache dir) so repeated
    inference calls do not re-download or re-instantiate weights. Loading is
    kept separate from inference; see ``vlm_inference.run_vlm_inference``.
    """
    config = config or BaselineVLMConfig.from_env()

    cached = _LOADED_MODELS.get(_cache_key(config))
    if cached is not None:
        logger.debug("Reusing cached VLM '%s' on '%s'", config.model_id, config.device)
        return cached

    missing = get_missing_vlm_dependencies()
    if missing:
        raise VLMModelError(
            "Missing dependencies required for VLM inference: "
            + ", ".join(missing)
            + ". Install with: pip install torch transformers"
        )

    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info(
        "Loading VLM '%s' (device=%s dtype=%s)", config.model_id, config.device, config.dtype
    )
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if config.cache_dir:
        load_kwargs["cache_dir"] = config.cache_dir

    try:
        processor = AutoProcessor.from_pretrained(config.model_id, **load_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_id,
            dtype=_resolve_dtype(config.dtype),
            **load_kwargs,
        )
    except OSError as exc:
        raise VLMModelError(
            f"Could not download or find VLM '{config.model_id}'. "
            "Check the model id, network access, or point VLM_CACHE_DIR at a local cache. "
            f"Original error: {exc}"
        ) from exc
    except ValueError as exc:
        raise VLMModelError(
            f"Model '{config.model_id}' is not a supported image-text-to-text model for the "
            f"installed Transformers version. Original error: {exc}"
        ) from exc
    except MemoryError as exc:
        raise VLMModelError(
            f"Ran out of memory while loading '{config.model_id}'. Use a smaller checkpoint "
            f"or a machine with more RAM. Original error: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as an actionable VLM error
        if _is_out_of_memory_error(exc):
            raise VLMModelError(
                f"Out of memory while loading '{config.model_id}' on '{config.device}'. "
                "Use a smaller model, reduce dtype precision on GPU, or free memory. "
                f"Original error: {exc}"
            ) from exc
        raise VLMModelError(f"Failed to load VLM '{config.model_id}': {exc}") from exc

    try:
        model.to(config.device)
    except Exception as exc:  # noqa: BLE001 - device placement failures must be actionable
        raise VLMModelError(
            f"Failed to move model '{config.model_id}' to device '{config.device}': {exc}"
        ) from exc
    model.eval()

    loaded = LoadedVLM(model=model, processor=processor, config=config)
    _LOADED_MODELS[_cache_key(config)] = loaded
    logger.info("Loaded VLM '%s' on '%s'", config.model_id, config.device)
    return loaded


def unload_baseline_vlms() -> None:
    """Drop all cached baseline models and release GPU memory if applicable."""
    _LOADED_MODELS.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _is_out_of_memory_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    message = str(exc).lower()
    return (
        name == "OutOfMemoryError"
        or "out of memory" in message
        or "cannot allocate memory" in message
    )
