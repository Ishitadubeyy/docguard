"""Baseline VLM inference: one image plus one prompt in, structured result out.

Model loading lives in ``vlm_loader``; this module only runs generation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.document_understanding.vlm_config import BaselineVLMConfig
from src.document_understanding.vlm_interface import VLMModelError
from src.document_understanding.vlm_loader import LoadedVLM, load_baseline_vlm

logger = logging.getLogger(__name__)

ImageInput = Image.Image | np.ndarray | str | Path


class InvalidImageError(ValueError):
    """Raised when the provided image cannot be read or interpreted."""


def load_image(image: ImageInput) -> Image.Image:
    """Normalize supported image inputs into an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise InvalidImageError(f"Image file not found: {path}")
        try:
            with Image.open(path) as opened:
                return opened.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(f"Could not read image '{path}': {exc}") from exc

    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            return Image.fromarray(array.astype(np.uint8)).convert("RGB")
        if array.ndim == 3 and array.shape[2] in {3, 4}:
            return Image.fromarray(array[:, :, :3].astype(np.uint8)).convert("RGB")
        raise InvalidImageError(f"Unsupported image array shape: {array.shape}")

    raise InvalidImageError(f"Unsupported image input type: {type(image).__name__}")


def _memory_usage(device: str) -> dict[str, Any]:
    memory: dict[str, Any] = {}

    try:
        import psutil

        process = psutil.Process()
        memory["process_rss_mb"] = round(process.memory_info().rss / (1024**2), 2)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - memory reporting must never break inference
        logger.debug("Could not read process memory: %s", exc)

    if device.startswith("cuda"):
        try:
            import torch

            memory["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 2)
            memory["cuda_max_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / (1024**2), 2
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read CUDA memory: %s", exc)

    return memory


def _extract_response(decoded: str, prompt: str) -> str:
    for marker in ("Assistant:", "<|im_start|>assistant"):
        if marker in decoded:
            return decoded.rsplit(marker, maxsplit=1)[-1].strip()
    if prompt and prompt in decoded:
        return decoded.split(prompt, maxsplit=1)[-1].strip()
    return decoded.strip()


def run_vlm_inference(
    image: ImageInput,
    prompt: str,
    config: BaselineVLMConfig | None = None,
    *,
    loaded_model: LoadedVLM | None = None,
) -> dict[str, Any]:
    """Run a single image + prompt through the baseline VLM.

    Returns a dict with ``response``, ``model_name``, ``device``,
    ``inference_time`` and ``memory``.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    pil_image = load_image(image)
    loaded = loaded_model or load_baseline_vlm(config)
    config = loaded.config

    import torch

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
    }
    if config.do_sample:
        generation_kwargs["temperature"] = config.temperature

    start = time.perf_counter()
    try:
        text = loaded.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = loaded.processor(text=text, images=pil_image, return_tensors="pt")
        inputs = {key: value.to(config.device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = loaded.model.generate(**inputs, **generation_kwargs)
        decoded = loaded.processor.decode(output_ids[0], skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError as exc:
        raise VLMModelError(
            f"Out of GPU memory during inference with '{config.model_id}'. "
            f"Reduce max_new_tokens or image size. Original error: {exc}"
        ) from exc
    except MemoryError as exc:
        raise VLMModelError(
            f"Out of memory during inference with '{config.model_id}'. "
            f"Reduce max_new_tokens or image size. Original error: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as an actionable VLM error
        raise VLMModelError(f"VLM inference failed for '{config.model_id}': {exc}") from exc

    inference_time = time.perf_counter() - start
    response = _extract_response(decoded, prompt)

    logger.info(
        "VLM inference complete: model=%s device=%s time=%.2fs",
        config.model_id,
        config.device,
        inference_time,
    )

    return {
        "response": response,
        "model_name": config.model_id,
        "device": config.device,
        "inference_time": round(inference_time, 3),
        "memory": _memory_usage(config.device),
    }
