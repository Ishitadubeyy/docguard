"""Unit tests for the baseline VLM layer (no model downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.document_understanding.prompts import (
    BASELINE_PROMPTS,
    build_document_qa_prompt,
    get_baseline_prompt,
    list_baseline_prompts,
)
from src.document_understanding.vlm_config import (
    DEFAULT_BASELINE_MODEL_ID,
    BaselineVLMConfig,
    default_dtype_for,
)
from src.document_understanding.vlm_inference import (
    InvalidImageError,
    load_image,
    run_vlm_inference,
)
from src.document_understanding.vlm_loader import LoadedVLM


class FakeProcessor:
    """Minimal processor stub with the interface used by run_vlm_inference."""

    def __init__(self, response: str = "A synthetic invoice.") -> None:
        self.response = response
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, add_generation_prompt=True):
        self.calls.append({"messages": messages})
        return "<chat>"

    def __call__(self, text, images, return_tensors="pt"):
        import torch

        return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}

    def decode(self, output_ids, skip_special_tokens=True):
        return f"User: prompt\nAssistant: {self.response}"


class FakeModel:
    def generate(self, **kwargs):
        import torch

        return torch.zeros((1, 8), dtype=torch.long)


@pytest.fixture
def synthetic_image(tmp_path: Path) -> Path:
    from scripts.create_synthetic_invoice import create_synthetic_invoice

    return create_synthetic_invoice(tmp_path / "synthetic_invoice.png")


# --- configuration ---------------------------------------------------------


def test_default_config_uses_configurable_model_id():
    config = BaselineVLMConfig()
    assert config.model_id == DEFAULT_BASELINE_MODEL_ID
    assert config.device in {"cpu", "cuda"}
    assert config.dtype == default_dtype_for(config.device)
    assert config.max_new_tokens > 0
    assert config.do_sample is False


def test_config_overrides_and_serialization():
    config = BaselineVLMConfig(
        model_id="org/other-vlm",
        device="cpu",
        dtype="float32",
        max_new_tokens=32,
        temperature=0.7,
        cache_dir="/tmp/models",
    )
    assert config.do_sample is True
    payload = config.to_dict()
    assert payload["model_id"] == "org/other-vlm"
    assert payload["cache_dir"] == "/tmp/models"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": ""},
        {"max_new_tokens": 0},
        {"temperature": -1.0},
        {"dtype": "int4"},
    ],
)
def test_invalid_config_values_rejected(kwargs):
    with pytest.raises(ValueError):
        BaselineVLMConfig(**kwargs)


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("VLM_MODEL_ID", "org/env-vlm")
    monkeypatch.setenv("VLM_DEVICE", "cpu")
    monkeypatch.setenv("VLM_MAX_NEW_TOKENS", "64")
    monkeypatch.setenv("VLM_TEMPERATURE", "0.0")
    config = BaselineVLMConfig.from_env()
    assert (config.model_id, config.device, config.max_new_tokens) == ("org/env-vlm", "cpu", 64)
    assert config.dtype == "float32"


# --- prompts ---------------------------------------------------------------


def test_baseline_prompt_registry_covers_required_tasks():
    assert set(BASELINE_PROMPTS) == {"describe", "classify", "key_value"}
    assert "qa" in list_baseline_prompts()


@pytest.mark.parametrize("name", ["describe", "classify", "key_value"])
def test_baseline_prompts_are_non_empty(name):
    prompt = get_baseline_prompt(name)
    assert prompt.strip()


def test_qa_prompt_includes_question():
    prompt = build_document_qa_prompt("What is the invoice number?")
    assert "What is the invoice number?" in prompt


def test_qa_prompt_requires_question():
    with pytest.raises(ValueError):
        get_baseline_prompt("qa")
    with pytest.raises(ValueError):
        build_document_qa_prompt("   ")


def test_unknown_prompt_name_rejected():
    with pytest.raises(ValueError):
        get_baseline_prompt("summarize")


# --- image handling --------------------------------------------------------


def test_load_image_accepts_path_pil_and_array(synthetic_image: Path):
    assert load_image(synthetic_image).mode == "RGB"
    assert load_image(Image.new("L", (10, 10))).mode == "RGB"
    assert load_image(np.zeros((10, 10, 3), dtype=np.uint8)).size == (10, 10)


def test_missing_image_raises(tmp_path: Path):
    with pytest.raises(InvalidImageError):
        load_image(tmp_path / "does_not_exist.png")


def test_corrupt_image_raises(tmp_path: Path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a real png")
    with pytest.raises(InvalidImageError):
        load_image(broken)


def test_unsupported_image_type_raises():
    with pytest.raises(InvalidImageError):
        load_image(42)  # type: ignore[arg-type]


# --- inference contract ----------------------------------------------------


def test_response_structure_with_stub_model(synthetic_image: Path):
    pytest.importorskip("torch")
    config = BaselineVLMConfig(model_id="stub/model", device="cpu", max_new_tokens=8)
    loaded = LoadedVLM(model=FakeModel(), processor=FakeProcessor(), config=config)

    result = run_vlm_inference(synthetic_image, "Describe this document.", loaded_model=loaded)

    assert set(result) >= {"response", "model_name", "device", "inference_time", "memory"}
    assert result["response"] == "A synthetic invoice."
    assert result["model_name"] == "stub/model"
    assert result["device"] == "cpu"
    assert isinstance(result["inference_time"], float)
    assert isinstance(result["memory"], dict)


def test_inference_rejects_empty_prompt(synthetic_image: Path):
    pytest.importorskip("torch")
    config = BaselineVLMConfig(model_id="stub/model", device="cpu")
    loaded = LoadedVLM(model=FakeModel(), processor=FakeProcessor(), config=config)
    with pytest.raises(ValueError):
        run_vlm_inference(synthetic_image, "   ", loaded_model=loaded)


def test_inference_rejects_invalid_image():
    config = BaselineVLMConfig(model_id="stub/model", device="cpu")
    loaded = LoadedVLM(model=FakeModel(), processor=FakeProcessor(), config=config)
    with pytest.raises(InvalidImageError):
        run_vlm_inference("/no/such/image.png", "Describe this document.", loaded_model=loaded)


# --- loading ---------------------------------------------------------------


def test_loading_unknown_model_reports_actionable_error():
    pytest.importorskip("transformers")
    from src.document_understanding.vlm_interface import VLMModelError
    from src.document_understanding.vlm_loader import load_baseline_vlm

    config = BaselineVLMConfig(
        model_id="docguard/definitely-not-a-real-model",
        device="cpu",
        local_files_only=True,
    )
    with pytest.raises(VLMModelError) as excinfo:
        load_baseline_vlm(config)
    assert "docguard/definitely-not-a-real-model" in str(excinfo.value)
