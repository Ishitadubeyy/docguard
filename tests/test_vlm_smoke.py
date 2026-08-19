"""Live VLM smoke test.

Separate from the unit tests: this test downloads/loads real weights and runs
real inference. Run explicitly with:

    pytest tests/test_vlm_smoke.py -m integration -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.document_understanding.prompts import get_baseline_prompt
from src.document_understanding.vlm_config import BaselineVLMConfig
from src.document_understanding.vlm_inference import run_vlm_inference
from tests.conftest import requires_vlm

pytestmark = [pytest.mark.integration, requires_vlm]


@pytest.fixture
def synthetic_invoice(tmp_path: Path) -> Path:
    from scripts.create_synthetic_invoice import create_synthetic_invoice

    return create_synthetic_invoice(tmp_path / "synthetic_invoice.png")


def test_live_baseline_inference(synthetic_invoice: Path):
    config = BaselineVLMConfig(max_new_tokens=64)
    result = run_vlm_inference(synthetic_invoice, get_baseline_prompt("describe"), config)

    assert isinstance(result["response"], str)
    assert result["response"].strip()
    assert result["model_name"] == config.model_id
    assert result["device"] == config.device
    assert result["inference_time"] > 0
