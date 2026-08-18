"""Tests for document understanding configuration and parsing."""

from __future__ import annotations

import json

import pytest

from src.document_understanding.config import DEFAULT_VLM_MODEL_ID, TaskType, VLMConfig
from src.document_understanding.eval_logging import build_inference_log
from src.document_understanding.models import Document, DocumentPage, DocumentUnderstandingResult
from src.document_understanding.parser import (
    extract_json_text,
    iter_json_candidates,
    parse_json_response,
    parse_structured_output,
    validate_extraction_result,
)


def test_default_vlm_config_uses_small_model() -> None:
    config = VLMConfig()
    assert config.model_id == DEFAULT_VLM_MODEL_ID
    assert config.device == "cpu"
    assert config.dtype == "float32"


def test_vlm_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_MODEL_ID", "custom/model")
    monkeypatch.setenv("VLM_DEVICE", "cpu")
    monkeypatch.setenv("VLM_MAX_NEW_TOKENS", "128")
    monkeypatch.setenv("VLM_DTYPE", "float32")

    config = VLMConfig.from_env()
    assert config.model_id == "custom/model"
    assert config.max_new_tokens == 128


def test_task_type_from_string() -> None:
    task = TaskType.from_string("document_understanding")
    assert task is TaskType.DOCUMENT_UNDERSTANDING


def test_task_type_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unsupported task"):
        TaskType.from_string("unknown_task")


def test_document_representation(sample_array) -> None:
    page = DocumentPage(page_number=1, image=sample_array, ocr_text="hello")
    document = Document(document_id="doc-1", pages=[page], metadata={"filename": "a.png"})

    assert document.document_id == "doc-1"
    assert document.pages[0].ocr_text == "hello"
    assert document.metadata["filename"] == "a.png"


def test_parse_clean_json() -> None:
    raw = json.dumps(
        {
            "document_type": "invoice",
            "fields": {"invoice_number": "INV-1"},
            "tables": [],
            "summary": "Invoice",
        }
    )
    result = parse_structured_output(raw)
    assert result.parse_success is True
    assert result.schema_valid is True
    assert result.schema_normalized is False
    assert result.document_type == "invoice"
    assert result.fields["invoice_number"] == "INV-1"
    assert result.parse_errors == []
    assert result.schema_errors == []


def test_extract_json_from_markdown_fence() -> None:
    raw = 'Here is the result:\n```json\n{"document_type": "invoice"}\n```'
    assert extract_json_text(raw) == '{"document_type": "invoice"}'


def test_extract_json_from_surrounding_explanation() -> None:
    raw = (
        "Based on the document, here is the extraction:\n"
        '{"document_type": "invoice", "fields": {}, "tables": [], "summary": "test"}\n'
        "End of response."
    )
    result = parse_structured_output(raw)
    assert result.parse_success is True
    assert result.document_type == "invoice"
    assert result.raw_response == raw


def test_parse_json_response_rejects_malformed_output() -> None:
    with pytest.raises(ValueError, match="Malformed JSON"):
        parse_json_response('{"document_type": }')


def test_parse_structured_output_handles_malformed_model_output() -> None:
    result = parse_structured_output("not json at all", page_number=1)
    assert result.parse_errors
    assert result.parse_success is False
    assert result.page_number == 1
    assert result.raw_response == "not json at all"


def test_parse_structured_output_handles_empty_model_response() -> None:
    result = parse_structured_output("   ")
    assert result.parse_success is False
    assert result.parse_errors == ["Model output is empty"]
    assert result.document_type is None
    assert result.fields == {}


def test_parse_structured_output_records_missing_fields_without_fabrication() -> None:
    result = parse_structured_output('{"document_type": "invoice"}')
    assert result.parse_success is True
    assert result.document_type == "invoice"
    assert result.fields == {}
    assert result.tables == []
    assert result.summary is None
    assert "missing required field: fields" in result.schema_errors
    assert "missing required field: tables" in result.schema_errors
    assert "missing required field: summary" in result.schema_errors
    assert result.parse_errors == []
    assert result.schema_valid is False
    assert result.schema_normalized is False


def test_parse_structured_output_unwraps_nested_text_object() -> None:
    raw = json.dumps(
        {
            "text": {
                "document_type": "invoice",
                "fields": {"total_amount": "20.00"},
                "tables": [],
                "summary": "Invoice",
            }
        }
    )
    result = parse_structured_output(raw)
    assert result.parse_success is True
    assert result.document_type == "invoice"
    assert 'Unwrapped unexpected nested "text" object' in result.schema_errors
    assert result.parse_errors == []


def test_parse_structured_output_rejects_invalid_field_types() -> None:
    raw = json.dumps(
        {
            "document_type": 123,
            "fields": "bad",
            "tables": {},
            "summary": 99,
        }
    )
    result = parse_structured_output(raw)
    assert result.parse_success is True
    assert result.document_type is None
    assert result.fields == {}
    assert result.tables == []
    assert result.summary is None
    assert "document_type must be a string or null" in result.schema_errors
    assert "fields must be an object" in result.schema_errors
    assert "tables must be a list" in result.schema_errors
    assert "summary must be a string or null" in result.schema_errors
    assert result.parse_errors == []


def test_iter_json_candidates_prefers_first_valid_object() -> None:
    raw = 'Noise {"broken": } then {"document_type": "invoice", "fields": {}, "tables": [], "summary": "x"}'
    candidates = iter_json_candidates(raw)
    assert len(candidates) >= 2
    assert extract_json_text(raw) == (
        '{"document_type": "invoice", "fields": {}, "tables": [], "summary": "x"}'
    )


def test_validate_extraction_result_sets_null_confidence() -> None:
    result = validate_extraction_result(
        {
            "document_type": "invoice",
            "fields": {"total": "10.00"},
            "tables": [],
            "summary": "Invoice",
            "confidence": None,
        }
    )
    assert result.document_type == "invoice"
    assert result.confidence is None
    assert result.parse_success is True
    assert result.parse_errors == []
    assert result.schema_errors == []


def test_validate_extraction_result_rejects_invalid_confidence() -> None:
    result = validate_extraction_result(
        {
            "document_type": "invoice",
            "fields": {},
            "tables": [],
            "summary": None,
            "confidence": "high",
        }
    )
    assert result.confidence is None
    assert "confidence must be a number or null" in result.schema_errors


def test_parse_structured_output_preserves_raw_response() -> None:
    raw = '{"document_type":"receipt","fields":{},"tables":[],"summary":"ok"}'
    result = parse_structured_output(raw, page_number=2)
    assert result.raw_response == raw
    assert result.page_number == 2


def test_build_inference_log_records_required_fields() -> None:
    result = DocumentUnderstandingResult(
        document_type="invoice",
        fields={"invoice_number": "1", "date": None},
        tables=[],
        summary="Invoice",
        parse_success=True,
        schema_valid=True,
        schema_normalized=False,
    )
    log = build_inference_log(
        model_id="mock/vlm",
        task="document_understanding",
        latency_seconds=1.5,
        result=result,
        ocr_text_length=120,
        page_number=1,
    )
    payload = log.to_dict()
    assert payload["model_id"] == "mock/vlm"
    assert payload["task"] == "document_understanding"
    assert payload["latency_seconds"] == 1.5
    assert payload["parse_success"] is True
    assert payload["schema_valid"] is True
    assert payload["schema_normalized"] is False
    assert payload["document_type"] == "invoice"
    assert payload["number_of_extracted_fields"] == 1
    assert payload["ocr_text_length"] == 120
    assert payload["timestamp"]


def test_document_understanding_result_to_dict() -> None:
    result = DocumentUnderstandingResult(document_type="receipt", fields={"total": "5"})
    payload = result.to_dict()
    assert payload["document_type"] == "receipt"
    assert payload["fields"]["total"] == "5"


def test_evaluation_metrics_do_not_fabricate_scores() -> None:
    from evaluation.benchmarks.document_understanding.metrics import evaluate_sample
    from evaluation.benchmarks.document_understanding.schema import BenchmarkSample

    sample = BenchmarkSample(
        sample_id="s1",
        document_type="invoice",
        fields={"total": "10.00"},
        corrected_text="Invoice total 10.00",
    )
    results = evaluate_sample(
        model="mock/vlm",
        dataset="dev",
        task="document_understanding",
        sample=sample,
        prediction={
            "document_type": "invoice",
            "fields": {"total": "10.00"},
            "tables": [],
            "summary": "Invoice total 10.00",
            "confidence": None,
            "raw_response": '{"document_type":"invoice"}',
        },
        latency_seconds=1.2,
    )

    metrics = {result.metric: result.score for result in results}
    assert metrics["field_extraction_accuracy"] == 1.0
    assert metrics["document_classification_accuracy"] == 1.0
    assert metrics["json_parse_success"] == 1.0
    assert metrics["schema_compliance"] == 1.0
    assert metrics["schema_normalization_rate"] == 0.0
    assert metrics["field_extraction_count"] == 1.0


def test_parse_already_valid_nested_schema() -> None:
    raw = json.dumps(
        {
            "document_type": "INVOICE",
            "fields": {
                "invoice_number": "INV-2026-001",
                "date": "18 August 2026",
                "customer_name": "Rahul Sharma",
                "company": "Example Technologies Pvt Ltd",
                "total_amount": "20000",
            },
            "tables": [],
            "summary": "Invoice from Example Technologies Pvt Ltd",
        }
    )
    result = parse_structured_output(raw)
    assert result.parse_success is True
    assert result.schema_valid is True
    assert result.schema_normalized is False
    assert result.fields["invoice_number"] == "INV-2026-001"
    assert result.schema_errors == []
    assert result.parse_errors == []


def test_parse_normalizes_flat_known_fields() -> None:
    raw = json.dumps(
        {
            "document_type": "INVOICE",
            "invoice_number": "INV-2026-001",
            "date": "18 August 2026",
            "customer_name": "Rahul Sharma",
            "company": "Example Technologies Pvt Ltd",
            "total_amount": "20000",
        }
    )
    result = parse_structured_output(raw)
    assert result.parse_success is True
    assert result.schema_valid is False
    assert result.schema_normalized is True
    assert result.document_type == "INVOICE"
    assert result.fields == {
        "invoice_number": "INV-2026-001",
        "date": "18 August 2026",
        "customer_name": "Rahul Sharma",
        "company": "Example Technologies Pvt Ltd",
        "total_amount": "20000",
    }
    assert "unexpected top-level field: invoice_number" in result.schema_errors
    assert "unexpected top-level field: date" in result.schema_errors
    assert "missing required field: tables" in result.schema_errors
    assert "missing required field: summary" in result.schema_errors
    assert result.parse_errors == []


def test_parse_records_unknown_top_level_fields() -> None:
    raw = json.dumps(
        {
            "document_type": "invoice",
            "fields": {"invoice_number": "INV-1"},
            "tables": [],
            "summary": "Invoice",
            "vendor_id": "V-99",
        }
    )
    result = parse_structured_output(raw)
    assert result.parse_success is True
    assert result.schema_valid is False
    assert result.schema_normalized is False
    assert "unexpected top-level field: vendor_id" in result.schema_errors
    assert result.fields["invoice_number"] == "INV-1"


def test_parse_malformed_json_sets_parse_errors_only() -> None:
    result = parse_structured_output("{not valid json")
    assert result.parse_success is False
    assert result.schema_valid is False
    assert result.schema_normalized is False
    assert result.parse_errors
    assert result.schema_errors == []


def test_parse_preserves_raw_response_after_normalization() -> None:
    raw = json.dumps(
        {
            "document_type": "INVOICE",
            "invoice_number": "INV-2026-001",
            "date": "18 August 2026",
        }
    )
    result = parse_structured_output(raw, page_number=3)
    assert result.raw_response == raw
    assert result.page_number == 3


def test_schema_normalization_flag_only_when_flat_fields_moved() -> None:
    valid_raw = json.dumps(
        {
            "document_type": "invoice",
            "fields": {"invoice_number": "INV-1"},
            "tables": [],
            "summary": "Invoice",
        }
    )
    flat_raw = json.dumps({"document_type": "invoice", "invoice_number": "INV-1"})

    valid_result = parse_structured_output(valid_raw)
    flat_result = parse_structured_output(flat_raw)

    assert valid_result.schema_normalized is False
    assert flat_result.schema_normalized is True


def test_schema_errors_preserved_separate_from_parse_errors() -> None:
    raw = json.dumps({"document_type": "invoice", "invoice_number": "INV-1"})
    result = parse_structured_output(raw)
    assert result.parse_success is True
    assert result.parse_errors == []
    assert result.schema_errors
    assert all("unexpected top-level field" in error or "missing required field" in error for error in result.schema_errors)


def test_flat_field_normalization_does_not_count_as_schema_valid() -> None:
    raw = json.dumps(
        {
            "document_type": "INVOICE",
            "invoice_number": "INV-2026-001",
            "customer_name": "Rahul Sharma",
        }
    )
    result = parse_structured_output(raw)
    assert result.schema_valid is False
    assert result.schema_normalized is True
    assert result.fields["invoice_number"] == "INV-2026-001"
    assert result.fields["customer_name"] == "Rahul Sharma"
    assert sum(1 for value in result.fields.values() if value) == 2


def test_evaluation_metrics_separate_json_and_schema_compliance() -> None:
    from evaluation.benchmarks.document_understanding.metrics import evaluate_sample
    from evaluation.benchmarks.document_understanding.schema import BenchmarkSample

    flat_prediction = json.dumps(
        {
            "document_type": "INVOICE",
            "invoice_number": "INV-2026-001",
            "customer_name": "Rahul Sharma",
        }
    )
    sample = BenchmarkSample(
        sample_id="flat-schema",
        document_type="INVOICE",
        fields={"invoice_number": "INV-2026-001", "customer_name": "Rahul Sharma"},
    )
    results = evaluate_sample(
        model="mock/vlm",
        dataset="dev",
        task="document_understanding",
        sample=sample,
        prediction=flat_prediction,
    )
    metrics = {result.metric: result.score for result in results}
    assert metrics["json_parse_success"] == 1.0
    assert metrics["schema_compliance"] == 0.0
    assert metrics["schema_normalization_rate"] == 1.0
    assert metrics["field_extraction_count"] == 2.0
