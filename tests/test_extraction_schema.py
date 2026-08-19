"""Unit tests for document understanding schemas, parsers, and metrics."""

from __future__ import annotations

import pytest

from src.document_understanding.evaluation import (
    classification_accuracy,
    exact_match,
    field_accuracy,
    qa_exact_match,
    table_accuracy,
)
from src.document_understanding.extraction_schema import (
    AnalysisTask,
    DocumentType,
    normalize_field_name,
    normalize_value,
    parse_classification,
    parse_key_value,
    parse_question_answer,
    parse_summary,
    parse_table,
    schema_for,
)
from src.document_understanding.task_prompts import build_task_prompt


class TestTaskValidation:
    def test_supported_tasks_resolve(self):
        assert AnalysisTask.from_string("classification") is AnalysisTask.CLASSIFICATION
        assert AnalysisTask.from_string(" Table_Extraction ") is AnalysisTask.TABLE_EXTRACTION

    def test_unsupported_task_rejected(self):
        with pytest.raises(ValueError, match="Unsupported task"):
            AnalysisTask.from_string("forgery_detection")

    def test_schema_for_known_document_types(self):
        assert "invoice_number" in schema_for(DocumentType.INVOICE)
        assert "net_salary" in schema_for("salary_slip")
        assert schema_for(None) == ()


class TestPromptGeneration:
    def test_prompts_are_distinct_per_task(self):
        prompts = {
            build_task_prompt(task, question="What is the total amount?")
            for task in AnalysisTask
        }
        assert len(prompts) == len(AnalysisTask)

    def test_key_value_prompt_lists_schema_fields(self):
        prompt = build_task_prompt("key_value_extraction", document_type="bank_statement")
        for field in schema_for("bank_statement"):
            assert field in prompt

    def test_prompts_state_grounding_rules(self):
        prompt = build_task_prompt("summarization")
        assert "only information that is visible" in prompt
        assert "null" in prompt

    def test_question_answering_requires_question(self):
        with pytest.raises(ValueError, match="question must be a non-empty string"):
            build_task_prompt("question_answering", question="   ")

    def test_ocr_context_is_appended_when_provided(self):
        prompt = build_task_prompt("classification", ocr_text="INVOICE\nINV-001")
        assert "INV-001" in prompt
        assert "source of truth" in prompt


class TestClassificationParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("This is an invoice.", "invoice"),
            ('{"document_type": "bank_statement"}', "bank_statement"),
            ("The image shows a payslip for March.", "salary_slip"),
            ("An identity card belonging to a person.", "identity_document"),
            ("A photo of a cat.", "unknown"),
        ],
    )
    def test_labels_are_evidence_based(self, raw, expected):
        result = parse_classification(raw)
        assert result.document_type == expected
        assert result.raw_response == raw

    def test_empty_output_is_unknown_and_flagged(self):
        result = parse_classification("")
        assert result.document_type == "unknown"
        assert result.parse_success is False
        assert result.parse_errors


class TestKeyValueParsing:
    def test_parses_json_fields_into_schema(self):
        raw = '{"fields": {"invoice_number": "INV-001", "customer": "Alice Example"}}'
        result = parse_key_value(raw, DocumentType.INVOICE)
        assert result.fields["invoice_number"] == "INV-001"
        assert result.fields["amount"] is None
        assert result.parse_success is True

    def test_parses_line_format_and_aliases(self):
        raw = "Invoice No: INV-002\nTotal Amount: 1200\nCustomer Name: Bob Sample"
        result = parse_key_value(raw, "invoice")
        assert result.fields["invoice_number"] == "INV-002"
        assert result.fields["amount"] == "1200"
        assert result.fields["customer"] == "Bob Sample"

    def test_unexpected_fields_are_preserved(self):
        result = parse_key_value("Vendor: Acme\nInvoice Number: INV-003", "invoice")
        assert result.fields["additional_fields"] == {"vendor": "Acme"}

    def test_parses_list_of_field_objects(self):
        raw = '{"fields": [{"employee_id": "EMP-1"}, {"net_salary": "35000"}]}'
        result = parse_key_value(raw, "salary_slip")
        assert result.fields["employee_id"] == "EMP-1"
        assert result.fields["net_salary"] == "35000"

    def test_placeholder_values_become_null(self):
        result = parse_key_value('{"fields": {"amount": "N/A"}}', "invoice")
        assert result.fields["amount"] is None

    def test_malformed_output_is_reported_not_invented(self):
        result = parse_key_value("I cannot read this document", "invoice")
        assert result.parse_success is False
        assert all(value is None for value in result.fields.values())
        assert result.parse_errors


class TestTableParsing:
    def test_parses_json_columns_and_rows(self):
        raw = '{"columns": ["Date", "Amount"], "rows": [["2026-02-03", "40000"]]}'
        result = parse_table(raw)
        assert result.table == {"columns": ["Date", "Amount"], "rows": [["2026-02-03", "40000"]]}
        assert result.parse_success is True

    def test_parses_markdown_pipe_table(self):
        raw = "| Date | Amount |\n| --- | --- |\n| 2026-02-10 | 15000 |"
        result = parse_table(raw)
        assert result.table["columns"] == ["Date", "Amount"]
        assert result.table["rows"] == [["2026-02-10", "15000"]]

    def test_echoed_schema_placeholders_are_not_rows(self):
        result = parse_table('{"columns": ["<column>"], "rows": [["<cell or null>"]]}')
        assert result.table["rows"] == []
        assert result.parse_success is False

    def test_rows_are_not_fabricated(self):
        result = parse_table("The document has a table of transactions.")
        assert result.table == {"columns": [], "rows": []}
        assert result.parse_success is False
        assert result.raw_response


class TestSummaryAndQAParsing:
    def test_summary_strips_code_fences(self):
        result = parse_summary("```\nAn invoice for one laptop.\n```")
        assert result.summary == "An invoice for one laptop"
        assert result.parse_success is True

    def test_empty_summary_is_flagged(self):
        result = parse_summary("   ")
        assert result.summary is None
        assert result.parse_errors

    def test_answer_extracted_from_json_or_text(self):
        assert parse_question_answer('{"answer": "INV-001"}', "q").answer == "INV-001"
        assert parse_question_answer("INV-001", "q").answer == "INV-001"

    def test_not_visible_answer_becomes_null(self):
        result = parse_question_answer("not visible in document", "What is the IBAN?")
        assert result.answer is None
        assert result.parse_success is False

    def test_confidence_is_never_invented(self):
        for result in (
            parse_classification("invoice"),
            parse_key_value("Amount: 10", "invoice"),
            parse_summary("An invoice."),
        ):
            assert result.confidence is None


class TestNormalization:
    def test_field_names_normalized_and_aliased(self):
        assert normalize_field_name("Invoice No") == "invoice_number"
        assert normalize_field_name(" Net Pay ") == "net_salary"

    def test_values_trimmed_and_nulled(self):
        assert normalize_value("  INV-001.  ") == "INV-001"
        assert normalize_value("unknown") is None
        assert normalize_value(None) is None


class TestEvaluationMetrics:
    def test_exact_match_ignores_case_and_punctuation(self):
        assert exact_match("Alice Example.", "alice example") is True
        assert exact_match(None, "alice") is False

    def test_classification_accuracy(self):
        scores = classification_accuracy([("invoice", "invoice"), ("unknown", "salary_slip")])
        assert scores == {"total": 2, "correct": 1, "accuracy": 0.5}

    def test_field_accuracy_reports_per_field(self):
        scores = field_accuracy(
            {"invoice_number": "INV-001", "amount": None},
            {"invoice_number": "INV-001", "amount": "50000"},
        )
        assert scores["accuracy"] == 0.5
        assert scores["per_field"] == {"invoice_number": True, "amount": False}

    def test_table_accuracy_counts_rows_and_cells(self):
        scores = table_accuracy(
            {"columns": ["a", "b"], "rows": [["1", "2"], ["3", "9"]]},
            [["1", "2"], ["3", "4"]],
        )
        assert scores["correct_rows"] == 1
        assert scores["correct_cells"] == 3
        assert scores["row_accuracy"] == 0.5

    def test_qa_exact_match(self):
        assert qa_exact_match([("INV-001", "INV-001"), (None, "22500")])["exact_match"] == 0.5
