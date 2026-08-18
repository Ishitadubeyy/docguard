"""Schemas and parsers for document understanding task outputs.

This module turns raw VLM text into structured task results. It never invents
values: anything the model did not produce stays ``None`` and the raw response
is always preserved so failures can be inspected.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from src.document_understanding.parser import iter_json_candidates


class AnalysisTask(str, Enum):
    """Document understanding tasks supported by the analyzer."""

    CLASSIFICATION = "classification"
    KEY_VALUE_EXTRACTION = "key_value_extraction"
    TABLE_EXTRACTION = "table_extraction"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"

    @classmethod
    def from_string(cls, value: str) -> AnalysisTask:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(task.value for task in cls)
            raise ValueError(
                f"Unsupported task '{value}'. Supported tasks: {supported}"
            ) from exc


class DocumentType(str, Enum):
    """Synthetic document types covered by this phase."""

    INVOICE = "invoice"
    BANK_STATEMENT = "bank_statement"
    SALARY_SLIP = "salary_slip"
    IDENTITY_DOCUMENT = "identity_document"
    UNKNOWN = "unknown"


DOCUMENT_FIELD_SCHEMAS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.INVOICE: (
        "invoice_number",
        "customer",
        "date",
        "item",
        "quantity",
        "amount",
    ),
    DocumentType.BANK_STATEMENT: (
        "account_name",
        "statement_period",
        "transaction_date",
        "description",
        "debit",
        "credit",
        "balance",
    ),
    DocumentType.SALARY_SLIP: (
        "employee_name",
        "employee_id",
        "pay_period",
        "basic_salary",
        "allowances",
        "deductions",
        "net_salary",
    ),
    DocumentType.IDENTITY_DOCUMENT: (
        "document_type",
        "name",
        "document_number",
        "date_of_birth",
        "expiry_date",
    ),
    DocumentType.UNKNOWN: (),
}

# Words the model may use for each document type. Matching is evidence based:
# a label is only assigned when its phrase actually occurs in the response.
_CLASSIFICATION_ALIASES: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.INVOICE: ("invoice", "bill", "tax invoice"),
    DocumentType.BANK_STATEMENT: (
        "bank_statement",
        "bank statement",
        "account statement",
        "statement of account",
    ),
    DocumentType.SALARY_SLIP: (
        "salary_slip",
        "salary slip",
        "payslip",
        "pay slip",
        "pay stub",
        "payroll slip",
    ),
    DocumentType.IDENTITY_DOCUMENT: (
        "identity_document",
        "identity document",
        "identity card",
        "id card",
        "identification card",
        "passport",
        "driving licence",
        "driver's license",
    ),
}

# Field name synonyms the model tends to emit, mapped to the schema field name.
_FIELD_ALIASES: dict[str, str] = {
    "invoice_no": "invoice_number",
    "invoice_id": "invoice_number",
    "bill_number": "invoice_number",
    "customer_name": "customer",
    "client": "customer",
    "billed_to": "customer",
    "invoice_date": "date",
    "issue_date": "date",
    "item_name": "item",
    "description_of_item": "item",
    "qty": "quantity",
    "total": "amount",
    "total_amount": "amount",
    "amount_due": "amount",
    "account_holder": "account_name",
    "account_holder_name": "account_name",
    "period": "statement_period",
    "statement_dates": "statement_period",
    "date_of_transaction": "transaction_date",
    "withdrawal": "debit",
    "deposit": "credit",
    "closing_balance": "balance",
    "employee": "employee_name",
    "emp_name": "employee_name",
    "emp_id": "employee_id",
    "employee_code": "employee_id",
    "pay_month": "pay_period",
    "salary_period": "pay_period",
    "basic": "basic_salary",
    "basic_pay": "basic_salary",
    "allowance": "allowances",
    "deduction": "deductions",
    "net_pay": "net_salary",
    "take_home": "net_salary",
    "full_name": "name",
    "holder_name": "name",
    "id_number": "document_number",
    "document_no": "document_number",
    "dob": "date_of_birth",
    "birth_date": "date_of_birth",
    "date_of_expiry": "expiry_date",
    "valid_until": "expiry_date",
}

_NULL_TOKENS = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "not visible",
    "not visible in document",
    "not available",
    "not provided",
}

# Models sometimes echo the prompt's schema placeholders such as "<value>".
_PLACEHOLDER = re.compile(r"^<[^<>]*>$")

_KEY_VALUE_LINE = re.compile(r"^\s*[-*\u2022]?\s*([A-Za-z][A-Za-z0-9 _/#.()-]{0,48}?)\s*[:=]\s*(.*)$")


@dataclass
class TaskResult:
    """Structured output of a single document understanding task."""

    task: str
    raw_response: str
    document_type: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    table: dict[str, Any] | None = None
    summary: str | None = None
    question: str | None = None
    answer: str | None = None
    confidence: None = None
    parse_success: bool = False
    parse_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_field_name(name: str) -> str:
    """Normalize a model-produced field name to snake_case schema wording."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return _FIELD_ALIASES.get(cleaned, cleaned)


def normalize_value(value: Any) -> Any:
    """Strip formatting noise; return ``None`` for placeholder values."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, dict)):
        return value

    text = str(value).strip().strip("\"'").strip()
    text = re.sub(r"[\s,.;]+$", "", text)
    if text.lower() in _NULL_TOKENS or _PLACEHOLDER.match(text):
        return None
    return text or None


def schema_for(document_type: DocumentType | str | None) -> tuple[str, ...]:
    """Return the expected field names for a document type."""
    if document_type is None:
        return ()
    doc_type = (
        document_type
        if isinstance(document_type, DocumentType)
        else DocumentType(str(document_type).strip().lower())
    )
    return DOCUMENT_FIELD_SCHEMAS.get(doc_type, ())


def _load_json_payload(raw_response: str) -> Any | None:
    """Return the first parseable JSON value in the response, else ``None``."""
    for candidate in iter_json_candidates(raw_response):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    stripped = raw_response.strip()
    if stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None
    return None


def parse_classification(raw_response: str) -> TaskResult:
    """Map a free-form classification response onto a known document type."""
    result = TaskResult(task=AnalysisTask.CLASSIFICATION.value, raw_response=raw_response)
    if not raw_response or not raw_response.strip():
        result.document_type = DocumentType.UNKNOWN.value
        result.parse_errors.append("Model output is empty")
        return result

    text = raw_response.lower()
    payload = _load_json_payload(raw_response)
    if isinstance(payload, dict):
        declared = payload.get("document_type") or payload.get("type")
        if isinstance(declared, str):
            text = f"{declared.lower()} {text}"

    best: tuple[int, DocumentType] | None = None
    for doc_type, aliases in _CLASSIFICATION_ALIASES.items():
        for alias in aliases:
            position = text.find(alias)
            if position != -1 and (best is None or position < best[0]):
                best = (position, doc_type)

    if best is None:
        result.document_type = DocumentType.UNKNOWN.value
        result.parse_errors.append("No known document type found in model output")
        return result

    result.document_type = best[1].value
    result.parse_success = True
    return result


def _flatten_scalar_entries(source: Any) -> dict[str, Any]:
    """Collect scalar key-value pairs from a dict or a list of dicts.

    Small models often answer with ``{"fields": [{...}, {...}]}`` instead of a
    flat object; entries are merged in order and the first value for a key wins.
    """
    entries = source if isinstance(source, list) else [source]
    collected: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if isinstance(value, (dict, list)):
                continue
            name = normalize_field_name(key)
            if name:
                collected.setdefault(name, normalize_value(value))
    return collected


def _fields_from_json(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        source = payload["fields"] if isinstance(payload.get("fields"), (dict, list)) else payload
        return _flatten_scalar_entries(source)
    if isinstance(payload, list):
        return _flatten_scalar_entries(payload)
    return {}


def _fields_from_lines(raw_response: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for line in raw_response.splitlines():
        match = _KEY_VALUE_LINE.match(line)
        if not match:
            continue
        key = normalize_field_name(match.group(1))
        if not key:
            continue
        fields.setdefault(key, normalize_value(match.group(2)))
    return fields


def parse_key_value(
    raw_response: str,
    document_type: DocumentType | str | None = None,
) -> TaskResult:
    """Parse key-value pairs from JSON or ``key: value`` lines.

    Schema fields for the document type are always present (``None`` when the
    model did not produce them); anything else the model emitted is kept under
    ``additional_fields`` rather than being discarded.
    """
    result = TaskResult(
        task=AnalysisTask.KEY_VALUE_EXTRACTION.value,
        raw_response=raw_response,
        document_type=str(document_type) if document_type else None,
    )

    extracted = _fields_from_json(_load_json_payload(raw_response)) or _fields_from_lines(
        raw_response
    )
    expected = schema_for(document_type)

    fields: dict[str, Any] = {name: extracted.get(name) for name in expected}
    additional = {key: value for key, value in extracted.items() if key not in fields}
    if expected:
        if additional:
            fields["additional_fields"] = additional
    else:
        fields.update(additional)

    result.fields = fields
    result.parse_success = bool(extracted)
    if not extracted:
        result.parse_errors.append("No key-value pairs found in model output")
    return result


def _rows_from_json(payload: Any) -> tuple[list[str], list[list[Any]]] | None:
    if isinstance(payload, dict) and "rows" in payload:
        columns = [str(column) for column in payload.get("columns") or []]
        raw_rows = payload.get("rows") or []
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
        columns = [str(key) for key in payload[0]]
        raw_rows = payload
    else:
        return None

    rows: list[list[Any]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            if not columns:
                columns = [str(key) for key in row]
            rows.append([normalize_value(row.get(column)) for column in columns])
        elif isinstance(row, list):
            rows.append([normalize_value(cell) for cell in row])
    return columns, rows


def _rows_from_pipe_table(raw_response: str) -> tuple[list[str], list[list[Any]]]:
    columns: list[str] = []
    rows: list[list[Any]] = []

    for line in raw_response.splitlines():
        stripped = line.strip()
        if stripped.count("|") < 2:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell or "") for cell in cells):
            continue
        if not columns:
            columns = cells
            continue
        rows.append([normalize_value(cell) for cell in cells])

    return columns, rows


def parse_table(raw_response: str) -> TaskResult:
    """Parse a table into ``{"columns": [...], "rows": [...]}``.

    When the response cannot be interpreted as a table the raw text is kept and
    the failure recorded; rows are never fabricated.
    """
    result = TaskResult(task=AnalysisTask.TABLE_EXTRACTION.value, raw_response=raw_response)

    parsed = _rows_from_json(_load_json_payload(raw_response))
    if parsed is None:
        parsed = _rows_from_pipe_table(raw_response)

    columns, rows = parsed
    rows = [row for row in rows if any(cell is not None for cell in row)]
    result.table = {"columns": columns, "rows": rows}
    result.parse_success = bool(rows)
    if not rows:
        result.parse_errors.append("No table rows found in model output")
    return result


def parse_summary(raw_response: str) -> TaskResult:
    """Keep the model summary as-is, minus code fences and empty output."""
    result = TaskResult(task=AnalysisTask.SUMMARIZATION.value, raw_response=raw_response)

    payload = _load_json_payload(raw_response)
    summary: Any = None
    if isinstance(payload, dict) and isinstance(payload.get("summary"), str):
        summary = payload["summary"]
    else:
        summary = re.sub(r"```[a-zA-Z]*", "", raw_response).replace("```", "").strip()

    summary = normalize_value(summary)
    result.summary = summary if isinstance(summary, str) else None
    result.parse_success = result.summary is not None
    if not result.parse_success:
        result.parse_errors.append("Model produced no summary text")
    return result


def parse_question_answer(raw_response: str, question: str) -> TaskResult:
    """Extract an answer, treating explicit 'not visible' replies as ``None``."""
    result = TaskResult(
        task=AnalysisTask.QUESTION_ANSWERING.value,
        raw_response=raw_response,
        question=question,
    )

    payload = _load_json_payload(raw_response)
    answer: Any
    if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
        answer = payload["answer"]
    else:
        answer = raw_response

    answer = normalize_value(answer)
    result.answer = answer if isinstance(answer, str) else None
    result.parse_success = result.answer is not None
    if not result.parse_success:
        result.parse_errors.append("Model gave no answer visible in the document")
    return result
