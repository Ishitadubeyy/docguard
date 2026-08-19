"""Task-specific prompts for the document understanding layer.

Prompts live here only; the analyzer and inference layers never build prompt
text inline. Every prompt tells the model to use visible information only, to
avoid inventing values, to use ``null`` when a value is missing, and to return
the requested structure.
"""

from __future__ import annotations

from src.document_understanding.extraction_schema import (
    DOCUMENT_FIELD_SCHEMAS,
    AnalysisTask,
    DocumentType,
    schema_for,
)

GROUNDING_RULES = (
    "Use only information that is visible in the document image. "
    "Do not invent, guess, or complete missing values. "
    "Use null when a value is not visible. "
    "Return exactly the requested structure and nothing else."
)

CLASSIFICATION_LABELS = tuple(
    doc_type.value for doc_type in DOCUMENT_FIELD_SCHEMAS if doc_type is not DocumentType.UNKNOWN
) + (DocumentType.UNKNOWN.value,)

CLASSIFICATION_PROMPT = (
    "Classify this document into exactly one of these types: "
    + ", ".join(CLASSIFICATION_LABELS)
    + ".\n"
    + GROUNDING_RULES
    + "\nRespond with JSON only: {\"document_type\": \"<one of the types above>\"}"
)

KEY_VALUE_PROMPT_TEMPLATE = (
    "Extract the following fields from this document image: {fields}.\n"
    + GROUNDING_RULES
    + "\nRespond with JSON only, one key per requested field: "
    "{{\"fields\": {{\"<field>\": \"<value or null>\"}}}}"
)

GENERIC_KEY_VALUE_PROMPT = (
    "Extract every key-value pair that is visible in this document image.\n"
    + GROUNDING_RULES
    + "\nRespond with JSON only: {\"fields\": {\"<field>\": \"<value or null>\"}}"
)

TABLE_PROMPT = (
    "Extract the table of rows visible in this document image, such as line "
    "items or transactions.\n"
    + GROUNDING_RULES
    + "\nRespond with JSON only: {\"columns\": [\"<column>\"], "
    "\"rows\": [[\"<cell or null>\"]]}. "
    "Return an empty rows list when no table is visible."
)

SUMMARIZATION_PROMPT = (
    "Summarize this document image in two or three sentences. "
    "Describe the document type and the information it actually shows.\n"
    + GROUNDING_RULES
    + "\nRespond with the summary text only."
)

QUESTION_ANSWERING_PROMPT_TEMPLATE = (
    "Answer the question using only this document image.\n"
    + GROUNDING_RULES
    + "\nIf the answer is not visible, reply exactly 'not visible in document'. "
    "Respond with the answer only.\nQuestion: {question}"
)

OCR_CONTEXT_TEMPLATE = (
    "Supporting OCR text (may contain recognition errors, the image remains the "
    "source of truth; do not correct the OCR layer):\n{ocr_text}"
)


def build_ocr_context(ocr_text: str | None) -> str:
    """Render optional OCR context, or an empty string when unavailable."""
    if not ocr_text or not ocr_text.strip():
        return ""
    return OCR_CONTEXT_TEMPLATE.format(ocr_text=ocr_text.strip())


def build_key_value_prompt(document_type: DocumentType | str | None = None) -> str:
    """Build the key-value prompt, scoped to a document type when known."""
    fields = schema_for(document_type)
    if not fields:
        return GENERIC_KEY_VALUE_PROMPT
    return KEY_VALUE_PROMPT_TEMPLATE.format(fields=", ".join(fields))


def build_question_answering_prompt(question: str) -> str:
    """Build the document question answering prompt."""
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string for question_answering")
    return QUESTION_ANSWERING_PROMPT_TEMPLATE.format(question=question.strip())


def build_task_prompt(
    task: AnalysisTask | str,
    *,
    question: str | None = None,
    document_type: DocumentType | str | None = None,
    ocr_text: str | None = None,
) -> str:
    """Build the prompt for a document understanding task."""
    resolved = AnalysisTask.from_string(task)

    if resolved is AnalysisTask.CLASSIFICATION:
        prompt = CLASSIFICATION_PROMPT
    elif resolved is AnalysisTask.KEY_VALUE_EXTRACTION:
        prompt = build_key_value_prompt(document_type)
    elif resolved is AnalysisTask.TABLE_EXTRACTION:
        prompt = TABLE_PROMPT
    elif resolved is AnalysisTask.SUMMARIZATION:
        prompt = SUMMARIZATION_PROMPT
    else:
        prompt = build_question_answering_prompt(question or "")

    context = build_ocr_context(ocr_text)
    return f"{prompt}\n\n{context}" if context else prompt
