"""Generate the synthetic document set used by experiment 02.

Every value below is fictional. No real identities, accounts, or scraped
documents are used. The script also writes the ground truth file consumed by
the evaluation runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = Path("data/synthetic/documents")
GROUND_TRUTH_FILENAME = "ground_truth.json"

SYNTHETIC_DOCUMENTS: list[dict[str, Any]] = [
    {
        "document_id": "synthetic_invoice",
        "document_type": "invoice",
        "title": "INVOICE",
        "body": [
            "Invoice Number: INV-001",
            "Customer: Alice Example",
            "Item: Laptop",
            "Quantity: 1",
            "Amount: 50000",
            "Date: 2026-01-01",
        ],
        "fields": {
            "invoice_number": "INV-001",
            "customer": "Alice Example",
            "date": "2026-01-01",
            "item": "Laptop",
            "quantity": "1",
            "amount": "50000",
        },
        "table": {
            "columns": ["Item", "Quantity", "Amount"],
            "rows": [["Laptop", "1", "50000"]],
        },
        "question": "What is the invoice number?",
        "answer": "INV-001",
    },
    {
        "document_id": "synthetic_bank_statement",
        "document_type": "bank_statement",
        "title": "BANK STATEMENT",
        "body": [
            "Account Name: Bob Sample",
            "Statement Period: 2026-02-01 to 2026-02-28",
        ],
        "fields": {
            "account_name": "Bob Sample",
            "statement_period": "2026-02-01 to 2026-02-28",
            "transaction_date": "2026-02-03",
            "description": "Salary Credit",
            "debit": "0",
            "credit": "40000",
            "balance": "40000",
        },
        "table": {
            "columns": ["Date", "Description", "Debit", "Credit", "Balance"],
            "rows": [
                ["2026-02-03", "Salary Credit", "0", "40000", "40000"],
                ["2026-02-10", "Rent Payment", "15000", "0", "25000"],
                ["2026-02-18", "Grocery Store", "2500", "0", "22500"],
            ],
        },
        "question": "What is the closing balance?",
        "answer": "22500",
    },
    {
        "document_id": "synthetic_salary_slip",
        "document_type": "salary_slip",
        "title": "SALARY SLIP",
        "body": [
            "Employee Name: Carol Placeholder",
            "Employee ID: EMP-1234",
            "Pay Period: 2026-03",
            "Basic Salary: 30000",
            "Allowances: 8000",
            "Deductions: 3000",
            "Net Salary: 35000",
        ],
        "fields": {
            "employee_name": "Carol Placeholder",
            "employee_id": "EMP-1234",
            "pay_period": "2026-03",
            "basic_salary": "30000",
            "allowances": "8000",
            "deductions": "3000",
            "net_salary": "35000",
        },
        "table": {
            "columns": ["Component", "Amount"],
            "rows": [
                ["Basic Salary", "30000"],
                ["Allowances", "8000"],
                ["Deductions", "3000"],
            ],
        },
        "question": "What is the net salary?",
        "answer": "35000",
    },
    {
        "document_id": "synthetic_identity_document",
        "document_type": "identity_document",
        "title": "IDENTITY CARD",
        "body": [
            "Document Type: Identity Card",
            "Name: Dave Fictional",
            "Document Number: ID-987654",
            "Date of Birth: 1990-05-20",
            "Expiry Date: 2030-05-19",
        ],
        "fields": {
            "document_type": "Identity Card",
            "name": "Dave Fictional",
            "document_number": "ID-987654",
            "date_of_birth": "1990-05-20",
            "expiry_date": "2030-05-19",
        },
        "table": {"columns": [], "rows": []},
        "question": "What is the document number?",
        "answer": "ID-987654",
    },
]

FOOTER = "Synthetic document generated for DocGuard AI research. Not a real document."


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render(spec: dict[str, Any], output: Path) -> Path:
    image = Image.new("RGB", (1000, 760), "white")
    draw = ImageDraw.Draw(image)

    y = 50
    draw.text((70, y), spec["title"], fill="black", font=_font(44))
    y += 80

    for line in spec["body"]:
        draw.text((70, y), line, fill="black", font=_font(30))
        y += 46

    table = spec.get("table") or {}
    if table.get("rows"):
        y += 24
        column_x = [70, 260, 560, 700, 840]
        header_font = _font(26)
        for index, column in enumerate(table["columns"]):
            draw.text((column_x[index], y), column, fill="black", font=header_font)
        y += 36
        draw.line((70, y - 6, 950, y - 6), fill="black", width=2)
        for row in table["rows"]:
            for index, cell in enumerate(row):
                draw.text((column_x[index], y), str(cell), fill="black", font=_font(26))
            y += 40

    draw.text((70, 700), FOOTER, fill="black", font=_font(20))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def create_synthetic_documents(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Render all synthetic documents and write the ground truth file."""
    ground_truth: list[dict[str, Any]] = []

    for spec in SYNTHETIC_DOCUMENTS:
        path = _render(spec, output_dir / f"{spec['document_id']}.png")
        ground_truth.append(
            {
                "document_id": spec["document_id"],
                "image_path": str(path).replace("\\", "/"),
                "document_type": spec["document_type"],
                "fields": spec["fields"],
                "table": spec["table"],
                "question": spec["question"],
                "answer": spec["answer"],
            }
        )

    ground_truth_path = output_dir / GROUND_TRUTH_FILENAME
    payload = {"documents": ground_truth}
    ground_truth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"ground_truth_path": str(ground_truth_path), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = create_synthetic_documents(args.output_dir)
    for document in result["documents"]:
        print(f"Created: {document['image_path']}")
    print(f"Ground truth: {result['ground_truth_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
