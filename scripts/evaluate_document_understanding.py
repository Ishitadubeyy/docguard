"""Run the five document understanding tasks over the synthetic document set.

Results are written verbatim: model output is never corrected, and metrics are
computed only where ground truth exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_synthetic_documents import (
    DEFAULT_OUTPUT_DIR,
    GROUND_TRUTH_FILENAME,
    create_synthetic_documents,
)
from src.document_understanding.document_analyzer import DocumentAnalyzer
from src.document_understanding.evaluation import (
    classification_accuracy,
    field_accuracy,
    qa_exact_match,
    table_accuracy,
)
from src.document_understanding.extraction_schema import AnalysisTask
from src.document_understanding.vlm_config import BaselineVLMConfig

logger = logging.getLogger("evaluate_document_understanding")

DEFAULT_RESULTS_PATH = Path("experiments/experiment_02_document_understanding/results.json")


def load_ground_truth(documents_dir: Path) -> list[dict[str, Any]]:
    path = documents_dir / GROUND_TRUTH_FILENAME
    if not path.exists():
        create_synthetic_documents(documents_dir)
    return json.loads(path.read_text(encoding="utf-8"))["documents"]


def run_evaluation(
    documents_dir: Path = DEFAULT_OUTPUT_DIR,
    config: BaselineVLMConfig | None = None,
) -> dict[str, Any]:
    """Run every task against every synthetic document and score the output."""
    config = config or BaselineVLMConfig.from_env()
    analyzer = DocumentAnalyzer(config)
    documents = load_ground_truth(documents_dir)

    runs: list[dict[str, Any]] = []
    classification_pairs: list[tuple[Any, Any]] = []
    qa_pairs: list[tuple[Any, Any]] = []
    key_value_reports: list[dict[str, Any]] = []
    table_reports: list[dict[str, Any]] = []

    for document in documents:
        for task in AnalysisTask:
            logger.info("Running %s on %s", task.value, document["document_id"])
            record: dict[str, Any] = {
                "document_id": document["document_id"],
                "expected_document_type": document["document_type"],
                "task": task.value,
            }
            try:
                payload = analyzer.analyze(
                    image_path=document["image_path"],
                    task=task,
                    question=document["question"],
                    document_type=(
                        document["document_type"]
                        if task is AnalysisTask.KEY_VALUE_EXTRACTION
                        else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - recorded as a failed run
                record.update({"status": "FAILED", "error": str(exc)})
                runs.append(record)
                continue

            record.update(
                {
                    "status": "RUN",
                    "raw_output": payload["raw_response"],
                    "parsed_output": {
                        key: payload[key]
                        for key in ("document_type", "fields", "table", "summary", "answer")
                        if payload.get(key) not in (None, {})
                    },
                    "parse_success": payload["parse_success"],
                    "parse_errors": payload["parse_errors"],
                    "inference_time": payload["inference_time"],
                    "device": payload["device"],
                    "memory": payload["memory"],
                }
            )

            if task is AnalysisTask.CLASSIFICATION:
                classification_pairs.append((payload["document_type"], document["document_type"]))
                record["expected"] = document["document_type"]
            elif task is AnalysisTask.KEY_VALUE_EXTRACTION:
                scores = field_accuracy(payload["fields"], document["fields"])
                record["field_accuracy"] = scores
                key_value_reports.append({"document_id": document["document_id"], **scores})
            elif task is AnalysisTask.TABLE_EXTRACTION:
                scores = table_accuracy(payload["table"], document["table"]["rows"])
                record["table_accuracy"] = scores
                table_reports.append({"document_id": document["document_id"], **scores})
            elif task is AnalysisTask.QUESTION_ANSWERING:
                qa_pairs.append((payload["answer"], document["answer"]))
                record["expected"] = document["answer"]

            runs.append(record)

    executed = [run for run in runs if run["status"] == "RUN"]
    times = [run["inference_time"] for run in executed if run.get("inference_time")]

    total_fields = sum(report["total_fields"] for report in key_value_reports)
    correct_fields = sum(report["correct_fields"] for report in key_value_reports)
    total_cells = sum(report["total_cells"] for report in table_reports)
    correct_cells = sum(report["correct_cells"] for report in table_reports)
    expected_rows = sum(report["expected_rows"] for report in table_reports)
    correct_rows = sum(report["correct_rows"] for report in table_reports)

    return {
        "status": "RUN" if executed else "FAILED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model_id,
        "config": config.to_dict(),
        "documents_evaluated": [document["document_id"] for document in documents],
        "tasks_evaluated": [task.value for task in AnalysisTask],
        "metrics": {
            "classification": classification_accuracy(classification_pairs),
            "key_value_extraction": {
                "total_fields": total_fields,
                "correct_fields": correct_fields,
                "accuracy": round(correct_fields / total_fields, 4) if total_fields else None,
                "per_document": key_value_reports,
            },
            "table_extraction": {
                "expected_rows": expected_rows,
                "correct_rows": correct_rows,
                "row_accuracy": round(correct_rows / expected_rows, 4) if expected_rows else None,
                "total_cells": total_cells,
                "correct_cells": correct_cells,
                "cell_accuracy": round(correct_cells / total_cells, 4) if total_cells else None,
                "per_document": table_reports,
            },
            "question_answering": qa_exact_match(qa_pairs),
            "summarization": {
                "score": None,
                "note": "No numerical summarization quality score in this phase.",
            },
        },
        "inference_time_seconds": {
            "runs": len(times),
            "total": round(sum(times), 3) if times else None,
            "mean": round(sum(times) / len(times), 3) if times else None,
            "min": min(times) if times else None,
            "max": max(times) if times else None,
        },
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    results = run_evaluation(args.documents_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Status: {results['status']}")
    print(json.dumps(results["metrics"], indent=2))
    print(f"Written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
