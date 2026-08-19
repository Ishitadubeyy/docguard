"""CLI for the Phase 3 document understanding task layer.

Examples::

    python scripts/run_document_analysis.py --image data/synthetic/documents/synthetic_invoice.png \
        --task classification --pretty
    python scripts/run_document_analysis.py --image data/synthetic/documents/synthetic_invoice.png \
        --task question_answering --question "What is the total amount?" --pretty
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.document_understanding.document_analyzer import SUPPORTED_TASKS, DocumentAnalyzer
from src.document_understanding.extraction_schema import DocumentType
from src.document_understanding.vlm_config import BaselineVLMConfig
from src.document_understanding.vlm_interface import VLMModelError
from src.document_understanding.vlm_loader import get_missing_vlm_dependencies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DocGuard document understanding task.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a document image")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=list(SUPPORTED_TASKS),
        help="Document understanding task to run",
    )
    parser.add_argument("--question", type=str, default=None, help="Question for question_answering")
    parser.add_argument(
        "--document-type",
        type=str,
        default=None,
        choices=[doc_type.value for doc_type in DocumentType],
        help="Optional schema hint for key_value_extraction",
    )
    parser.add_argument(
        "--ocr-text-file",
        type=Path,
        default=None,
        help="Optional file with OCR text to pass to the model as supporting context",
    )
    parser.add_argument("--model-id", type=str, default=None, help="Override the VLM model id")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, auto)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Generation length override")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def build_config(args: argparse.Namespace) -> BaselineVLMConfig:
    base = BaselineVLMConfig.from_env()
    return BaselineVLMConfig(
        model_id=args.model_id or base.model_id,
        device=args.device or base.device,
        dtype="" if args.device else base.dtype,
        max_new_tokens=(
            args.max_new_tokens if args.max_new_tokens is not None else base.max_new_tokens
        ),
        temperature=base.temperature,
        cache_dir=base.cache_dir,
        trust_remote_code=base.trust_remote_code,
        local_files_only=base.local_files_only,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    missing = get_missing_vlm_dependencies()
    if missing:
        print("Missing dependencies required for VLM inference:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print("\nInstall with: pip install torch transformers accelerate", file=sys.stderr)
        return 1

    try:
        ocr_text = (
            args.ocr_text_file.read_text(encoding="utf-8") if args.ocr_text_file else None
        )
        analyzer = DocumentAnalyzer(build_config(args))
        payload = analyzer.analyze(
            image_path=args.image,
            task=args.task,
            question=args.question,
            document_type=args.document_type,
            ocr_result=ocr_text,
        )
    except VLMModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "\nIf the model is unavailable locally, download it first or set VLM_MODEL_ID "
            "to a cached checkpoint, for example HuggingFaceTB/SmolVLM-256M-Instruct.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
