"""CLI for Phase 2 document understanding with VLM inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.document_understanding.config import TaskType, VLMConfig
from src.document_understanding.pipeline import run_document_understanding_pipeline
from src.document_understanding.prompts import get_baseline_prompt, list_baseline_prompts
from src.document_understanding.vlm_config import BaselineVLMConfig
from src.document_understanding.vlm_inference import run_vlm_inference
from src.document_understanding.vlm_loader import get_missing_vlm_dependencies
from src.ingestion.validator import validate_document_path
from src.ocr.dependencies import get_missing_dependencies
from src.preprocessing.config import PreprocessingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DocGuard Phase 2 document understanding with OCR + VLM.",
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to a PDF, PNG, JPG, or JPEG file")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Free-form prompt. Runs the baseline VLM directly on the image (no OCR pipeline).",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        choices=list_baseline_prompts(),
        help="Reusable baseline prompt to use instead of --prompt",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question for --prompt-template qa",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override the configured Hugging Face VLM model id",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the generation length for baseline prompt mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override for baseline prompt mode (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=TaskType.DOCUMENT_UNDERSTANDING.value,
        choices=[task.value for task in TaskType],
        help="Document understanding task to run",
    )
    parser.add_argument("--page", type=int, default=None, help="Optional page number for multi-page PDFs")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument("--no-grayscale", action="store_true", help="Disable grayscale preprocessing")
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskew preprocessing")
    parser.add_argument("--threshold", action="store_true", help="Enable adaptive threshold preprocessing")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    baseline_mode = bool(args.prompt or args.prompt_template)
    if baseline_mode:
        return _run_baseline(args)

    missing_ocr = get_missing_dependencies()
    if missing_ocr:
        print("Missing dependencies required for OCR:", file=sys.stderr)
        for item in missing_ocr:
            print(f"  - {item}", file=sys.stderr)
        return 1

    missing_vlm = get_missing_vlm_dependencies()
    if missing_vlm:
        print("Missing dependencies required for VLM inference:", file=sys.stderr)
        for item in missing_vlm:
            print(f"  - {item}", file=sys.stderr)
        print(
            "\nInstall with: pip install torch transformers accelerate",
            file=sys.stderr,
        )
        return 1

    try:
        validate_document_path(args.image)
        task = TaskType.from_string(args.task)
        config = VLMConfig.from_env()
        preprocessing_config = PreprocessingConfig(
            grayscale=not args.no_grayscale,
            deskew=not args.no_deskew,
            threshold=args.threshold,
        )

        payload = run_document_understanding_pipeline(
            args.image,
            task,
            vlm_config=config,
            preprocessing_config=preprocessing_config,
            page_number=args.page,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "\nIf the model is not available locally, download it first or set VLM_MODEL_ID "
            "to a cached checkpoint. Example:\n"
            "  set VLM_MODEL_ID=HuggingFaceTB/SmolVLM-256M-Instruct",
            file=sys.stderr,
        )
        return 1

    indent = 2 if args.pretty else None
    print(json.dumps(payload, indent=indent))
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    """Run the image-only baseline VLM path (no OCR, no JSON schema)."""
    missing_vlm = get_missing_vlm_dependencies()
    if missing_vlm:
        print("Missing dependencies required for VLM inference:", file=sys.stderr)
        for item in missing_vlm:
            print(f"  - {item}", file=sys.stderr)
        print("\nInstall with: pip install torch transformers accelerate", file=sys.stderr)
        return 1

    try:
        prompt = (
            args.prompt
            if args.prompt
            else get_baseline_prompt(args.prompt_template, question=args.question)
        )
        base = BaselineVLMConfig.from_env()
        config = BaselineVLMConfig(
            model_id=args.model_id or base.model_id,
            device=args.device or base.device,
            dtype="" if args.device else base.dtype,
            max_new_tokens=args.max_new_tokens or base.max_new_tokens,
            temperature=base.temperature,
            cache_dir=base.cache_dir,
            trust_remote_code=base.trust_remote_code,
            local_files_only=base.local_files_only,
        )
        result = run_vlm_inference(args.image, prompt, config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "\nIf the model is unavailable locally, download it first or set VLM_MODEL_ID "
            "to a cached checkpoint, for example HuggingFaceTB/SmolVLM-256M-Instruct.",
            file=sys.stderr,
        )
        return 1

    result["prompt"] = prompt
    print(json.dumps(result, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
