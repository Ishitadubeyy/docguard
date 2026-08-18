"""CLI utility for manual Phase 1 OCR testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.loader import ingest_document
from src.ocr.dependencies import get_missing_dependencies
from src.ocr.pipeline import process_document
from src.preprocessing.config import PreprocessingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DocGuard Phase 1 ingestion, preprocessing, and OCR on a document.",
    )
    parser.add_argument("document", type=Path, help="Path to a PDF, PNG, JPG, or JPEG file")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate and print ingestion metadata without running OCR",
    )
    parser.add_argument(
        "--no-grayscale",
        action="store_true",
        help="Disable grayscale conversion during preprocessing",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Disable deskewing during preprocessing",
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Apply adaptive thresholding during preprocessing",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.metadata_only:
        missing = get_missing_dependencies()
        if missing:
            print("Missing dependencies required for OCR:", file=sys.stderr)
            for item in missing:
                print(f"  - {item}", file=sys.stderr)
            return 1

    try:
        if args.metadata_only:
            metadata = ingest_document(args.document)
            payload = {
                "document_id": metadata.document_id,
                "filename": metadata.filename,
                "file_type": metadata.file_type,
                "file_size_bytes": metadata.file_size_bytes,
                "page_count": metadata.page_count,
                "source_path": metadata.source_path,
            }
        else:
            config = PreprocessingConfig(
                grayscale=not args.no_grayscale,
                deskew=not args.no_deskew,
                threshold=args.threshold,
            )
            result = process_document(args.document, preprocessing_config=config)
            payload = result.to_dict()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    indent = 2 if args.pretty else None
    print(json.dumps(payload, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
