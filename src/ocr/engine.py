"""Tesseract-based OCR engine."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.ocr.dependencies import require_ocr_dependencies
from src.ocr.models import OCRBlock, OCRPageResult


class TesseractOCREngine:
    """Run OCR on a single image using Tesseract via pytesseract."""

    def __init__(self, lang: str = "eng", min_confidence: float = 0.0) -> None:
        self.lang = lang
        self.min_confidence = min_confidence

    def recognize(self, image: np.ndarray, page_number: int) -> OCRPageResult:
        """Recognize text in an image and return structured page output."""
        require_ocr_dependencies()

        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            output_type=Output.DICT,
        )

        line_groups: dict[tuple[int, int, int, int], list[dict[str, int | str]]] = (
            defaultdict(list)
        )

        for index, text in enumerate(data["text"]):
            cleaned = str(text).strip()
            if not cleaned:
                continue

            confidence_raw = float(data["conf"][index])
            if confidence_raw < 0:
                continue

            confidence = confidence_raw / 100.0
            if confidence < self.min_confidence:
                continue

            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
                int(data["page_num"][index]),
            )
            line_groups[key].append(
                {
                    "text": cleaned,
                    "left": int(data["left"][index]),
                    "top": int(data["top"][index]),
                    "width": int(data["width"][index]),
                    "height": int(data["height"][index]),
                    "confidence": confidence,
                }
            )

        blocks: list[OCRBlock] = []
        page_lines: list[str] = []

        for key in sorted(line_groups):
            words = line_groups[key]
            line_text = " ".join(str(word["text"]) for word in words)
            left = min(int(word["left"]) for word in words)
            top = min(int(word["top"]) for word in words)
            right = max(int(word["left"]) + int(word["width"]) for word in words)
            bottom = max(int(word["top"]) + int(word["height"]) for word in words)
            avg_confidence = sum(float(word["confidence"]) for word in words) / len(words)

            blocks.append(
                OCRBlock(
                    text=line_text,
                    bbox=[left, top, right, bottom],
                    confidence=round(avg_confidence, 4),
                )
            )
            page_lines.append(line_text)

        return OCRPageResult(
            page_number=page_number,
            text="\n".join(page_lines),
            blocks=blocks,
        )
