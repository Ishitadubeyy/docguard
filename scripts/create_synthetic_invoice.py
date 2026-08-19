"""Generate a synthetic invoice image for VLM baseline experiments.

The content is entirely fictional: no real identities, accounts, or scraped
documents are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_OUTPUT = Path("data/synthetic/synthetic_invoice.png")

LINES = [
    ("INVOICE", 48),
    ("", 0),
    ("Invoice Number: INV-001", 32),
    ("Customer: Alice Example", 32),
    ("Item: Laptop", 32),
    ("Quantity: 1", 32),
    ("Amount: 50000", 32),
    ("Date: 2026-01-01", 32),
    ("", 0),
    ("Synthetic document generated for DocGuard AI testing.", 24),
]


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


def create_synthetic_invoice(output: Path = DEFAULT_OUTPUT) -> Path:
    """Write a synthetic invoice PNG and return its path."""
    output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)

    y = 60
    for text, size in LINES:
        if not text:
            y += 30
            continue
        draw.text((70, y), text, fill="black", font=_font(size))
        y += size + 22

    image.save(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = create_synthetic_invoice(args.output)
    print(f"Created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
