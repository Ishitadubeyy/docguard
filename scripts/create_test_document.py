from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path("data/raw/sample_invoice.png")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (1400, 1000), "white")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("arial.ttf", 48)
        heading_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        title_font = heading_font = body_font = ImageFont.load_default()

    draw.text((80, 60), "SAMPLE INVOICE", fill="black", font=title_font)

    draw.text((80, 150), "Invoice Number: INV-2026-001", fill="black", font=body_font)
    draw.text((80, 200), "Date: 18 August 2026", fill="black", font=body_font)

    draw.text((80, 300), "Customer Information", fill="black", font=heading_font)
    draw.text((80, 360), "Name: Rahul Sharma", fill="black", font=body_font)
    draw.text((80, 410), "Company: Example Technologies Pvt Ltd", fill="black", font=body_font)

    draw.text((80, 520), "Item", fill="black", font=heading_font)
    draw.text((600, 520), "Quantity", fill="black", font=heading_font)
    draw.text((850, 520), "Amount", fill="black", font=heading_font)

    draw.text((80, 580), "AI Document Processing", fill="black", font=body_font)
    draw.text((600, 580), "2", fill="black", font=body_font)
    draw.text((850, 580), "₹20,000", fill="black", font=body_font)

    draw.text((80, 680), "Total Amount: ₹20,000", fill="black", font=heading_font)

    draw.text(
        (80, 820),
        "This is a synthetic test document for DocGuard AI.",
        fill="black",
        font=body_font,
    )

    image.save(OUTPUT)
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()