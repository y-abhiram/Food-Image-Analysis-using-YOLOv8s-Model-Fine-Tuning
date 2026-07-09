from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def annotate_image(image_path: Path, items: list[dict[str, object]]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for item in items:
        box = item["box"]
        label = str(item["label"])
        confidence = float(item["confidence"])
        calories = item["nutrition"]["calories"]
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
        caption = f"{label} {confidence:.2f} | {calories:.0f} kcal"

        draw.rectangle((x1, y1, x2, y2), outline=(0, 135, 90), width=4)
        text_box = draw.textbbox((x1, y1), caption)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, y1 - text_height - 8)
        draw.rectangle((x1, label_top, x1 + text_width + 8, y1), fill=(0, 135, 90))
        draw.text((x1 + 4, label_top + 2), caption, fill="white")

    return image


def save_annotated_image(image_path: Path, output_path: Path, items: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotate_image(image_path, items).save(output_path)
