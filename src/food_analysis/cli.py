from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import FoodImageAnalyzer
from .visualization import save_annotated_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze food items and nutrition from an image.")
    parser.add_argument("--image", required=True, type=Path, help="Path to .jpg, .jpeg, or .png image")
    parser.add_argument("--model", required=True, type=Path, help="Path to fine-tuned YOLO weights")
    parser.add_argument(
        "--nutrition",
        type=Path,
        default=Path("data/nutrition.csv"),
        help="Path to nutrition CSV",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    parser.add_argument("--save-annotated", type=Path, help="Optional path for annotated output image")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyzer = FoodImageAnalyzer(
        model_path=args.model,
        nutrition_csv=args.nutrition,
        confidence_threshold=args.conf,
    )
    result = analyzer.analyze(args.image)
    if args.save_annotated:
        save_annotated_image(args.image, args.save_annotated, result["items"])

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(result["message"])
    if not result["items"]:
        return

    rows = [
        [
            item["label"],
            item["confidence"],
            item["estimated_portion_factor"],
            item["nutrition"]["calories"],
            item["nutrition"]["protein_g"],
            item["nutrition"]["carbs_g"],
            item["nutrition"]["fat_g"],
        ]
        for item in result["items"]
    ]
    print(_format_table(rows, ["Food", "Conf", "Portion", "Calories", "Protein", "Carbs", "Fat"]))
    print("\nTotal nutrition")
    print(json.dumps(result["total_nutrition"], indent=2))


def _format_table(rows: list[list[object]], headers: list[str]) -> str:
    table = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table))
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(
        headers[index].ljust(widths[index]) for index in range(len(headers))
    ) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in table
    ]
    return "\n".join([header_line, separator, *body])


if __name__ == "__main__":
    main()
