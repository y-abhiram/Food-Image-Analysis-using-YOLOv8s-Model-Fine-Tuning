from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


KEEP_CLASSES = [
    "boiled egg",
    "idly",
    "medu vadai",
    "nei satham",
    "paal kolukattai",
    "pidi kolukattai",
    "poorna kolukattai",
    "puthina chutney",
    "rice",
    "roti",
]

MIN_BOX_AREA = 0.001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter Roboflow YOLO/segmentation labels into a clean YOLO detection dataset."
    )
    parser.add_argument("--source", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("datasets/food_multi_filtered"))
    parser.add_argument(
        "--preset",
        choices=["filtered", "all"],
        default="filtered",
        help="'filtered' keeps stronger assignment classes; 'all' keeps every source class",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_names = _load_inline_names(args.source / "data.yaml")
    keep_classes = source_names if args.preset == "all" else KEEP_CLASSES
    source_to_target = {
        source_names.index(name): target_id
        for target_id, name in enumerate(keep_classes)
        if name in source_names
    }

    if len(source_to_target) != len(keep_classes):
        missing = [name for name in keep_classes if name not in source_names]
        raise ValueError(f"Missing expected classes in source data.yaml: {missing}")

    if args.output.exists():
        shutil.rmtree(args.output)

    copied_images = 0
    copied_labels = 0
    kept_objects = 0
    empty_negative_images = 0

    for split in ["train", "valid", "test"]:
        image_out = args.output / split / "images"
        label_out = args.output / split / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        source_image_dir = args.source / split / "images"
        source_label_dir = args.source / split / "labels"
        if not source_image_dir.exists():
            continue

        for image_path in sorted(source_image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = source_label_dir / f"{image_path.stem}.txt"
            output_rows = []

            if label_path.exists():
                for row in label_path.read_text(encoding="utf-8").splitlines():
                    parsed = _convert_label_row(row, source_to_target)
                    if parsed:
                        output_rows.append(parsed)

            if not output_rows and split != "train":
                continue

            shutil.copy2(image_path, image_out / image_path.name)
            (label_out / f"{image_path.stem}.txt").write_text(
                "\n".join(output_rows) + ("\n" if output_rows else ""),
                encoding="utf-8",
            )
            copied_images += 1
            copied_labels += 1
            kept_objects += len(output_rows)
            if not output_rows:
                empty_negative_images += 1

    _write_yaml(args.output / "data.yaml", args.output, keep_classes)
    print(f"Created: {args.output}")
    print(f"Images: {copied_images}")
    print(f"Label files: {copied_labels}")
    print(f"Objects: {kept_objects}")
    print(f"Empty negative images: {empty_negative_images}")


def _load_inline_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("names:"):
            import ast

            return ast.literal_eval(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not find inline names list in {path}")


def _convert_label_row(row: str, source_to_target: dict[int, int]) -> str | None:
    parts = row.split()
    if len(parts) < 5:
        return None

    source_class = int(float(parts[0]))
    if source_class not in source_to_target:
        return None

    values = [float(value) for value in parts[1:]]
    if len(values) == 4:
        x_center, y_center, width, height = values
    else:
        xs = values[0::2]
        ys = values[1::2]
        if not xs or not ys:
            return None
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

    x_center = _clamp(x_center)
    y_center = _clamp(y_center)
    width = _clamp(width)
    height = _clamp(height)
    if width <= 0 or height <= 0:
        return None
    if width * height < MIN_BOX_AREA:
        return None

    target_class = source_to_target[source_class]
    return f"{target_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _write_yaml(path: Path, dataset_root: Path, class_names: list[str]) -> None:
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    path.write_text(
        f"""path: {dataset_root.as_posix()}
train: train/images
val: valid/images
test: test/images

names:
{names_block}
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
