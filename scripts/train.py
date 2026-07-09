from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO food detector.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/food_detection.yaml"),
        help="YOLO dataset YAML path",
    )
    parser.add_argument("--base-model", default="yolov8n.pt", help="Pre-trained YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example '0' for first CUDA GPU, 'cpu' for CPU, or omit for auto",
    )
    parser.add_argument("--project", default="runs/food-analysis")
    parser.add_argument("--name", default="yolov8-food")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install dependencies with: pip install -r requirements.txt") from exc

    model = YOLO(args.base_model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        plots=True,
    )
    save_dir = Path(results.save_dir)
    print(f"Training complete. Best weights are under {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
