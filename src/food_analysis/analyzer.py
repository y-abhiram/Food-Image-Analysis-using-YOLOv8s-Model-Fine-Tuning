from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .nutrition import NutritionDatabase, normalize_label
from .schemas import BoundingBox, Detection, Nutrition

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}


class FoodImageAnalyzer:
    def __init__(
        self,
        model_path: Path,
        nutrition_csv: Path,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nutrition_db = NutritionDatabase(nutrition_csv)
        self.model = self._load_model(model_path)

    def analyze(self, image_path: Path) -> dict[str, Any]:
        self._validate_image(image_path)
        image_width, image_height = Image.open(image_path).size
        image_area = float(image_width * image_height)

        detections = self._detect(image_path)
        food_items = []
        total = Nutrition.zero()

        for detection in detections:
            base_nutrition = self.nutrition_db.get(detection.label)
            if base_nutrition is None:
                continue

            # Area is a rough proxy for portion size. Clamp to avoid wild estimates
            # from unusually close or tiny detections.
            area_ratio = detection.box.area / image_area if image_area else 0.0
            portion_factor = min(max(area_ratio / 0.18, 0.5), 2.0)
            nutrition = base_nutrition.scaled(portion_factor)
            total += nutrition

            food_items.append(
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "box": {
                        "x1": round(detection.box.x1, 2),
                        "y1": round(detection.box.y1, 2),
                        "x2": round(detection.box.x2, 2),
                        "y2": round(detection.box.y2, 2),
                    },
                    "estimated_portion_factor": round(portion_factor, 2),
                    "nutrition": nutrition.as_dict(),
                }
            )

        if not food_items:
            return {
                "image": str(image_path),
                "message": "No food items detected.",
                "items": [],
                "total_nutrition": Nutrition.zero().as_dict(),
            }

        return {
            "image": str(image_path),
            "message": "Food items detected.",
            "items": food_items,
            "total_nutrition": total.as_dict(),
        }

    @staticmethod
    def _load_model(model_path: Path) -> Any:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}. Train the model first or pass --model."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install dependencies with: pip install -r requirements.txt") from exc
        return YOLO(str(model_path))

    def _detect(self, image_path: Path) -> list[Detection]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue
                class_id = int(box.cls[0])
                label = normalize_label(str(names[class_id]))
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )
        return _remove_duplicate_detections(detections)

    @staticmethod
    def _validate_image(image_path: Path) -> None:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_FORMATS))
            raise ValueError(f"Unsupported image format. Supported formats: {supported}")
        with Image.open(image_path) as image:
            image.verify()


def _remove_duplicate_detections(detections: list[Detection]) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for existing in kept:
            if detection.label != existing.label:
                continue
            if _iou(detection.box, existing.box) >= 0.25 or _overlap_ratio(detection.box, existing.box) >= 0.65:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / union if union else 0.0


def _overlap_ratio(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller_area = min(first.area, second.area)
    return intersection / smaller_area if smaller_area else 0.0
