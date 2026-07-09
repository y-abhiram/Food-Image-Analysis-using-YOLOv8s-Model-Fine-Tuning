from __future__ import annotations

import csv
from pathlib import Path

from .schemas import Nutrition


class NutritionDatabase:
    def __init__(self, nutrition_csv: Path) -> None:
        self.records = self._load(nutrition_csv)

    def get(self, label: str) -> Nutrition | None:
        return self.records.get(normalize_label(label))

    @staticmethod
    def _load(path: Path) -> dict[str, Nutrition]:
        if not path.exists():
            raise FileNotFoundError(f"Nutrition file not found: {path}")

        records: dict[str, Nutrition] = {}
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = normalize_label(row["label"])
                records[label] = Nutrition(
                    calories=float(row["calories"]),
                    protein_g=float(row["protein_g"]),
                    carbs_g=float(row["carbs_g"]),
                    fat_g=float(row["fat_g"]),
                )
        return records


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")
