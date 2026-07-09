from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: BoundingBox


@dataclass(frozen=True)
class Nutrition:
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float

    def scaled(self, factor: float) -> "Nutrition":
        return Nutrition(
            calories=self.calories * factor,
            protein_g=self.protein_g * factor,
            carbs_g=self.carbs_g * factor,
            fat_g=self.fat_g * factor,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "calories": round(self.calories, 2),
            "protein_g": round(self.protein_g, 2),
            "carbs_g": round(self.carbs_g, 2),
            "fat_g": round(self.fat_g, 2),
        }

    @staticmethod
    def zero() -> "Nutrition":
        return Nutrition(0.0, 0.0, 0.0, 0.0)

    def __add__(self, other: "Nutrition") -> "Nutrition":
        return Nutrition(
            calories=self.calories + other.calories,
            protein_g=self.protein_g + other.protein_g,
            carbs_g=self.carbs_g + other.carbs_g,
            fat_g=self.fat_g + other.fat_g,
        )
