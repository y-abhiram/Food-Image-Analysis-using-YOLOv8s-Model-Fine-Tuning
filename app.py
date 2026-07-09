from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from food_analysis.analyzer import FoodImageAnalyzer
from food_analysis.visualization import annotate_image

DEFAULT_MODEL = Path("runs/detect/runs/food-analysis/yolov8-food-2/weights/best.pt")
DEFAULT_NUTRITION = Path("data/nutrition.csv")
DEFAULT_DATA = Path("data/food_detection.yaml")


st.set_page_config(page_title="Food Image Analysis", layout="wide")


@st.cache_resource
def load_analyzer(model_path: str, nutrition_path: str, confidence: float) -> FoodImageAnalyzer:
    return FoodImageAnalyzer(
        model_path=Path(model_path),
        nutrition_csv=Path(nutrition_path),
        confidence_threshold=confidence,
    )


@st.cache_data
def load_class_names(data_yaml: str) -> list[str]:
    path = Path(data_yaml)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def main() -> None:
    st.title("Food Image Analysis")
    st.caption("This model predicts only the food classes used during fine-tuning.")

    with st.sidebar:
        model_path = st.text_input("Model", value=str(DEFAULT_MODEL))
        nutrition_path = st.text_input("Nutrition CSV", value=str(DEFAULT_NUTRITION))
        confidence = st.slider("Confidence", min_value=0.10, max_value=0.90, value=0.75, step=0.05)
        class_names = load_class_names(str(DEFAULT_DATA))
        with st.expander("Trained classes"):
            st.write(", ".join(class_names))

    uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to run detection.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
        file.write(uploaded.getbuffer())
        image_path = Path(file.name)

    try:
        analyzer = load_analyzer(model_path, nutrition_path, confidence)
        result = analyzer.analyze(image_path)
    except Exception as exc:
        st.error(str(exc))
        return

    left, right = st.columns([1.2, 1])
    with left:
        annotated = annotate_image(image_path, result["items"])
        st.image(annotated, caption="Detected food items", use_container_width=True)

    with right:
        st.subheader(result["message"])
        if not result["items"]:
            st.warning("No food items detected.")
            return

        st.info("Predictions are limited to the trained classes shown in the sidebar.")

        st.dataframe(_items_table(result["items"]), use_container_width=True, hide_index=True)
        st.subheader("Total nutrition")
        st.dataframe(pd.DataFrame([result["total_nutrition"]]), use_container_width=True, hide_index=True)


def _items_table(items: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for item in items:
        nutrition = item["nutrition"]
        rows.append(
            {
                "food": item["label"],
                "confidence": item["confidence"],
                "portion": item["estimated_portion_factor"],
                "calories": nutrition["calories"],
                "protein_g": nutrition["protein_g"],
                "carbs_g": nutrition["carbs_g"],
                "fat_g": nutrition["fat_g"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
