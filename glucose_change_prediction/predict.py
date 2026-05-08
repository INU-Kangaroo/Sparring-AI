from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from paths import MODELS_DIR
from service.curve import build_curve_points, soften_curve_near_peak


MODEL_PATH = MODELS_DIR / "service_model.joblib"
REQUIRED_BUNDLE_KEYS = {"features", "targets", "models"}
ALLOWED_MEAL_TYPES = {"breakfast", "lunch", "dinner"}


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    bundle = joblib.load(str(MODEL_PATH))
    missing = REQUIRED_BUNDLE_KEYS - set(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle missing keys: {sorted(missing)}")

    return bundle


def parse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    baseline_glucose = float(payload["baselineGlucose"])
    sex = str(payload["sex"]).strip().upper()
    if sex not in {"M", "F"}:
        raise ValueError("sex must be 'M' or 'F'")

    meal = payload["meal"]
    total_carbs = float(meal["carbs"])
    total_protein = float(meal["protein"])
    total_fat = float(meal["fat"])
    total_fiber = float(meal["fiber"])
    total_kcal = float(meal["kcal"])
    meal_type = str(meal["mealType"]).strip().lower()

    if meal_type not in ALLOWED_MEAL_TYPES:
        raise ValueError("mealType must be one of: breakfast, lunch, dinner")
    if total_fiber > total_carbs:
        raise ValueError("fiber cannot be greater than carbs")

    return {
        "meal_type": meal_type,
        "total_kcal": total_kcal,
        "total_carbs": total_carbs,
        "effective_carbs": max(total_carbs - total_fiber, 0.0),
        "total_protein": total_protein,
        "total_fat": total_fat,
        "total_fiber": total_fiber,
        "baseline_glucose": baseline_glucose,
        "sex": sex,
    }


def predict_meal_response(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = load_model_bundle()
    row = parse_payload(payload)
    frame = pd.DataFrame([row], columns=bundle["features"])
    models = bundle["models"]

    delta30 = float(models["delta_30"].predict(frame)[0])
    delta60 = float(models["delta_60"].predict(frame)[0])
    delta120 = float(models["delta_120"].predict(frame)[0])
    peak_delta = float(models["peak_delta"].predict(frame)[0])
    peak_minute = int(np.clip(np.rint(models["peak_minute"].predict(frame)[0]), 1, 120))

    if peak_minute == 30:
        peak_delta = max(peak_delta, delta30)
    elif peak_minute == 60:
        peak_delta = max(peak_delta, delta60)
    elif peak_minute == 120:
        peak_delta = max(peak_delta, delta120)

    delta30, delta60, delta120 = soften_curve_near_peak(
        peak_minute=peak_minute,
        delta30=delta30,
        delta60=delta60,
        delta120=delta120,
        peak_delta=peak_delta,
    )

    delta30 = round(delta30, 1)
    delta60 = round(delta60, 1)
    delta120 = round(delta120, 1)
    peak_delta = round(peak_delta, 1)

    return {
        "peakDelta": peak_delta,
        "peakMinute": peak_minute,
        "curve": build_curve_points(
            delta30=delta30,
            delta60=delta60,
            delta120=delta120,
            peak_delta=peak_delta,
            peak_minute=peak_minute,
        ),
    }
