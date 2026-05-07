from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
import math


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_catboost_service_shape_with_sex.joblib"


@lru_cache(maxsize=1)
def load_model_bundle() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    bundle = joblib.load(str(MODEL_PATH))

    required_keys = {"features", "targets", "models"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle missing keys: {sorted(missing)}")

    return bundle


def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
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

    allowed_meal_types = {"breakfast", "lunch", "dinner"}
    if meal_type not in allowed_meal_types:
        raise ValueError("mealType must be one of: breakfast, lunch, dinner")

    if total_fiber > total_carbs:
        raise ValueError("fiber cannot be greater than carbs")

    effective_carbs = max(total_carbs - total_fiber, 0.0)

    return {
        "meal_type": meal_type,
        "total_kcal": total_kcal,
        "total_carbs": total_carbs,
        "effective_carbs": effective_carbs,
        "total_protein": total_protein,
        "total_fat": total_fat,
        "total_fiber": total_fiber,
        "baseline_glucose": baseline_glucose,
        "sex": sex,
    }


def _smooth_reduction(value: float, scale: float, max_reduction: float) -> float:
    reduction = max_reduction * (1.0 - math.exp(-max(value, 0.0) / scale))
    return 1.0 - reduction


def _smooth_increase(value: float, scale: float, max_increase: float) -> float:
    increase = max_increase * (1.0 - math.exp(-max(value, 0.0) / scale))
    return 1.0 + increase


def _smooth_peak_cap(carbs: float) -> float:
    return 22.0 + 38.0 * (1.0 - math.exp(-max(carbs, 0.0) / 55.0))


def _estimate_peak_minute(
    meal_type: str,
    carbs: float,
    protein: float,
    fat: float,
    fiber: float,
    delta30: float,
    delta60: float,
    delta120: float,
    peak_delta: float,
) -> int:
    score = 0.0

    if meal_type == "breakfast":
        score -= 0.18
    elif meal_type == "dinner":
        score += 0.10

    score += min(fat / 20.0, 1.2) * 0.32
    score += min(protein / 30.0, 1.0) * 0.12
    score += min(fiber / 10.0, 1.0) * 0.20

    if carbs <= 25:
        score -= 0.20
    elif carbs >= 70:
        score += 0.08

    if delta60 > 0:
        early_ratio = delta30 / max(delta60, 1e-6)
        if early_ratio >= 0.78:
            score -= 0.32
        elif early_ratio >= 0.65:
            score -= 0.18
        elif early_ratio <= 0.35:
            score += 0.08

    gap_60_peak = peak_delta - delta60
    if gap_60_peak > 8:
        score += 0.22
    elif gap_60_peak > 4:
        score += 0.10
    elif gap_60_peak < 1.5:
        score -= 0.08

    if delta60 > 0:
        late_ratio = delta120 / max(delta60, 1e-6)
        if late_ratio > 0.92:
            score += 0.12
        elif late_ratio < 0.78:
            score -= 0.10

    candidate_scores = {
        60: abs(score - 0.00),
        75: abs(score - 0.42),
        90: abs(score - 1.30),
        105: abs(score - 2.10),
    }

    return min(candidate_scores, key=candidate_scores.get)


def _align_peak_with_curve(
    peak_minute: int,
    delta30: float,
    delta60: float,
    delta120: float,
    peak_delta: float,
) -> float:
    if peak_minute == 30:
        return delta30
    elif peak_minute == 60:
        return delta60
    elif peak_minute == 120:
        return delta120

    if peak_minute == 75:
        lower = delta60
        upper = max(delta60 + 8.0, delta60 * 1.16)
        return min(max(peak_delta, lower), upper)

    if peak_minute == 90:
        lower = delta60
        upper = max(delta60 + 10.0, delta60 * 1.20)
        return min(max(peak_delta, lower), upper)

    if peak_minute == 105:
        lower = max(delta60, delta120)
        upper = max(lower + 6.0, lower * 1.12)
        return min(max(peak_delta, lower), upper)

    return peak_delta


def _finalize_peak_minute(
    peak_minute: int,
    delta30: float,
    delta60: float,
    delta120: float,
    peak_delta: float,
) -> int:
    if peak_minute > 60 and abs(peak_delta - delta60) <= 0.5:
        return 60

    if peak_minute > 30 and abs(peak_delta - delta30) <= 0.5:
        return 30

    if abs(peak_delta - delta120) <= 0.5:
        return 120

    return peak_minute


def _apply_postprocess_rules(
    meal_type: str,
    carbs: float,
    fiber: float,
    protein: float,
    fat: float,
    delta30: float,
    delta60: float,
    delta120: float,
    peak_delta: float,
):
    peak_delta = min(peak_delta, delta60 + 12.0)
    peak_cap = _smooth_peak_cap(carbs)
    peak_delta = min(peak_delta, peak_cap)

    low_carb_factor = _smooth_reduction(
        max(30.0 - carbs, 0.0),
        scale=12.0,
        max_reduction=0.12,
    )
    delta30 *= low_carb_factor
    delta60 *= low_carb_factor
    delta120 *= low_carb_factor
    peak_delta *= low_carb_factor

    carb_boost = _smooth_increase(
        max(carbs - 25.0, 0.0),
        scale=35.0,
        max_increase=0.12,
    )
    delta30 *= carb_boost
    delta60 *= carb_boost
    delta120 *= carb_boost
    peak_delta *= carb_boost

    fiber_factor = _smooth_reduction(fiber, scale=6.0, max_reduction=0.10)
    protein_factor = _smooth_reduction(protein, scale=24.0, max_reduction=0.07)
    fat_factor = _smooth_reduction(fat, scale=14.0, max_reduction=0.06)

    delta60 *= fiber_factor * protein_factor * fat_factor
    delta120 *= fiber_factor * protein_factor * fat_factor
    peak_delta *= fiber_factor * protein_factor * fat_factor

    early_factor = (
        _smooth_reduction(fat, scale=12.0, max_reduction=0.10)
        * _smooth_reduction(protein, scale=22.0, max_reduction=0.06)
        * _smooth_reduction(fiber, scale=5.0, max_reduction=0.08)
    )
    delta30 *= early_factor

    if meal_type == "breakfast":
        delta30 *= 0.95
        if delta60 > 0:
            delta30 = min(delta30, delta60 * 0.96)

    elif meal_type == "lunch":
        delta60 *= 1.03
        peak_delta *= 1.02

    elif meal_type == "dinner":
        delta60 *= 1.02
        delta120 *= 0.95
        peak_delta *= 1.03

    if meal_type == "breakfast":
        delta120 = min(delta120, peak_delta * 0.97)
        delta120 = min(delta120, delta60 * 0.96)
    else:
        delta120 = min(delta120, delta60 * 0.94)

    if peak_delta < delta60:
        peak_delta = delta60

    if meal_type == "breakfast" and delta30 > delta60:
        delta30 = min(delta30, delta60 * 0.98)

    delta30 = max(delta30, -5.0)
    delta60 = max(delta60, -5.0)
    delta120 = max(delta120, -10.0)
    peak_delta = max(peak_delta, 0.0)

    return delta30, delta60, delta120, peak_delta


def _build_curve(
    delta30: float,
    delta60: float,
    delta120: float,
    peak_delta: float,
    peak_minute: int,
):
    points = {
        0: 0.0,
        30: delta30,
        60: delta60,
        120: delta120,
    }

    if peak_minute not in points:
        points[peak_minute] = peak_delta
    else:
        points[peak_minute] = max(points[peak_minute], peak_delta)

    curve = [
        {"minute": minute, "delta": round(points[minute], 1)}
        for minute in sorted(points.keys())
    ]
    return curve


def predict_meal_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    bundle = load_model_bundle()
    features = bundle["features"]
    models = bundle["models"]

    row = _validate_payload(payload)
    X = pd.DataFrame([row], columns=features)

    raw_delta30 = float(models["delta_30"].predict(X)[0])
    raw_delta60 = float(models["delta_60"].predict(X)[0])
    raw_delta120 = float(models["delta_120"].predict(X)[0])
    raw_peak_delta = float(models["peak_delta"].predict(X)[0])

    delta30, delta60, delta120, peak_delta = _apply_postprocess_rules(
        meal_type=row["meal_type"],
        carbs=row["total_carbs"],
        fiber=row["total_fiber"],
        protein=row["total_protein"],
        fat=row["total_fat"],
        delta30=raw_delta30,
        delta60=raw_delta60,
        delta120=raw_delta120,
        peak_delta=raw_peak_delta,
    )

    peak_minute = _estimate_peak_minute(
        meal_type=row["meal_type"],
        carbs=row["total_carbs"],
        protein=row["total_protein"],
        fat=row["total_fat"],
        fiber=row["total_fiber"],
        delta30=delta30,
        delta60=delta60,
        delta120=delta120,
        peak_delta=peak_delta,
    )

    peak_delta = _align_peak_with_curve(
        peak_minute=peak_minute,
        delta30=delta30,
        delta60=delta60,
        delta120=delta120,
        peak_delta=peak_delta,
    )

    peak_minute = _finalize_peak_minute(
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

    curve = _build_curve(
        delta30=delta30,
        delta60=delta60,
        delta120=delta120,
        peak_delta=peak_delta,
        peak_minute=peak_minute,
    )

    return {
        "delta30": delta30,
        "delta60": delta60,
        "peakDelta": peak_delta,
        "peakMinute": peak_minute,
        "curve": curve,
    }