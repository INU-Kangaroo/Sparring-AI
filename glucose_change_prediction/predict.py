from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from paths import MODELS_DIR
from service.curve import build_curve_points, soften_curve_near_peak


MODEL_PATH = MODELS_DIR / "service_model.joblib"
REQUIRED_BUNDLE_KEYS = {"features", "targets", "models"}
@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    # 서버 시작 후 반복 요청에서 같은 모델 번들 재사용
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    bundle = joblib.load(str(MODEL_PATH))
    missing = REQUIRED_BUNDLE_KEYS - set(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle missing keys: {sorted(missing)}")

    return bundle


def parse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    meal = payload["meal"]
    # 요청 필드를 학습 피처 이름과 순서에 맞춰 변환
    return {
        "meal_type": meal["mealType"],
        "total_kcal": float(meal["kcal"]),
        "total_carbs": float(meal["carbs"]),
        "effective_carbs": float(meal["carbs"] - meal["fiber"]),
        "total_protein": float(meal["protein"]),
        "total_fat": float(meal["fat"]),
        "total_fiber": float(meal["fiber"]),
        "baseline_glucose": float(payload["baselineGlucose"]),
        "sex": payload["sex"],
    }


def predict_meal_response(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = load_model_bundle()
    row = parse_payload(payload)
    frame = pd.DataFrame([row], columns=bundle["features"])
    models = bundle["models"]

    # 각 시점 변화량과 피크 정보를 개별 모델로 예측
    delta30 = float(models["delta_30"].predict(frame)[0])
    delta60 = float(models["delta_60"].predict(frame)[0])
    delta120 = float(models["delta_120"].predict(frame)[0])
    peak_delta = float(models["peak_delta"].predict(frame)[0])
    peak_minute = int(np.clip(np.rint(models["peak_minute"].predict(frame)[0]), 1, 120))

    # 피크가 고정 시점과 겹치면 해당 시점 값보다 작아지지 않게 보정
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

    # 응답 스키마에 맞춰 피크 정보와 곡선 포인트 조합
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
