import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from joblib import load

# ============================
# Paths
# ============================
MODEL_PATH = Path("models/meal_model_core_logcarb_clip_mono.joblib")
META_PATH = Path("models/meal_model_core_logcarb_clip_mono_meta.json")
CALIB_PATH = Path("models/calibration_delta60.json")


# ============================
# Time / formatting helpers
# ============================
def _parse_iso(ts: str) -> datetime:
    """
    Accepts ISO8601 string like:
      - 2026-01-27T03:00:00.000Z
      - 2026-01-27T03:00:00+00:00
    Returns timezone-aware UTC datetime.
    """
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _meal_type_to_label(code: int) -> str:
    mapping = {1: "breakfast", 2: "lunch", 3: "dinner", 4: "snack"}
    try:
        return mapping.get(int(code), "unknown")
    except Exception:
        return "unknown"


def _shape_fraction(t_min: int) -> float:
    """
    Heuristic curve for 0~60 min progression.
    """
    t = max(0, min(60, int(t_min)))
    x = t / 60.0
    return float(x ** 0.65)


def _clamp_glucose(g: float) -> float:
    """
    Safety clamp for glucose values.
    """
    return float(np.clip(float(g), 40.0, 400.0))


# ============================
# Cached loading
# ============================
@lru_cache(maxsize=1)
def load_model_meta_calib() -> Tuple[Any, Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Load model/meta/calibration once per process.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta not found: {META_PATH}")

    model = load(str(MODEL_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    calib = None
    if CALIB_PATH.exists():
        calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))

    return model, meta, calib


# Backward compatible name (some main.py used this)
def load_model_and_meta():
    model, meta, _calib = load_model_meta_calib()
    return model, meta


# ============================
# Calibration helpers
# ============================
def _carb_bin_label(carbs: float, calib: Optional[Dict[str, Any]]) -> str:
    """
    Map carbs to a bin label.
    Uses calibration file bins if present, else fallback bins.
    """
    if not calib:
        if carbs < 10: return "0-10"
        if carbs < 20: return "10-20"
        if carbs < 40: return "20-40"
        if carbs < 70: return "40-70"
        if carbs < 100: return "70-100"
        if carbs < 150: return "100-150"
        return "150+"

    bins = calib.get("carb_bins")
    labels = calib.get("carb_bin_labels")
    if not bins or not labels:
        return "unknown"

    idx = np.digitize([carbs], bins)[0] - 1
    idx = max(0, min(idx, len(labels) - 1))
    return str(labels[idx])


def _apply_calibration(
    pred_delta60_raw: float,
    *,
    carbs: float,
    is_insulin_user: int,
    calib: Optional[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    pred_delta60_raw -> calibrated_delta60
    Scaling per (insulin group, carb bin).
    """
    info = {"used": False, "group": None, "carb_bin": None, "scale": 1.0}

    d = float(pred_delta60_raw)
    carbs = float(carbs)
    is_insulin_user = int(is_insulin_user)

    if not calib:
        return d, info

    group_key = "insulin" if is_insulin_user == 1 else "non_insulin"
    group = calib.get("groups", {}).get(group_key)
    if not group:
        return d, info

    carb_bin = _carb_bin_label(carbs, calib)

    b = group.get("bins", {}).get(carb_bin)
    if isinstance(b, dict) and "scale" in b:
        scale = float(b["scale"])
    else:
        scale = float(group.get("fallback", {}).get("scale", 1.0))

    d2 = d * scale

    info.update({"used": True, "group": group_key, "carb_bin": carb_bin, "scale": scale})
    return float(d2), info


# ============================
# Guardrails (service policy)
# ============================
def _apply_guardrails_delta60(
    d: float,
    *,
    carbs: float,
    is_insulin_user: int,
) -> Tuple[float, Dict[str, Any]]:
    """
    Policy:
    - if carbs > 0, enforce minimum positive rise (no "flat" or negative)
      -> different for insulin vs non-insulin
    - final hard clip
    """
    info = {"before": float(d), "after": None, "rules_applied": []}
    d = float(d)
    carbs = float(carbs)
    is_insulin_user = int(is_insulin_user)

    if carbs > 0:
        if is_insulin_user == 1:
            # insulin users: higher minimum rise
            min_rise = min(30.0, 0.35 * carbs)   # 20g->7, 40g->14, 70g->24.5
            rule_name = "min_rise_if_carbs_insulin"
        else:
            # non-insulin: moderate minimum rise
            min_rise = min(80.0, 0.80 * carbs)   # 20g->4, 40g->8, 70g->14
            rule_name = "min_rise_if_carbs_non_insulin"

        if d < min_rise:
            info["rules_applied"].append(
                {"rule": rule_name, "min_delta": float(min_rise), "carbs": float(carbs)}
            )
            d = float(min_rise)

    HARD_LOW = -20.0
    HARD_HIGH = 80.0
    d2 = float(np.clip(d, HARD_LOW, HARD_HIGH))
    if d2 != d:
        info["rules_applied"].append({"rule": "hard_clip", "low": HARD_LOW, "high": HARD_HIGH})
    d = d2

    info["after"] = float(d)
    return d, info


# ============================
# Feature building
# ============================
def _build_feature_row(req: Dict[str, Any], meta: Dict[str, Any]) -> pd.DataFrame:
    numeric_feats: List[str] = meta["features"]["numeric"]
    cat_feats: List[str] = meta["features"]["categorical"]

    pre_glucose = float(req["glucose_history"][-1])
    carbs = float(req["carb_intake"])
    carbs_log = float(np.log1p(max(0.0, carbs)))

    dt0 = _parse_iso(req["timestamp"])
    hour = float(dt0.hour)
    weekday = float(dt0.weekday())

    steps = float(req.get("steps", 0) or 0)
    intensity = float(req.get("intensity", 0) or 0)

    meal_label = _meal_type_to_label(req.get("meal_type", -1))

    row = {}
    defaults_num = {
        "pre_glucose": pre_glucose,
        "carbs_log": carbs_log,
        "steps": steps,
        "intensity": intensity,
        "hour": hour,
        "weekday": weekday,
    }
    for f in numeric_feats:
        row[f] = defaults_num.get(f, 0.0)

    defaults_cat = {"meal_type": meal_label}
    for f in cat_feats:
        row[f] = defaults_cat.get(f, "unknown")

    return pd.DataFrame([row], columns=numeric_feats + cat_feats)


# ============================
# Main predict function (used by API)
# ============================
def predict_glucose(req: Dict[str, Any]) -> Dict[str, Any]:
    model, meta, calib = load_model_meta_calib()

    X = _build_feature_row(req, meta)

    pre_glucose = float(req["glucose_history"][-1])
    carbs = float(req.get("carb_intake", 0) or 0)
    dt0 = _parse_iso(req["timestamp"])
    is_insulin_user = int(req.get("is_insulin_user", 0) or 0)

    offset_main = int(req.get("prediction_offset_minutes", 60))
    offset_main = max(5, offset_main)

    # 1) raw model prediction (delta_60)
    raw_delta60 = float(model.predict(X)[0])

    # 2) calibration (data-driven scaling)
    cal_delta60, cal_info = _apply_calibration(
        raw_delta60, carbs=carbs, is_insulin_user=is_insulin_user, calib=calib
    )

    # 3) guardrails (service policy)
    delta60, guard = _apply_guardrails_delta60(
        cal_delta60, carbs=carbs, is_insulin_user=is_insulin_user
    )

    # main predicted_glucose at offset_main
    frac_main = _shape_fraction(offset_main)
    pred_glucose_main = _clamp_glucose(pre_glucose + (delta60 * frac_main))
    predicted_time = dt0 + timedelta(minutes=offset_main)

    # forecast points
    step_minutes = int(req.get("step_minutes", 5))
    horizon_minutes = int(req.get("horizon_minutes", 60))
    step_minutes = max(1, step_minutes)
    horizon_minutes = max(step_minutes, horizon_minutes)

    n_steps = horizon_minutes // step_minutes
    forecast = []
    peak_val = -1e18
    peak_point = None

    for s in range(1, n_steps + 1):
        off = s * step_minutes
        frac = _shape_fraction(off)
        g = _clamp_glucose(pre_glucose + (delta60 * frac))
        t = dt0 + timedelta(minutes=off)

        point = {
            "time": t.isoformat().replace("+00:00", "Z"),
            "predicted_glucose": float(g),
            "step": s,
            "offset_minutes": off,
        }
        forecast.append(point)

        if g > peak_val:
            peak_val = g
            peak_point = point

    # milestones
    milestones = {}
    for m in [10, 30, 60]:
        if m <= horizon_minutes and forecast:
            nearest = min(forecast, key=lambda p: abs(p["offset_minutes"] - m))
            milestones[str(m)] = nearest

    resp = {
        "predicted_glucose": float(pred_glucose_main),
        "prediction_offset_minutes": offset_main,
        "predicted_time": predicted_time.isoformat().replace("+00:00", "Z"),
        "forecast": forecast,
        "milestones": milestones if milestones else None,
        "peak": {
            "peak_glucose": float(peak_point["predicted_glucose"]),
            "peak_time": peak_point["time"],
            "peak_offset_minutes": int(peak_point["offset_minutes"]),
        } if peak_point else None,
    }

    # debug
    if bool(req.get("debug", False)):
        meal_label = _meal_type_to_label(req.get("meal_type", -1))
        resp["debug"] = {
            "pred_delta60_raw": raw_delta60,
            "pred_delta60_calibrated": cal_delta60,
            "pred_delta60_final": delta60,
            "pre_glucose": pre_glucose,
            "calibration": cal_info,
            "guardrails": guard,
            "features_used": {
                "meal_type": meal_label,
                "hour": dt0.hour,
                "weekday": dt0.weekday(),
                "steps": float(req.get("steps", 0) or 0),
                "intensity": float(req.get("intensity", 0) or 0),
                "carbs": carbs,
                "carbs_log": float(np.log1p(max(0.0, carbs))),
                "is_insulin_user": is_insulin_user,
            },
        }

    return resp