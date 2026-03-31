import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from joblib import load


REALSHAPE_MODEL_PATH = Path("models/glucose_service_like_models_v4_realshape.joblib")
REALSHAPE_META_PATH = Path("models/glucose_service_like_models_v4_realshape_meta.json")
BASELINE_MODEL_PATH = Path("models/glucose_service_like_models_v4_nonmeal_residual.joblib")
BASELINE_META_PATH = Path("models/glucose_service_like_models_v4_nonmeal_residual_meta.json")


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _normalize_measurement_label(v: Any) -> str:
    allowed = {"공복", "식전", "식후", "기타"}
    if v in allowed:
        return str(v)
    return "기타"


def _normalize_insulin_type(v: Any) -> str:
    if v is None:
        return "Unknown"
    s = str(v).strip()
    if not s:
        return "Unknown"
    return s


def _has_named_insulin_type(v: Any) -> bool:
    return _normalize_insulin_type(v).lower() not in {"unknown", "none", ""}


def _meal_type_to_label(code: int) -> str:
    mapping = {
        0: "unknown",
        1: "breakfast",
        2: "lunch",
        3: "dinner",
        4: "snack",
    }
    try:
        return mapping.get(int(code), "unknown")
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def load_model_meta() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_path = REALSHAPE_MODEL_PATH if REALSHAPE_MODEL_PATH.exists() else BASELINE_MODEL_PATH
    meta_path = REALSHAPE_META_PATH if REALSHAPE_META_PATH.exists() else BASELINE_META_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    model_bundle = load(str(model_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model_bundle, meta


def _select_model_family(models: Dict[str, Any], req: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(models, dict) or "segment_models" not in models:
        graph_models = models
        if isinstance(models, dict) and "graph_models" in models:
            graph_models = models.get("graph_models", {})
        return graph_models

    carbs_raw = float(req.get("carb_intake", 0) or 0)
    segment_key = "meal" if carbs_raw > 0 else "non_meal"
    segment_bundle = models["segment_models"].get(segment_key)
    if segment_bundle is None:
        fallback_key = "non_meal" if "non_meal" in models["segment_models"] else next(iter(models["segment_models"]))
        segment_bundle = models["segment_models"][fallback_key]

    return segment_bundle.get("graph_models", {})


def _predict_delta_with_model_entry(
    model_entry: Any,
    X: pd.DataFrame,
    req: Dict[str, Any],
) -> float:
    if hasattr(model_entry, "predict"):
        return float(model_entry.predict(X)[0])

    if not isinstance(model_entry, dict):
        raise ValueError("Unsupported model entry format.")

    if "base" not in model_entry:
        raise ValueError("Model entry dict must contain 'base'.")

    pred_delta = float(model_entry["base"].predict(X)[0])
    carbs_raw = float(req.get("carb_intake", 0) or 0)
    current_glucose = float(X.iloc[0].get("current_glucose", 0.0))
    bolus_dose_60m = float(X.iloc[0].get("bolus_dose_60m", 0.0))

    non_meal_residual = model_entry.get("non_meal_residual")
    if non_meal_residual is not None:
        apply_non_meal = carbs_raw <= float(model_entry.get("non_meal_max_carbs", 0.0))
        if apply_non_meal:
            min_glucose = float(model_entry.get("non_meal_min_glucose", 0.0))
            min_bolus_60m = float(model_entry.get("non_meal_min_bolus_60m", 0.0))
            if current_glucose >= min_glucose or bolus_dose_60m >= min_bolus_60m:
                pred_delta += float(non_meal_residual.predict(X)[0])

    meal_residual = model_entry.get("meal_residual")
    if meal_residual is not None:
        residual_min_carbs = float(model_entry.get("residual_min_carbs", 5.0))
        if carbs_raw >= residual_min_carbs:
            pred_delta += float(meal_residual.predict(X)[0])

    return pred_delta


def _extract_glucose_points(req: Dict[str, Any]) -> List[Tuple[datetime, float, str]]:
    gh = req.get("glucose_history", [])
    if not gh:
        raise ValueError("glucoseHistory must contain at least 1 item")

    points = []
    for x in gh:
        g = float(x["glucose_level"])
        dt = _parse_iso(x["measured_at"])
        label = _normalize_measurement_label(x.get("measurement_label", "기타"))
        points.append((dt, g, label))

    points.sort(key=lambda x: x[0])
    return points


def _extract_insulin_event_features(req: Dict[str, Any], dt0: datetime) -> Dict[str, float | str]:
    events = req.get("insulin_events", []) or []
    bolus_30 = 0.0
    bolus_60 = 0.0
    bolus_120 = 0.0
    latest_basal_daily = None
    latest_basal_ts = None
    latest_bolus_type = None
    latest_bolus_ts = None
    latest_any_type = None
    latest_any_ts = None

    for ev in events:
        try:
            et = str(ev.get("event_type", ev.get("eventType", ""))).strip().lower()
            dose = float(ev.get("dose", 0) or 0)
            used_at_raw = ev.get("used_at", ev.get("usedAt"))
            insulin_type = _normalize_insulin_type(ev.get("insulin_type", ev.get("insulinType", "Unknown")))
            if used_at_raw is None:
                continue
            used_at = _parse_iso(str(used_at_raw))
            delta_min = (dt0 - used_at).total_seconds() / 60.0
            if delta_min < 0:
                continue

            if latest_any_ts is None or used_at > latest_any_ts:
                latest_any_ts = used_at
                latest_any_type = insulin_type

            if et == "bolus":
                if delta_min <= 30:
                    bolus_30 += dose
                if delta_min <= 60:
                    bolus_60 += dose
                if delta_min <= 120:
                    bolus_120 += dose
                if latest_bolus_ts is None or used_at > latest_bolus_ts:
                    latest_bolus_ts = used_at
                    latest_bolus_type = insulin_type

            elif et == "basal":
                if latest_basal_ts is None or used_at > latest_basal_ts:
                    latest_basal_ts = used_at
                    latest_basal_daily = dose
        except Exception:
            continue

    # Basal event is usually daily total in this API shape -> convert to hourly-like scale.
    basal_hourly = 0.0
    if latest_basal_daily is not None:
        basal_hourly = max(0.0, float(latest_basal_daily) / 24.0)

    insulin_type = latest_bolus_type or latest_any_type or _normalize_insulin_type(req.get("insulin_type", "Unknown"))

    return {
        "bolus_30": float(bolus_30),
        "bolus_60": float(bolus_60),
        "bolus_120": float(bolus_120),
        "basal_hourly": float(basal_hourly),
        "insulin_type": insulin_type,
        "is_insulin_user": 1.0,
    }


def _resolve_insulin_inputs(req: Dict[str, Any], dt0: datetime) -> Dict[str, float | str]:
    derived = _extract_insulin_event_features(req=req, dt0=dt0)

    explicit_bolus = float(req.get("insulin_bolus", 0) or 0)
    explicit_basal = float(req.get("insulin_basal", 0) or 0)
    explicit_type = _normalize_insulin_type(req.get("insulin_type", "Unknown"))
    explicit_user = float(req.get("is_insulin_user", 1.0) or 0)

    insulin_bolus = explicit_bolus if explicit_bolus > 0 else float(derived["bolus_30"])
    insulin_basal = explicit_basal if explicit_basal > 0 else float(derived["basal_hourly"])
    insulin_type = explicit_type if _has_named_insulin_type(explicit_type) else str(derived["insulin_type"])
    is_insulin_user = 1.0 if (explicit_user > 0 or _has_named_insulin_type(insulin_type)) else float(derived["is_insulin_user"])

    bolus_dose_60m = max(float(req.get("bolus_dose_60m", 0) or 0), float(derived["bolus_60"]))
    bolus_dose_120m = max(float(req.get("bolus_dose_120m", 0) or 0), float(derived["bolus_120"]))
    bolus_carb_input_30m = float(req.get("bolus_carb_input_30m", 0) or 0)
    bolus_carb_input_60m = float(req.get("bolus_carb_input_60m", 0) or 0)
    bolus_carb_input_120m = float(req.get("bolus_carb_input_120m", 0) or 0)
    temp_basal_active = float(1.0 if bool(req.get("temp_basal_active", False)) else 0.0)
    temp_basal_value = float(req.get("temp_basal_value", 0) or 0)
    insulin_total_60m = float(req.get("insulin_total_60m", 0) or 0)
    if insulin_total_60m <= 0:
        insulin_total_60m = bolus_dose_60m + insulin_basal
    insulin_total_120m = float(req.get("insulin_total_120m", 0) or 0)
    if insulin_total_120m <= 0:
        insulin_total_120m = bolus_dose_120m + insulin_basal
    insulin_onboard_proxy = float(req.get("insulin_onboard_proxy", 0) or 0)
    if insulin_onboard_proxy <= 0:
        insulin_onboard_proxy = 0.5 * insulin_bolus + 0.3 * bolus_dose_60m + 0.2 * bolus_dose_120m
    basal_bolus_ratio = float(req.get("basal_bolus_ratio", 0) or 0)
    if basal_bolus_ratio <= 0:
        basal_bolus_ratio = insulin_basal / (bolus_dose_60m + 1e-3)

    has_any_source = any([
        insulin_bolus > 0,
        insulin_basal > 0,
        bolus_dose_60m > 0,
        bolus_dose_120m > 0,
        temp_basal_active > 0,
        temp_basal_value > 0,
        insulin_total_60m > 0,
        insulin_total_120m > 0,
        insulin_onboard_proxy > 0,
    ])
    if not has_any_source and is_insulin_user <= 0:
        raise ValueError("Insulin input is required.")

    return {
        "insulin_bolus": float(insulin_bolus),
        "insulin_basal": float(insulin_basal),
        "insulin_type": insulin_type,
        "is_insulin_user": float(is_insulin_user),
        "bolus_dose_60m": float(bolus_dose_60m),
        "bolus_dose_120m": float(bolus_dose_120m),
        "bolus_carb_input_30m": float(bolus_carb_input_30m),
        "bolus_carb_input_60m": float(bolus_carb_input_60m),
        "bolus_carb_input_120m": float(bolus_carb_input_120m),
        "temp_basal_active": float(temp_basal_active),
        "temp_basal_value": float(temp_basal_value),
        "insulin_total_60m": float(insulin_total_60m),
        "insulin_total_120m": float(insulin_total_120m),
        "insulin_onboard_proxy": float(insulin_onboard_proxy),
        "basal_bolus_ratio": float(basal_bolus_ratio),
    }


def _build_feature_row(req: Dict[str, Any], meta: Dict[str, Any]) -> pd.DataFrame:
    numeric_feats: List[str] = meta["features"]["numeric"]
    cat_feats: List[str] = meta["features"]["categorical"]

    points = _extract_glucose_points(req)
    points = points[-3:]  # 최근 3개만 사용

    dt0 = _parse_iso(req["timestamp"])

    # 최신 것이 gh1
    latest = list(reversed(points))

    def _get_point(idx: int):
        if idx < len(latest):
            return latest[idx]
        return None, np.nan, "기타"

    gh1_dt, gh1_g, gh1_label = _get_point(0)
    gh2_dt, gh2_g, gh2_label = _get_point(1)
    gh3_dt, gh3_g, gh3_label = _get_point(2)

    def _age_min(prev_dt: datetime | None) -> float:
        if prev_dt is None:
            return 0.0
        return max((dt0 - prev_dt).total_seconds() / 60.0, 0.0)

    gh1_age_min = _age_min(gh1_dt)
    gh2_age_min = _age_min(gh2_dt) if not np.isnan(gh2_g) else 0.0
    gh3_age_min = _age_min(gh3_dt) if not np.isnan(gh3_g) else 0.0

    gh2_exists = 0.0 if np.isnan(gh2_g) else 1.0
    gh3_exists = 0.0 if np.isnan(gh3_g) else 1.0

    bg_1 = float(gh1_g)
    bg_2 = 0.0 if np.isnan(gh2_g) else float(gh2_g)
    bg_3 = 0.0 if np.isnan(gh3_g) else float(gh3_g)

    bg_diff_1 = bg_1 - bg_2 if gh2_exists == 1 else 0.0
    bg_diff_2 = bg_2 - bg_3 if gh2_exists == 1 and gh3_exists == 1 else 0.0

    gap_12 = gh2_age_min - gh1_age_min if gh2_exists == 1 else 0.0
    gap_23 = gh3_age_min - gh2_age_min if gh2_exists == 1 and gh3_exists == 1 else 0.0

    bg_slope_1 = bg_diff_1 / gap_12 if gap_12 > 0 else 0.0
    bg_slope_2 = bg_diff_2 / gap_23 if gap_23 > 0 else 0.0

    carbs_raw = float(req.get("carb_intake", 0) or 0)
    carbs_raw = max(0.0, carbs_raw)
    carbs_log = float(np.log1p(carbs_raw))
    meal_event_flag = 1.0 if carbs_raw > 0 else 0.0

    steps = float(req.get("steps", 0) or 0)
    intensity = float(req.get("intensity", 0) or 0)
    steps_log = float(np.log1p(max(0.0, steps)))
    has_activity = 1.0 if steps > 0 else 0.0

    insulin_inputs = _resolve_insulin_inputs(req=req, dt0=dt0)
    insulin_bolus = float(insulin_inputs["insulin_bolus"])
    insulin_basal = float(insulin_inputs["insulin_basal"])
    insulin_type = str(insulin_inputs["insulin_type"])
    is_insulin_user = float(insulin_inputs["is_insulin_user"])
    bolus_dose_60m = float(insulin_inputs["bolus_dose_60m"])
    bolus_dose_120m = float(insulin_inputs["bolus_dose_120m"])
    bolus_carb_input_30m = float(insulin_inputs["bolus_carb_input_30m"])
    bolus_carb_input_60m = float(insulin_inputs["bolus_carb_input_60m"])
    bolus_carb_input_120m = float(insulin_inputs["bolus_carb_input_120m"])
    temp_basal_active = float(insulin_inputs["temp_basal_active"])
    temp_basal_value = float(insulin_inputs["temp_basal_value"])
    insulin_total_60m = float(insulin_inputs["insulin_total_60m"])
    insulin_total_120m = float(insulin_inputs["insulin_total_120m"])
    insulin_onboard_proxy = float(insulin_inputs["insulin_onboard_proxy"])
    basal_bolus_ratio = float(insulin_inputs["basal_bolus_ratio"])
    bolus_per_carb = insulin_bolus / (carbs_raw + 1.0)
    carb_step_ratio = carbs_raw / (steps + 1.0)

    hour = float(dt0.hour)
    weekday = float(dt0.weekday())

    defaults_num = {
        "current_glucose": bg_1,
        "bg_1": bg_1,
        "bg_1_age_min": gh1_age_min,
        "bg_2": bg_2,
        "bg_2_age_min": gh2_age_min,
        "gh2_exists": gh2_exists,
        "bg_3": bg_3,
        "bg_3_age_min": gh3_age_min,
        "gh3_exists": gh3_exists,
        "bg_diff_1": bg_diff_1,
        "bg_diff_2": bg_diff_2,
        "bg_slope_1": bg_slope_1,
        "bg_slope_2": bg_slope_2,
        "carbs_raw": carbs_raw,
        "carbs_log": carbs_log,
        "meal_event_flag": meal_event_flag,
        "steps": steps,
        "steps_log": steps_log,
        "has_activity": has_activity,
        "intensity": intensity,
        "insulin_bolus": insulin_bolus,
        "insulin_basal": insulin_basal,
        "bolus_dose_60m": bolus_dose_60m,
        "bolus_dose_120m": bolus_dose_120m,
        "bolus_carb_input_30m": bolus_carb_input_30m,
        "bolus_carb_input_60m": bolus_carb_input_60m,
        "bolus_carb_input_120m": bolus_carb_input_120m,
        "temp_basal_active": temp_basal_active,
        "temp_basal_value": temp_basal_value,
        "insulin_total_60m": insulin_total_60m,
        "insulin_total_120m": insulin_total_120m,
        "insulin_onboard_proxy": insulin_onboard_proxy,
        "basal_bolus_ratio": basal_bolus_ratio,
        "is_insulin_user": is_insulin_user,
        "bolus_per_carb": bolus_per_carb,
        "carb_step_ratio": carb_step_ratio,
        "hour": hour,
        "weekday": weekday,
    }

    defaults_cat = {
        "meal_type": _meal_type_to_label(req.get("meal_type", 0)),
        "insulin_type": insulin_type,
        "gh1_label": gh1_label,
        "gh2_label": gh2_label,
        "gh3_label": gh3_label,
    }

    row = {}
    for f in numeric_feats:
        row[f] = defaults_num.get(f, 0.0)
    for f in cat_feats:
        row[f] = defaults_cat.get(f, "unknown")

    return pd.DataFrame([row], columns=numeric_feats + cat_feats)


def clip_glucose(values: List[float], low: float = 40.0, high: float = 400.0) -> List[float]:
    return [max(low, min(high, float(v))) for v in values]


def limit_step_change(values: List[float], max_up: float = 25.0, max_down: float = 25.0) -> List[float]:
    if not values:
        return values
    out = [float(values[0])]
    for v in values[1:]:
        prev = out[-1]
        diff = float(v) - prev
        if diff > max_up:
            v = prev + max_up
        elif diff < -max_down:
            v = prev - max_down
        out.append(float(v))
    return out


def smooth_forecast(values: List[float]) -> List[float]:
    if len(values) < 3:
        return values
    out = values[:]
    for i in range(1, len(values) - 1):
        out[i] = 0.25 * values[i - 1] + 0.50 * values[i] + 0.25 * values[i + 1]
    return [float(v) for v in out]


def postprocess_forecast(values: List[float]) -> List[float]:
    values = clip_glucose(values, 40.0, 400.0)
    values = limit_step_change(values, 25.0, 25.0)
    values = smooth_forecast(values)
    values = clip_glucose(values, 40.0, 400.0)
    return values


def apply_meal_guardrails(
    forecast: List[Dict[str, Any]],
    req: Dict[str, Any],
    current_glucose: float,
) -> List[Dict[str, Any]]:
    if not forecast:
        return forecast

    carbs = float(req.get("carb_intake", 0) or 0)
    dt0 = _parse_iso(req["timestamp"])
    insulin_inputs = _resolve_insulin_inputs(req=req, dt0=dt0)
    bolus = max(float(insulin_inputs["insulin_bolus"]), float(insulin_inputs["bolus_dose_60m"]))
    glucose_points = _extract_glucose_points(req)
    recent_rise = 0.0
    if len(glucose_points) >= 2:
        recent_rise = max(0.0, current_glucose - min(float(p[1]) for p in glucose_points[:-1]))
    out = [dict(p) for p in forecast]

    # 0) Slow down the early ramp so 10~40m does not spike too aggressively.
    early_total_rise_cap = float(np.clip(
        0.35 * carbs + 2.0 * min(bolus, 3.0) + 0.40 * recent_rise,
        12.0,
        38.0,
    ))
    for i in range(len(out)):
        off = int(out[i]["offset_minutes"])
        if off > 40:
            continue
        progress = off / 40.0
        allowed = current_glucose + early_total_rise_cap * progress
        cur = float(out[i]["predicted_glucose"])
        if cur > allowed:
            out[i]["predicted_glucose"] = round(float(allowed), 3)

    early_step_up_cap = float(np.clip(
        4.0 + 0.12 * carbs + 0.8 * min(bolus, 3.0),
        6.0,
        12.0,
    ))
    for i in range(1, len(out)):
        off = int(out[i]["offset_minutes"])
        if off > 60:
            continue
        prev = float(out[i - 1]["predicted_glucose"])
        cur = float(out[i]["predicted_glucose"])
        if cur > prev + early_step_up_cap:
            out[i]["predicted_glucose"] = round(prev + early_step_up_cap, 3)

    # 1) Compress late tail rise so 100~120m does not keep drifting upward.
    anchor_idx = max((i for i, p in enumerate(out) if int(p["offset_minutes"]) <= 90), default=None)
    if anchor_idx is not None and anchor_idx < len(out) - 1:
        anchor_off = int(out[anchor_idx]["offset_minutes"])
        anchor_val = float(out[anchor_idx]["predicted_glucose"])
        tail_total_rise_cap = float(np.clip(
            0.10 * carbs + 1.5 * min(bolus, 4.0) + 0.30 * recent_rise,
            4.0,
            18.0,
        ))
        if carbs < 10.0:
            tail_total_rise_cap = min(tail_total_rise_cap, 4.0)
        elif carbs < 25.0:
            tail_total_rise_cap = min(tail_total_rise_cap, 8.0)
        elif carbs < 50.0:
            tail_total_rise_cap = min(tail_total_rise_cap, 12.0)

        for i in range(anchor_idx + 1, len(out)):
            off = int(out[i]["offset_minutes"])
            if off <= anchor_off:
                continue
            progress = (off - anchor_off) / max(120 - anchor_off, 10)
            allowed = anchor_val + tail_total_rise_cap * progress
            cur = float(out[i]["predicted_glucose"])
            if cur > allowed:
                out[i]["predicted_glucose"] = round(float(allowed), 3)

    # 2) Once a meaningful early peak has already formed, do not allow late tail rebound.
    decline_started = False
    early_peak_formed = False
    early_peak_threshold = float(current_glucose + np.clip(0.20 * carbs + 2.0, 4.0, 12.0))
    for i in range(1, len(out)):
        prev = float(out[i - 1]["predicted_glucose"])
        cur = float(out[i]["predicted_glucose"])
        off = int(out[i]["offset_minutes"])
        if off <= 70 and prev >= early_peak_threshold:
            early_peak_formed = True
        if cur < prev - 0.5:
            decline_started = True
        if early_peak_formed and decline_started and off >= 80 and cur > prev:
            drop_step = 0.5 if off < 90 else 1.0
            out[i]["predicted_glucose"] = round(prev - drop_step, 3)

    # 3) For meal without bolus, avoid unrealistic deep drop below current.
    #    Keep 120m near current range for low/moderate carbs.
    if bolus <= 0.1:
        # Keep no-bolus meal case from dropping too deep below current.
        max_drop = float(np.clip(0.10 * carbs, 2.0, 6.0))
        floor_val = float(current_glucose - max_drop)
        for i in range(len(out)):
            g = float(out[i]["predicted_glucose"])
            if g < floor_val:
                out[i]["predicted_glucose"] = round(floor_val, 3)

        if carbs >= 15.0:
            # All requests are insulin-user cases.
            peak_floor = float(current_glucose + np.clip(0.60 * carbs, 10.0, 38.0))
            tail_floor = float(current_glucose + np.clip(0.40 * carbs, 7.0, 26.0))

            for i in range(len(out)):
                off = int(out[i]["offset_minutes"])
                g = float(out[i]["predicted_glucose"])
                if 40 <= off <= 80 and g < peak_floor:
                    out[i]["predicted_glucose"] = round(peak_floor, 3)
                elif off >= 90 and g < tail_floor:
                    out[i]["predicted_glucose"] = round(tail_floor, 3)

    return out


def build_peak_from_forecast(forecast: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not forecast:
        return None
    peak_idx = max(range(len(forecast)), key=lambda i: forecast[i]["predicted_glucose"])
    peak_pt = forecast[peak_idx]
    return {
        "peak_glucose": round(float(peak_pt["predicted_glucose"]), 3),
        "peak_time": peak_pt["time"],
        "peak_offset_minutes": int(peak_pt["offset_minutes"]),
    }


def predict_glucose(req: Dict[str, Any]) -> Dict[str, Any]:
    models, meta = load_model_meta()

    graph_models = _select_model_family(models=models, req=req)

    X = _build_feature_row(req, meta)

    glucose_points = _extract_glucose_points(req)
    current_glucose = float(glucose_points[-1][1])

    dt0 = _parse_iso(req["timestamp"])
    step_minutes = int(req.get("step_minutes", 10))
    horizon_minutes = int(req.get("horizon_minutes", 120))
    prediction_offset_minutes = int(req.get("prediction_offset_minutes", 120))

    meta_horizons = meta.get("graph_horizons", meta.get("horizons", []))
    available_horizons = sorted(int(h) for h in meta_horizons)
    chosen_horizons = [h for h in available_horizons if h <= horizon_minutes and h % step_minutes == 0]
    if not chosen_horizons:
        raise ValueError("No compatible graph horizons available.")

    raw_forecast = []
    for idx, h in enumerate(chosen_horizons, start=1):
        model_entry = graph_models[str(h)]
        pred_delta = _predict_delta_with_model_entry(model_entry, X, req)
        pred_glucose = current_glucose + pred_delta

        t = dt0 + timedelta(minutes=h)
        raw_forecast.append({
            "time": t.isoformat().replace("+00:00", "Z"),
            "predicted_glucose": float(pred_glucose),
            "step": idx,
            "offset_minutes": h,
        })

    raw_values = [pt["predicted_glucose"] for pt in raw_forecast]
    processed_values = postprocess_forecast(raw_values)

    forecast = []
    for i, pt in enumerate(raw_forecast):
        forecast.append({
            "time": pt["time"],
            "predicted_glucose": round(float(processed_values[i]), 3),
            "step": pt["step"],
            "offset_minutes": pt["offset_minutes"],
        })

    forecast = apply_meal_guardrails(
        forecast=forecast,
        req=req,
        current_glucose=current_glucose,
    )

    final_match = next((p for p in forecast if p["offset_minutes"] == prediction_offset_minutes), None)
    if final_match is None:
        fallback_h = max(h for h in chosen_horizons if h <= prediction_offset_minutes)
        final_match = next(p for p in forecast if p["offset_minutes"] == fallback_h)

    milestones = {}
    for h in (10, 30, 60, 90, 120):
        m = next((p for p in forecast if p["offset_minutes"] == h), None)
        if m is not None:
            milestones[str(h)] = m

    peak = build_peak_from_forecast(forecast)

    result = {
        "predicted_glucose": round(float(final_match["predicted_glucose"]), 3),
        "prediction_offset_minutes": int(final_match["offset_minutes"]),
        "predicted_time": final_match["time"],
        "forecast": forecast,
        "milestones": milestones if milestones else None,
        "peak": peak,
        "debug": None,
    }

    if bool(req.get("debug", False)):
        insulin_inputs = _resolve_insulin_inputs(req=req, dt0=dt0)
        result["debug"] = {
            "current_glucose": current_glucose,
            "graph_horizons": available_horizons,
            "used_horizons": chosen_horizons,
            "input_summary": {
                "carb_intake": float(req.get("carb_intake", 0) or 0),
                "glucose_history_len": len(glucose_points),
                "steps": float(req.get("steps", 0) or 0),
                "intensity": float(req.get("intensity", 0) or 0),
                "meal_type": int(req.get("meal_type", 0) or 0),
                "insulin_bolus": float(insulin_inputs["insulin_bolus"]),
                "bolus_dose_60m": float(insulin_inputs["bolus_dose_60m"]),
                "bolus_dose_120m": float(insulin_inputs["bolus_dose_120m"]),
                "insulin_basal": float(insulin_inputs["insulin_basal"]),
                "insulin_type": _normalize_insulin_type(insulin_inputs["insulin_type"]),
                "is_insulin_user": float(insulin_inputs["is_insulin_user"]),
            },
        }

    return result
