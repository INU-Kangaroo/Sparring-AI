import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from joblib import dump

from train_service_like_model_v4_nonmeal_residual import (
    _build_pipeline,
    _calc_metrics,
    _make_sample_weight,
    _safe_read_csv,
)


@dataclass
class TrainConfig:
    train_path: str = "last_data/train_dataset_service_like_labeled.csv"
    test_path: str = "last_data/test_dataset_service_like_labeled.csv"
    random_state: int = 42

    model_dir: str = "models"
    model_name: str = "glucose_service_like_models_v4_realshape.joblib"
    meta_name: str = "glucose_service_like_models_v4_realshape_meta.json"

    graph_horizons: Tuple[int, ...] = tuple(range(10, 121, 10))
    final_horizon: int = 120
    non_meal_max_carbs: float = 0.0
    non_meal_min_glucose: float = 180.0
    non_meal_min_bolus_60m: float = 0.5
    non_meal_min_rows: int = 500


def _prepare_realshape(df_raw: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    df = df_raw.copy()

    if "timestamp" not in df.columns:
        raise ValueError("Missing timestamp column.")
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["ts"].isna().all():
        raise ValueError("Failed to parse timestamp column.")

    df["current_glucose"] = pd.to_numeric(df.get("gh1_glucose"), errors="coerce")
    df["bg_1"] = df["current_glucose"]
    df["bg_1_age_min"] = 0.0

    df["gh2_exists"] = pd.to_numeric(df.get("gh2_exists", 0), errors="coerce").fillna(0).astype(int)
    df["gh3_exists"] = pd.to_numeric(df.get("gh3_exists", 0), errors="coerce").fillna(0).astype(int)

    df["bg_2"] = np.where(
        df["gh2_exists"] == 1,
        pd.to_numeric(df.get("gh2_glucose"), errors="coerce"),
        0.0,
    )
    df["bg_3"] = np.where(
        df["gh3_exists"] == 1,
        pd.to_numeric(df.get("gh3_glucose"), errors="coerce"),
        0.0,
    )
    df["bg_2_age_min"] = np.where(
        df["gh2_exists"] == 1,
        pd.to_numeric(df.get("gh2_age_min", 0.0), errors="coerce").fillna(0.0),
        0.0,
    )
    df["bg_3_age_min"] = np.where(
        df["gh3_exists"] == 1,
        pd.to_numeric(df.get("gh3_age_min", 0.0), errors="coerce").fillna(0.0),
        0.0,
    )

    df["gh1_label"] = df.get("gh1_label", "기타").fillna("기타").astype(str)
    df["gh2_label"] = np.where(
        df["gh2_exists"] == 1,
        df.get("gh2_label", "기타").fillna("기타").astype(str),
        "기타",
    )
    df["gh3_label"] = np.where(
        df["gh3_exists"] == 1,
        df.get("gh3_label", "기타").fillna("기타").astype(str),
        "기타",
    )

    df["carbs_raw"] = pd.to_numeric(df.get("carb_intake", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    df["carbs_log"] = np.log1p(df["carbs_raw"])
    df["meal_event_flag"] = (df["carbs_raw"] > 0).astype(int)

    df["steps"] = pd.to_numeric(df.get("steps", 0.0), errors="coerce").fillna(0.0)
    df["intensity"] = pd.to_numeric(df.get("intensity", 0.0), errors="coerce").fillna(0.0)
    df["steps_log"] = np.log1p(df["steps"].clip(lower=0.0))
    df["has_activity"] = (df["steps"] > 0).astype(int)

    bolus_event_dose = pd.to_numeric(df.get("insulin_bolus", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    basal_event_dose = pd.to_numeric(df.get("insulin_basal", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    # Match predict.py real-shape inference:
    # - bolus event is treated as a recent event within 30/60/120m
    # - basal event dose is treated as daily total and converted to hourly-like scale via /24
    df["insulin_bolus"] = bolus_event_dose
    df["insulin_basal"] = basal_event_dose / 24.0
    df["bolus_dose_60m"] = bolus_event_dose
    df["bolus_dose_120m"] = bolus_event_dose
    df["bolus_carb_input_30m"] = 0.0
    df["bolus_carb_input_60m"] = 0.0
    df["bolus_carb_input_120m"] = 0.0
    df["temp_basal_active"] = pd.to_numeric(df.get("temp_basal_active", 0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    df["temp_basal_value"] = pd.to_numeric(df.get("temp_basal_value", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    df["insulin_total_60m"] = df["bolus_dose_60m"] + df["insulin_basal"]
    df["insulin_total_120m"] = df["bolus_dose_120m"] + df["insulin_basal"]
    df["insulin_onboard_proxy"] = 0.5 * df["insulin_bolus"] + 0.3 * df["bolus_dose_60m"] + 0.2 * df["bolus_dose_120m"]
    df["basal_bolus_ratio"] = df["insulin_basal"] / (df["bolus_dose_60m"] + 1e-3)
    df["is_insulin_user"] = 1.0
    df["insulin_type"] = df.get("insulin_type", "Unknown").fillna("Unknown").astype(str)

    df["bolus_per_carb"] = df["insulin_bolus"] / (df["carbs_raw"] + 1.0)
    df["carb_step_ratio"] = df["carbs_raw"] / (df["steps"] + 1.0)

    df["bg_diff_1"] = np.where(df["gh2_exists"] == 1, df["bg_1"] - df["bg_2"], 0.0)
    df["bg_diff_2"] = np.where((df["gh2_exists"] == 1) & (df["gh3_exists"] == 1), df["bg_2"] - df["bg_3"], 0.0)
    gap_12 = (df["bg_2_age_min"] - df["bg_1_age_min"]).replace(0, np.nan)
    gap_23 = (df["bg_3_age_min"] - df["bg_2_age_min"]).replace(0, np.nan)
    df["bg_slope_1"] = np.where(df["gh2_exists"] == 1, df["bg_diff_1"] / gap_12, 0.0)
    df["bg_slope_2"] = np.where((df["gh2_exists"] == 1) & (df["gh3_exists"] == 1), df["bg_diff_2"] / gap_23, 0.0)
    for c in ["bg_diff_1", "bg_diff_2", "bg_slope_1", "bg_slope_2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["hour"] = df["ts"].dt.hour.astype(float)
    df["weekday"] = df["ts"].dt.weekday.astype(float)
    df["patient_id"] = df["patient_id"].astype(str)
    df["meal_type"] = df.get("meal_type", "unknown").fillna("unknown").astype(str)

    df = df[df["current_glucose"].between(40, 400, inclusive="both")]
    df = df[df["carbs_raw"].between(0, 300, inclusive="both")]
    df = df[df["steps"].between(0, 100000, inclusive="both")]
    df = df[df["intensity"].between(0, 1, inclusive="both")]
    df = df[df["hour"].between(0, 23, inclusive="both")]
    df = df[df["weekday"].between(0, 6, inclusive="both")]

    all_targets = sorted(set(cfg.graph_horizons + (cfg.final_horizon,)))
    for h in all_targets:
        y_col = f"y_glucose_{h}"
        d_col = f"delta_{h}"
        if y_col not in df.columns:
            raise ValueError(f"Missing target column: {y_col}")
        if d_col not in df.columns:
            df[d_col] = df[y_col] - df["current_glucose"]

    return df.reset_index(drop=True)


def _build_feature_lists() -> Tuple[List[str], List[str]]:
    numeric_features = [
        "current_glucose",
        "bg_1", "bg_1_age_min",
        "bg_2", "bg_2_age_min",
        "bg_3", "bg_3_age_min",
        "gh2_exists", "gh3_exists",
        "bg_diff_1", "bg_diff_2",
        "bg_slope_1", "bg_slope_2",
        "carbs_raw", "carbs_log",
        "meal_event_flag",
        "steps", "steps_log", "has_activity",
        "intensity",
        "insulin_bolus",
        "insulin_basal",
        "bolus_dose_60m",
        "bolus_dose_120m",
        "bolus_carb_input_30m",
        "bolus_carb_input_60m",
        "bolus_carb_input_120m",
        "temp_basal_active",
        "temp_basal_value",
        "insulin_total_60m",
        "insulin_total_120m",
        "insulin_onboard_proxy",
        "basal_bolus_ratio",
        "is_insulin_user",
        "bolus_per_carb",
        "carb_step_ratio",
        "hour",
        "weekday",
    ]
    categorical_features = ["meal_type", "insulin_type", "gh1_label", "gh2_label", "gh3_label"]
    return numeric_features, categorical_features


def _non_meal_residual_mask(df: pd.DataFrame, cfg: TrainConfig) -> pd.Series:
    return (
        (df["carbs_raw"] <= cfg.non_meal_max_carbs)
        & (
            (df["current_glucose"] >= cfg.non_meal_min_glucose)
            | (df["bolus_dose_60m"] >= cfg.non_meal_min_bolus_60m)
        )
    )


def main():
    cfg = TrainConfig()
    train_df = _prepare_realshape(_safe_read_csv(cfg.train_path), cfg)
    test_df = _prepare_realshape(_safe_read_csv(cfg.test_path), cfg)

    numeric_features, categorical_features = _build_feature_lists()
    numeric_features = [c for c in numeric_features if c in train_df.columns and c in test_df.columns]
    categorical_features = [c for c in categorical_features if c in train_df.columns and c in test_df.columns]

    graph_models: Dict[str, Any] = {}
    graph_metrics: Dict[str, Dict[str, float]] = {}

    for h in cfg.graph_horizons:
        target_col = f"delta_{h}"
        y_glucose_col = f"y_glucose_{h}"
        tr = train_df.dropna(subset=[target_col]).copy()
        te = test_df.dropna(subset=[target_col, y_glucose_col]).copy()
        if tr.empty or te.empty:
            continue

        X_tr = tr[numeric_features + categorical_features]
        y_tr = tr[target_col].values.astype(float)
        X_te = te[numeric_features + categorical_features]
        y_te_delta = te[target_col].values.astype(float)
        y_te_glucose = te[y_glucose_col].values.astype(float)

        base = _build_pipeline(
            numeric_features, categorical_features, cfg,
            objective="reg:pseudohubererror",
            n_estimators=1400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
        )
        base.fit(X_tr, y_tr, model__sample_weight=_make_sample_weight(tr))
        pred_delta = base.predict(X_te)

        non_meal_residual = None
        tr_nm_mask = _non_meal_residual_mask(tr, cfg)
        te_nm_mask = _non_meal_residual_mask(te, cfg)
        if int(tr_nm_mask.sum()) >= cfg.non_meal_min_rows:
            tr_nm = tr.loc[tr_nm_mask].copy()
            X_tr_nm = tr_nm[numeric_features + categorical_features]
            y_res = tr_nm[target_col].values.astype(float) - base.predict(X_tr_nm)
            non_meal_residual = _build_pipeline(
                numeric_features, categorical_features, cfg,
                objective="reg:squarederror",
                n_estimators=800,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=3,
            )
            residual_weight = np.ones(len(tr_nm), dtype=float)
            residual_weight += 0.6 * (tr_nm["current_glucose"].values >= cfg.non_meal_min_glucose).astype(float)
            residual_weight += 0.6 * (tr_nm["bolus_dose_60m"].values >= cfg.non_meal_min_bolus_60m).astype(float)
            residual_weight += 0.12 * np.log1p(np.abs(y_res))
            non_meal_residual.fit(X_tr_nm, y_res, model__sample_weight=residual_weight)
            if te_nm_mask.any():
                X_te_nm = te.loc[te_nm_mask, numeric_features + categorical_features]
                pred_delta[te_nm_mask.to_numpy()] += non_meal_residual.predict(X_te_nm)

        pred_glucose = np.clip(te["current_glucose"].values.astype(float) + pred_delta, 40.0, 400.0)
        metrics = _calc_metrics(y_te_glucose, pred_glucose, y_te_delta, pred_delta)
        metrics["n_train"] = int(len(tr))
        metrics["n_test"] = int(len(te))
        metrics["non_meal_residual_enabled"] = non_meal_residual is not None
        graph_metrics[str(h)] = metrics
        graph_models[str(h)] = {
            "base": base,
            "non_meal_residual": non_meal_residual,
            "non_meal_max_carbs": float(cfg.non_meal_max_carbs),
            "non_meal_min_glucose": float(cfg.non_meal_min_glucose),
            "non_meal_min_bolus_60m": float(cfg.non_meal_min_bolus_60m),
        }
        print(f"[{h:>3}m] MAE {metrics['mae_glucose']:.3f} | non_meal_residual={'on' if non_meal_residual is not None else 'off'}")

    final_model = graph_models.get("120")
    if final_model is None:
        raise ValueError("120-minute model was not trained.")
    final_metrics = graph_metrics["120"]

    os.makedirs(cfg.model_dir, exist_ok=True)
    model_path = os.path.join(cfg.model_dir, cfg.model_name)
    meta_path = os.path.join(cfg.model_dir, cfg.meta_name)
    dump({"graph_models": graph_models, "final_model_120": final_model}, model_path)

    meta = {
        "train_path_used": cfg.train_path,
        "test_path_used": cfg.test_path,
        "graph_horizons": list(cfg.graph_horizons),
        "final_horizon": cfg.final_horizon,
        "step_minutes": 10,
        "payload_mode": "real_shape_only",
        "features": {"numeric": numeric_features, "categorical": categorical_features},
        "graph_metrics": graph_metrics,
        "final_120_metrics": final_metrics,
        "notes": [
            "Trained on features reconstructed from real request shape only.",
            "Insulin inputs are rebuilt from event-like assumptions to match predict.py.",
            "Basal event dose is converted to hourly-like scale via /24, matching current inference.",
            "Additional non-meal residual is trained for carb_intake==0 and high-glucose/recent-bolus cases.",
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved model bundle: {model_path}")
    print(f"[DONE] Saved meta        : {meta_path}")


if __name__ == "__main__":
    main()
