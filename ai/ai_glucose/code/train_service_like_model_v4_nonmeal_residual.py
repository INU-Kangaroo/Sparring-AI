import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


@dataclass
class TrainConfig:
    train_path: str = "last_data/train_dataset_service_like_labeled.csv"
    test_path: str = "last_data/test_dataset_service_like_labeled.csv"
    random_state: int = 42

    model_dir: str = "models"
    model_name: str = "glucose_service_like_models_v4_nonmeal_residual.joblib"
    meta_name: str = "glucose_service_like_models_v4_nonmeal_residual_meta.json"

    graph_horizons: Tuple[int, ...] = tuple(range(10, 121, 10))
    final_horizon: int = 120
    non_meal_max_carbs: float = 0.0
    non_meal_min_glucose: float = 180.0
    non_meal_min_bolus_60m: float = 0.5
    non_meal_min_rows: int = 500
    non_meal_residual_min_horizon: int = 90
    non_meal_residual_max_horizon: int = 120


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def _prepare(df_raw: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    df = df_raw.copy()

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    elif "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "current_glucose" not in df.columns and "gh1_glucose" in df.columns:
        df["current_glucose"] = pd.to_numeric(df["gh1_glucose"], errors="coerce")
    if "bg_1" not in df.columns and "gh1_glucose" in df.columns:
        df["bg_1"] = pd.to_numeric(df["gh1_glucose"], errors="coerce")
    if "bg_1_age_min" not in df.columns:
        df["bg_1_age_min"] = pd.to_numeric(df.get("gh1_age_min", 0.0), errors="coerce").fillna(0.0)
    if "bg_2" not in df.columns and "gh2_glucose" in df.columns:
        df["bg_2"] = pd.to_numeric(df["gh2_glucose"], errors="coerce")
    if "bg_2_age_min" not in df.columns:
        df["bg_2_age_min"] = pd.to_numeric(df.get("gh2_age_min", np.nan), errors="coerce")
    if "bg_3" not in df.columns and "gh3_glucose" in df.columns:
        df["bg_3"] = pd.to_numeric(df["gh3_glucose"], errors="coerce")
    if "bg_3_age_min" not in df.columns:
        df["bg_3_age_min"] = pd.to_numeric(df.get("gh3_age_min", np.nan), errors="coerce")

    if "carbs_raw" not in df.columns and "carb_intake" in df.columns:
        df["carbs_raw"] = pd.to_numeric(df["carb_intake"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "carbs_log" not in df.columns:
        df["carbs_log"] = np.log1p(pd.to_numeric(df.get("carbs_raw", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0))
    if "meal_event_flag" not in df.columns:
        df["meal_event_flag"] = (pd.to_numeric(df.get("carbs_raw", 0.0), errors="coerce").fillna(0.0) > 0).astype(int)

    if "gh2_exists" not in df.columns:
        df["gh2_exists"] = pd.to_numeric(df.get("bg_2", np.nan), errors="coerce").notna().astype(int)
    if "gh3_exists" not in df.columns:
        df["gh3_exists"] = pd.to_numeric(df.get("bg_3", np.nan), errors="coerce").notna().astype(int)
    if "gh1_label" not in df.columns:
        df["gh1_label"] = "기타"
    if "gh2_label" not in df.columns:
        df["gh2_label"] = "기타"
    if "gh3_label" not in df.columns:
        df["gh3_label"] = "기타"
    if "insulin_type" not in df.columns:
        df["insulin_type"] = "unknown"
    df["insulin_type"] = df["insulin_type"].fillna("unknown").astype(str)

    numeric_cols = [
        "current_glucose", "bg_1", "bg_1_age_min", "bg_2", "bg_2_age_min", "bg_3", "bg_3_age_min",
        "carbs_raw", "carbs_log", "meal_event_flag", "steps", "intensity", "insulin_bolus", "insulin_basal",
        "bolus_dose_60m", "bolus_dose_120m", "bolus_carb_input_30m", "bolus_carb_input_60m", "bolus_carb_input_120m",
        "temp_basal_active", "temp_basal_value", "insulin_total_60m", "insulin_total_120m", "insulin_onboard_proxy",
        "basal_bolus_ratio", "is_insulin_user", "hour", "weekday",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["patient_id"] = df["patient_id"].astype(str)
    df["bg_diff_1"] = df["bg_1"] - df["bg_2"]
    df["bg_diff_2"] = df["bg_2"] - df["bg_3"]
    gap_12 = (df["bg_2_age_min"] - df["bg_1_age_min"]).replace(0, np.nan)
    gap_23 = (df["bg_3_age_min"] - df["bg_2_age_min"]).replace(0, np.nan)
    df["bg_slope_1"] = df["bg_diff_1"] / gap_12
    df["bg_slope_2"] = df["bg_diff_2"] / gap_23
    for c in ["bg_diff_1", "bg_diff_2", "bg_slope_1", "bg_slope_2"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["steps_log"] = np.log1p(df["steps"].clip(lower=0))
    df["has_activity"] = (df["steps"].fillna(0) > 0).astype(int)
    df["bolus_per_carb"] = df["insulin_bolus"] / (df["carbs_raw"] + 1.0)
    df["carb_step_ratio"] = df["carbs_raw"] / (df["steps"] + 1.0)

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


def _build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    cfg: TrainConfig,
    *,
    objective: str = "reg:squarederror",
    n_estimators: int = 1600,
    learning_rate: float = 0.03,
    max_depth: int = 5,
    min_child_weight: int = 4,
) -> Pipeline:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_features), ("cat", cat_pipe, categorical_features)],
        remainder="drop",
        sparse_threshold=0.3,
    )
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        reg_alpha=0.1,
        objective=objective,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def _make_sample_weight(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df), dtype=float)
    w += 1.0 * df["meal_event_flag"].fillna(0).to_numpy(dtype=float)
    carbs = df["carbs_raw"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    w += 0.4 * np.log1p(carbs)
    w += 0.4 * (carbs >= 20).astype(float)
    bolus = df["insulin_bolus"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    w += 0.15 * np.log1p(bolus)
    meal_type = df["meal_type"].fillna("unknown").astype(str).str.lower()
    lunch_flag = (meal_type == "lunch").to_numpy(dtype=float)
    low_carb_flag = ((carbs >= 15.0) & (carbs <= 35.0)).astype(float)
    low_bolus_flag = (bolus <= 0.5).astype(float)
    w += 1.2 * lunch_flag * low_carb_flag * low_bolus_flag
    return w


def _make_sample_weight_with_target(df: pd.DataFrame, y_delta: np.ndarray) -> np.ndarray:
    w = _make_sample_weight(df)
    w += 0.15 * np.log1p(np.abs(y_delta))
    return w


def _calc_metrics(y_true_glucose, y_pred_glucose, y_true_delta, y_pred_delta) -> Dict[str, float]:
    return {
        "mae_glucose": float(mean_absolute_error(y_true_glucose, y_pred_glucose)),
        "rmse_glucose": float(mean_squared_error(y_true_glucose, y_pred_glucose) ** 0.5),
        "mae_delta": float(mean_absolute_error(y_true_delta, y_pred_delta)),
        "rmse_delta": float(mean_squared_error(y_true_delta, y_pred_delta) ** 0.5),
    }


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
    train_df = _prepare(_safe_read_csv(cfg.train_path), cfg)
    test_df = _prepare(_safe_read_csv(cfg.test_path), cfg)

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
            "non_meal_residual_min_horizon": int(cfg.non_meal_residual_min_horizon),
            "non_meal_residual_max_horizon": int(cfg.non_meal_residual_max_horizon),
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
        "features": {"numeric": numeric_features, "categorical": categorical_features},
        "graph_metrics": graph_metrics,
        "final_120_metrics": final_metrics,
        "notes": [
            "Base is v1-style global model.",
            "Additional non-meal residual is trained for carb_intake==0 and high-glucose/recent-bolus cases.",
            "Inference adds residual only for non-meal correction-like inputs.",
            f"Non-meal residual is applied only at {cfg.non_meal_residual_min_horizon}~{cfg.non_meal_residual_max_horizon} minutes.",
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved model bundle: {model_path}")
    print(f"[DONE] Saved meta        : {meta_path}")


if __name__ == "__main__":
    main()
