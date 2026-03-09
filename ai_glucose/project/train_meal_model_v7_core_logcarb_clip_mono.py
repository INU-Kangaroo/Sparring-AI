import os
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


@dataclass
class TrainConfig:
    data_path: str = "output/train_dataset.csv"
    group_col: str = "subject"
    test_size: float = 0.2
    random_state: int = 42

    # winsorize percentiles for target delta_60
    clip_low_q: float = 0.01
    clip_high_q: float = 0.99

    model_dir: str = "models"
    model_name: str = "meal_model_core_logcarb_clip_mono.joblib"
    meta_name: str = "meal_model_core_logcarb_clip_mono_meta.json"


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV is empty.")
    return df


def _normalize_meal_type(s: pd.Series) -> pd.Series:
    x = s.fillna("unknown").astype(str).str.strip().str.lower()
    x = x.replace({"": "unknown", "nan": "unknown", "none": "unknown", "null": "unknown"})
    x = x.replace({"snacks": "snack"})
    return x


def _prepare(df_raw: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    df = df_raw.copy()

    # required
    for c in ["pre_glucose", "carbs", "delta_60", cfg.group_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # meal_type
    if "meal_type" not in df.columns:
        df["meal_type"] = "unknown"
    df["meal_type"] = _normalize_meal_type(df["meal_type"])

    # time features
    if "hour" not in df.columns or "weekday" not in df.columns:
        if "meal_time" in df.columns:
            ts = pd.to_datetime(df["meal_time"], errors="coerce", utc=True)
            df["hour"] = ts.dt.hour.fillna(0).astype(int)
            df["weekday"] = ts.dt.weekday.fillna(0).astype(int)
        else:
            df["hour"] = 0
            df["weekday"] = 0

    # activity defaults
    if "steps" not in df.columns:
        df["steps"] = 0.0
    if "intensity" not in df.columns:
        df["intensity"] = 0.0

    # numeric conversions
    for c in ["pre_glucose", "carbs", "delta_60", "steps", "intensity", "hour", "weekday"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["pre_glucose", "carbs", "delta_60", cfg.group_col])
    df[cfg.group_col] = df[cfg.group_col].astype(str)

    # sanity filters
    df = df[(df["pre_glucose"] >= 40) & (df["pre_glucose"] <= 400)]
    df = df[(df["carbs"] >= 0) & (df["carbs"] <= 300)]
    df = df[(df["steps"] >= 0) & (df["steps"] <= 50000)]
    df = df[(df["intensity"] >= 0) & (df["intensity"] <= 1)]
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]
    df = df[(df["weekday"] >= 0) & (df["weekday"] <= 6)]

    # log transform
    df["carbs_log"] = np.log1p(df["carbs"].astype(float))

    # evaluation helper
    df["g60"] = df["pre_glucose"] + df["delta_60"]

    return df


def _split(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    groups = df[cfg.group_col].values
    idx = np.arange(len(df))
    train_idx, test_idx = next(splitter.split(idx, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _build_pipeline(numeric_features: List[str], categorical_features: List[str], cfg: TrainConfig) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # numeric_features 순서에 맞춰 단조제약 벡터를 줘야 함
    # numeric_features = ["pre_glucose","carbs_log","steps","intensity","hour","weekday"]
    # 의도:
    #  - pre_glucose: 보통 높을수록 추가 상승폭은 줄 수 있음 -> -1 (약하게라도)
    #  - carbs_log: 탄수↑면 상승↑ -> +1
    #  - steps: 활동↑면 상승↓ -> -1
    #  - intensity: 활동↑면 상승↓ -> -1
    #  - hour/weekday: 단조 제약 없음 -> 0
    monotone = (-1, +1, -1, -1, 0, 0)

    model = XGBRegressor(
        n_estimators=2600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=-1,
        monotone_constraints=monotone,
    )

    return Pipeline([("pre", pre), ("model", model)])


def main():
    cfg = TrainConfig()

    print(f"[1] Load: {cfg.data_path}")
    df_raw = _safe_read_csv(cfg.data_path)

    print("[2] Prepare (CORE + logcarb)")
    df = _prepare(df_raw, cfg)
    print(f"   rows: {len(df)}  subjects: {df[cfg.group_col].nunique()}")

    train_df, test_df = _split(df, cfg)
    print(f"[3] Split by subject -> train: {len(train_df)}, test: {len(test_df)}")

    numeric_features = ["pre_glucose", "carbs_log", "steps", "intensity", "hour", "weekday"]
    categorical_features = ["meal_type"]

    # ✅ target winsorize(훈련 데이터 기준으로 컷)
    y_train_raw = train_df["delta_60"].values.astype(float)
    low = float(np.quantile(y_train_raw, cfg.clip_low_q))
    high = float(np.quantile(y_train_raw, cfg.clip_high_q))
    y_train = np.clip(y_train_raw, low, high)

    X_train = train_df[numeric_features + categorical_features]
    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df["delta_60"].values.astype(float)

    print(f"[INFO] delta_60 clip range on train: [{low:.3f}, {high:.3f}]")

    pipe = _build_pipeline(numeric_features, categorical_features, cfg)

    print("[4] Fit")
    pipe.fit(X_train, y_train)

    print("[5] Evaluate (raw test target)")
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    pred_g60 = test_df["pre_glucose"].values + pred
    true_g60 = test_df["pre_glucose"].values + y_test
    g60_mae = mean_absolute_error(true_g60, pred_g60)

    print(f"   delta_60 MAE : {mae:.3f}")
    print(f"   delta_60 RMSE: {rmse:.3f}")
    print(f"   g60 MAE      : {g60_mae:.3f}")

    os.makedirs(cfg.model_dir, exist_ok=True)
    model_path = os.path.join(cfg.model_dir, cfg.model_name)
    meta_path = os.path.join(cfg.model_dir, cfg.meta_name)

    dump(pipe, model_path)
    meta = {
        "data_path_used": cfg.data_path,
        "split": {"group_col": cfg.group_col, "test_size": cfg.test_size, "random_state": cfg.random_state},
        "features": {"numeric": numeric_features, "categorical": categorical_features},
        "target_col": "delta_60",
        "target_clip": {"low_q": cfg.clip_low_q, "high_q": cfg.clip_high_q, "low": low, "high": high},
        "metrics": {"delta60_mae": float(mae), "delta60_rmse": float(rmse), "g60_mae": float(g60_mae)},
        "model_notes": [
            "CORE features only (no confounders).",
            "carbs_log = log1p(carbs)",
            "winsorized target delta_60 on train to reduce tail-driven overprediction",
            "monotone constraints on numeric features: pre_glucose(-), carbs_log(+), steps(-), intensity(-), hour(0), weekday(0)",
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved model: {model_path}")
    print(f"[DONE] Saved meta : {meta_path}")


if __name__ == "__main__":
    main()