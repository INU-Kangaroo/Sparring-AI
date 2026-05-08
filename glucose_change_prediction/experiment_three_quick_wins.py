import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from paths import DATASET_NAME, MODELS_DIR, PROCESSED_DIR
from service_data_utils import get_cat_feature_indices, prepare_service_frame


TRAIN_DATA_PATH = PROCESSED_DIR / f"{DATASET_NAME}_train.csv"
TEST_DATA_PATH = PROCESSED_DIR / f"{DATASET_NAME}_test.csv"
RESULT_JSON_PATH = MODELS_DIR / "three_quick_wins_comparison.json"
PRED_CSV_PATH = PROCESSED_DIR / "three_quick_wins_peak_predictions.csv"

BASE_FEATURES = [
    "meal_type",
    "total_kcal",
    "total_carbs",
    "effective_carbs",
    "total_protein",
    "total_fat",
    "total_fiber",
    "baseline_glucose",
    "sex",
]
EXPERIMENT_FEATURES = BASE_FEATURES + ["hour"]
REGRESSION_TARGETS = ["delta_30", "delta_60", "delta_120", "peak_delta"]
PEAK_BINS = np.asarray([30, 45, 60, 75, 90, 105, 120], dtype=int)
BASE_PARAMS_BY_TARGET = {
    "delta_30": {"iterations": 300, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7, "subsample": 1.0},
    "delta_60": {"iterations": 500, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7, "subsample": 1.0},
    "delta_120": {"iterations": 300, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3, "subsample": 0.9},
    "peak_delta": {"iterations": 300, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5, "subsample": 0.9},
    "peak_minute": {"iterations": 400, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5, "subsample": 0.9},
}
EXPERIMENT_PARAMS_BY_TARGET = {
    "delta_30": {"iterations": 350, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7, "subsample": 1.0},
    "delta_60": {"iterations": 550, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7, "subsample": 1.0},
    "delta_120": {"iterations": 350, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3, "subsample": 0.9},
    "peak_delta": {"iterations": 350, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5, "subsample": 0.9},
}
PEAK_CLASSIFIER_PARAMS = {
    "loss_function": "MultiClass",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.9,
    "random_seed": 42,
    "verbose": False,
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_minutes(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    diff = np.abs(y_true_arr - y_pred_arr)
    return {
        "mae": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "exact_match": float(np.mean(diff == 0)),
        "within_5min": float(np.mean(diff <= 5)),
        "within_10min": float(np.mean(diff <= 10)),
        "within_15min": float(np.mean(diff <= 15)),
    }
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_service_frame(
        df,
        sex_missing_message="데이터셋에 'sex' 또는 'Gender' 컬럼이 없습니다.",
        meal_missing_message="데이터셋에 'meal_type' 컬럼이 없습니다.",
    )
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    return out


def snap_to_peak_bin(values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    idx = np.abs(raw[:, None] - PEAK_BINS[None, :]).argmin(axis=1)
    snapped = PEAK_BINS[idx]
    return pd.Series(snapped, index=values.index, dtype=int)
def train_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], target: str, params: dict):
    local_train = train_df.dropna(subset=features + [target]).copy()
    local_test = test_df.dropna(subset=features + [target]).copy()

    X_train = local_train[features]
    y_train = local_train[target]
    X_test = local_test[features]
    y_test = local_test[target]

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        **params,
    )
    model.fit(X_train, y_train, cat_features=get_cat_feature_indices(local_train, features))
    pred = model.predict(X_test)
    return evaluate_regression(y_test, pred)


def train_peak_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame):
    local_train = train_df.dropna(subset=BASE_FEATURES + ["peak_minute"]).copy()
    local_test = test_df.dropna(subset=BASE_FEATURES + ["peak_minute"]).copy()

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        **BASE_PARAMS_BY_TARGET["peak_minute"],
    )
    model.fit(
        local_train[BASE_FEATURES],
        local_train["peak_minute"],
        cat_features=get_cat_feature_indices(local_train, BASE_FEATURES),
    )
    pred = model.predict(local_test[BASE_FEATURES])
    return evaluate_minutes(local_test["peak_minute"], pred), local_test, pred


def train_peak_regressor_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    exclude_censored_from_train: bool = False,
):
    local_train = train_df.dropna(subset=features + ["peak_minute"]).copy()
    local_test = test_df.dropna(subset=features + ["peak_minute"]).copy()

    if exclude_censored_from_train:
        local_train = local_train[(local_train["peak_minute"] > 1) & (local_train["peak_minute"] < 120)].copy()

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        **BASE_PARAMS_BY_TARGET["peak_minute"],
    )
    model.fit(
        local_train[features],
        local_train["peak_minute"],
        cat_features=get_cat_feature_indices(local_train, features),
    )

    pred_all = model.predict(local_test[features])
    test_uncensored = local_test[(local_test["peak_minute"] > 1) & (local_test["peak_minute"] < 120)].copy()
    pred_uncensored = model.predict(test_uncensored[features])

    return {
        "overall_raw_minutes": evaluate_minutes(local_test["peak_minute"], pred_all),
        "uncensored_raw_minutes": evaluate_minutes(test_uncensored["peak_minute"], pred_uncensored),
        "train_rows": int(len(local_train)),
        "test_rows_all": int(len(local_test)),
        "test_rows_uncensored": int(len(test_uncensored)),
        "exclude_censored_from_train": bool(exclude_censored_from_train),
    }


def train_peak_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame):
    local_train = train_df.dropna(subset=EXPERIMENT_FEATURES + ["peak_minute"]).copy()
    local_test = test_df.dropna(subset=EXPERIMENT_FEATURES + ["peak_minute"]).copy()

    train_uncensored = local_train[(local_train["peak_minute"] > 1) & (local_train["peak_minute"] < 120)].copy()
    test_uncensored = local_test[(local_test["peak_minute"] > 1) & (local_test["peak_minute"] < 120)].copy()

    train_uncensored["peak_bin"] = snap_to_peak_bin(train_uncensored["peak_minute"])
    test_uncensored["peak_bin"] = snap_to_peak_bin(test_uncensored["peak_minute"])

    model = CatBoostClassifier(**PEAK_CLASSIFIER_PARAMS)
    model.fit(
        train_uncensored[EXPERIMENT_FEATURES],
        train_uncensored["peak_bin"],
        cat_features=get_cat_feature_indices(train_uncensored, EXPERIMENT_FEATURES),
    )

    pred_all = model.predict(local_test[EXPERIMENT_FEATURES]).reshape(-1).astype(int)
    pred_uncensored = model.predict(test_uncensored[EXPERIMENT_FEATURES]).reshape(-1).astype(int)

    all_metrics = evaluate_minutes(local_test["peak_minute"], pred_all)
    uncensored_metrics = evaluate_minutes(test_uncensored["peak_minute"], pred_uncensored)
    snapped_truth_metrics = evaluate_minutes(test_uncensored["peak_bin"], pred_uncensored)
    return {
        "overall_raw_minutes": all_metrics,
        "uncensored_raw_minutes": uncensored_metrics,
        "uncensored_binned_truth": snapped_truth_metrics,
        "train_rows_uncensored": int(len(train_uncensored)),
        "test_rows_all": int(len(local_test)),
        "test_rows_uncensored": int(len(test_uncensored)),
        "censored_train_rows_removed": int(len(local_train) - len(train_uncensored)),
        "censored_test_rows_flagged": int(len(local_test) - len(test_uncensored)),
    }, local_test, pred_all


def main():
    train_df = prepare(pd.read_csv(TRAIN_DATA_PATH))
    test_df = prepare(pd.read_csv(TEST_DATA_PATH))

    baseline_results = {}
    experiment_results = {}

    for target in REGRESSION_TARGETS:
        baseline_results[target] = train_regressor(
            train_df, test_df, BASE_FEATURES, target, BASE_PARAMS_BY_TARGET[target]
        )
        experiment_results[target] = train_regressor(
            train_df, test_df, EXPERIMENT_FEATURES, target, EXPERIMENT_PARAMS_BY_TARGET[target]
        )

    baseline_peak_metrics, _, base_peak_pred = train_peak_regressor(train_df, test_df)
    peak_hour_regression = train_peak_regressor_variant(
        train_df, test_df, EXPERIMENT_FEATURES, exclude_censored_from_train=False
    )
    peak_uncensored_regression = train_peak_regressor_variant(
        train_df, test_df, BASE_FEATURES, exclude_censored_from_train=True
    )
    peak_hour_uncensored_regression = train_peak_regressor_variant(
        train_df, test_df, EXPERIMENT_FEATURES, exclude_censored_from_train=True
    )
    experiment_peak_metrics, exp_peak_test, exp_peak_pred = train_peak_classifier(train_df, test_df)

    comparison = {
        "dataset_name": DATASET_NAME,
        "train_rows_after_filter": int(len(train_df)),
        "test_rows_after_filter": int(len(test_df)),
        "baseline_features": BASE_FEATURES,
        "experiment_features": EXPERIMENT_FEATURES,
        "peak_bins": PEAK_BINS.tolist(),
        "changes_applied": [
            "hour feature added",
            "peak_minute modeled as 15-minute bin classification",
            "peak_minute values at 1 and 120 excluded from peak-time training and flagged in evaluation",
        ],
        "baseline": {
            "regression_targets": baseline_results,
            "peak_minute": baseline_peak_metrics,
        },
        "peak_minute_ablations": {
            "hour_added_regression": peak_hour_regression,
            "uncensored_train_regression": peak_uncensored_regression,
            "hour_added_uncensored_train_regression": peak_hour_uncensored_regression,
        },
        "experiment": {
            "regression_targets": experiment_results,
            "peak_minute": experiment_peak_metrics,
        },
    }

    peak_pred_df = exp_peak_test[["subject", "meal_time", "peak_minute"]].copy()
    peak_pred_df["baseline_peak_minute_pred"] = np.asarray(base_peak_pred, dtype=float)
    peak_pred_df["experiment_peak_minute_pred"] = np.asarray(exp_peak_pred, dtype=int)
    peak_pred_df["peak_minute_bin_true"] = snap_to_peak_bin(peak_pred_df["peak_minute"])
    peak_pred_df["is_censored_edge"] = (
        (peak_pred_df["peak_minute"] <= 1) | (peak_pred_df["peak_minute"] >= 120)
    )

    RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    peak_pred_df.to_csv(PRED_CSV_PATH, index=False, encoding="utf-8-sig")

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"\nSaved comparison: {RESULT_JSON_PATH}")
    print(f"Saved peak predictions: {PRED_CSV_PATH}")


if __name__ == "__main__":
    main()
