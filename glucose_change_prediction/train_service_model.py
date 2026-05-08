import json

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from paths import DATASET_NAME, MODEL_METRICS_NAME, MODELS_DIR, MODEL_PREDICTIONS_NAME, PROCESSED_DIR
from service_data_utils import ALLOWED_MEAL_TYPES, get_cat_feature_indices, prepare_service_frame, require_columns


INPUT_CSV_PATH = PROCESSED_DIR / f"{DATASET_NAME}.csv"
TRAIN_DATA_PATH = PROCESSED_DIR / f"{DATASET_NAME}_train.csv"
TEST_DATA_PATH = PROCESSED_DIR / f"{DATASET_NAME}_test.csv"

MODEL_OUTPUT = MODELS_DIR / "service_model.joblib"
META_OUTPUT = MODELS_DIR / "service_model_meta.json"
METRICS_OUTPUT = MODELS_DIR / f"{MODEL_METRICS_NAME}.json"
PRED_OUTPUT = PROCESSED_DIR / f"{MODEL_PREDICTIONS_NAME}.csv"

TARGETS = ["delta_30", "delta_60", "delta_120", "peak_delta", "peak_minute"]
FEATURES = [
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

TEST_SUBJECTS = [
    "CGMacros-005",
    "CGMacros-009",
    "CGMacros-013",
    "CGMacros-028",
    "CGMacros-029",
    "CGMacros-039",
    "CGMacros-044",
    "CGMacros-046",
    "CGMacros-048",
]

PARAMS_BY_TARGET = {
    "delta_30": {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 7,
        "subsample": 1.0,
    },
    "delta_60": {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 7,
        "subsample": 1.0,
    },
    "delta_120": {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        "subsample": 0.9,
    },
    "peak_delta": {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5,
        "subsample": 0.9,
    },
    "peak_minute": {
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5,
        "subsample": 0.9,
    },
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_minutes(y_true, y_pred):
    diff = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return {
        "mae": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "exact_match": float(np.mean(diff == 0)),
        "within_5min": float(np.mean(diff <= 5)),
        "within_10min": float(np.mean(diff <= 10)),
        "within_15min": float(np.mean(diff <= 15)),
    }
def main():
    train_df = pd.read_csv(TRAIN_DATA_PATH).copy()
    test_df = pd.read_csv(TEST_DATA_PATH).copy()

    train_df = prepare_service_frame(
        train_df,
        sex_missing_message="데이터셋에 'sex' 또는 'Gender' 컬럼이 없습니다.",
        meal_missing_message="데이터셋에 'meal_type' 컬럼이 없습니다.",
    )
    test_df = prepare_service_frame(
        test_df,
        sex_missing_message="데이터셋에 'sex' 또는 'Gender' 컬럼이 없습니다.",
        meal_missing_message="데이터셋에 'meal_type' 컬럼이 없습니다.",
    )

    required = FEATURES + TARGETS
    require_columns(train_df, required, "train")
    require_columns(test_df, required, "test")
    if "subject" not in train_df.columns or "subject" not in test_df.columns:
        raise ValueError("train/test 데이터셋에 'subject' 컬럼이 필요합니다.")

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    print("Training final CatBoost models")
    print(f"input csv: {INPUT_CSV_PATH}")
    print(f"rows total : {len(full_df)}")
    print(f"rows train : {len(train_df)}")
    print(f"rows test  : {len(test_df)}")
    print(f"subjects total : {full_df['subject'].nunique()}")
    print(f"subjects train : {train_df['subject'].nunique()}")
    print(f"subjects test  : {test_df['subject'].nunique()}")
    print(f"test subject list: {TEST_SUBJECTS}")

    models = {}
    metrics = {}
    pred_base = test_df[["subject"]].copy()

    cat_features = get_cat_feature_indices(train_df, FEATURES)

    for target in TARGETS:
        local_train = train_df.dropna(subset=FEATURES + [target]).copy()
        local_test = test_df.dropna(subset=FEATURES + [target]).copy()

        X_train = local_train[FEATURES]
        y_train = local_train[target]

        X_test = local_test[FEATURES]
        y_test = local_test[target]

        params = {
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
            **PARAMS_BY_TARGET[target],
        }

        print(f"\n[target={target}]")
        print(f"params: {PARAMS_BY_TARGET[target]}")

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, cat_features=cat_features)
        models[target] = model

        pred = model.predict(X_test)
        result = evaluate_minutes(y_test, pred) if target == "peak_minute" else evaluate(y_test, pred)
        metrics[target] = result

        print(f"MAE : {result['mae']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"R2  : {result['r2']:.4f}")
        if target == "peak_minute":
            print(f"<=5m: {result['within_5min']:.4f}")
            print(f"<=10m: {result['within_10min']:.4f}")

        pred_base.loc[local_test.index, f"{target}_true"] = y_test.values
        pred_base.loc[local_test.index, f"{target}_pred"] = pred

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PRED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "features": FEATURES,
        "targets": TARGETS,
        "models": models,
    }
    joblib.dump(bundle, MODEL_OUTPUT)

    pred_base.to_csv(PRED_OUTPUT, index=False, encoding="utf-8-sig")

    metrics_payload = {
        "dataset_name": DATASET_NAME,
        "input_csv": str(INPUT_CSV_PATH.name),
        "train_data_name": TRAIN_DATA_PATH.name,
        "test_data_name": TEST_DATA_PATH.name,
        "allowed_meal_types": sorted(ALLOWED_MEAL_TYPES),
        "features": FEATURES,
        "targets": TARGETS,
        "rows_total": int(len(full_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "subjects_total": int(full_df["subject"].nunique()),
        "subjects_train": int(train_df["subject"].nunique()),
        "subjects_test": int(test_df["subject"].nunique()),
        "test_subject_list": TEST_SUBJECTS,
        "results": metrics,
    }

    meta = {
        "input_csv": str(INPUT_CSV_PATH),
        "train_data_path": str(TRAIN_DATA_PATH),
        "test_data_path": str(TEST_DATA_PATH),
        "allowed_meal_types": sorted(ALLOWED_MEAL_TYPES),
        "features": FEATURES,
        "targets": TARGETS,
        "rows_total": int(len(full_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "subjects_total": int(full_df["subject"].nunique()),
        "subjects_train": int(train_df["subject"].nunique()),
        "subjects_test": int(test_df["subject"].nunique()),
        "test_subject_list": TEST_SUBJECTS,
        "params_by_target": PARAMS_BY_TARGET,
        "test_metrics": metrics,
        "metrics_output": str(METRICS_OUTPUT),
        "model_output": str(MODEL_OUTPUT),
        "prediction_output": str(PRED_OUTPUT),
    }

    with open(META_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print("\n=== 저장 완료 ===")
    print(f"- {MODEL_OUTPUT}")
    print(f"- {META_OUTPUT}")
    print(f"- {METRICS_OUTPUT}")
    print(f"- {PRED_OUTPUT}")


if __name__ == "__main__":
    main()
