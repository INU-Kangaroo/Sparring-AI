from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV_PATH = BASE_DIR / "data" / "processed" / "last_final_no_activity_plus_clean_with_Gender.csv"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "last_final_no_activity_plus_clean_with_Gender_train.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "last_final_no_activity_plus_clean_with_Gender_test.csv"

MODEL_OUTPUT = BASE_DIR / "models" / "final_catboost_service_shape_with_sex.joblib"
META_OUTPUT = BASE_DIR / "models" / "final_catboost_service_shape_with_sex_meta.json"
PRED_OUTPUT = BASE_DIR / "data" / "processed" / "final_catboost_service_shape_with_sex_predictions.csv"

TARGETS = ["delta_30", "delta_60", "delta_120", "peak_delta"]
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

ALLOWED_MEAL_TYPES = {"breakfast", "lunch", "dinner"}

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
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def normalize_sex_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sex" not in df.columns and "Gender" in df.columns:
        df["sex"] = df["Gender"]

    if "sex" not in df.columns:
        raise ValueError("데이터셋에 'sex' 또는 'Gender' 컬럼이 없습니다.")

    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            "MALE": "M",
            "FEMALE": "F",
            "남": "M",
            "여": "F",
        })
    )

    df = df[df["sex"].isin(["M", "F"])].copy()
    return df


def normalize_meal_type_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "meal_type" not in df.columns:
        raise ValueError("데이터셋에 'meal_type' 컬럼이 없습니다.")

    df["meal_type"] = df["meal_type"].astype(str).str.strip().str.lower()
    df = df[df["meal_type"].isin(ALLOWED_MEAL_TYPES)].copy()
    return df


def main():
    train_df = pd.read_csv(TRAIN_DATA_PATH).copy()
    test_df = pd.read_csv(TEST_DATA_PATH).copy()

    train_df = normalize_sex_column(train_df)
    test_df = normalize_sex_column(test_df)

    train_df = normalize_meal_type_column(train_df)
    test_df = normalize_meal_type_column(test_df)

    required = FEATURES + TARGETS
    missing_train = [c for c in required if c not in train_df.columns]
    missing_test = [c for c in required if c not in test_df.columns]

    if missing_train:
        raise ValueError(f"train 데이터셋에 필요한 컬럼이 없습니다: {missing_train}")
    if missing_test:
        raise ValueError(f"test 데이터셋에 필요한 컬럼이 없습니다: {missing_test}")
    if "subject" not in train_df.columns or "subject" not in test_df.columns:
        raise ValueError("train/test 데이터셋에 'subject' 컬럼이 필요합니다.")

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    print("=== 최종 CatBoost 학습 ===")
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

    cat_features = [i for i, c in enumerate(FEATURES) if train_df[c].dtype == "object"]

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

        print(f"\n[target={target}] 학습 중...")
        print(f"params: {PARAMS_BY_TARGET[target]}")

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, cat_features=cat_features)
        models[target] = model

        pred = model.predict(X_test)
        result = evaluate(y_test, pred)
        metrics[target] = result

        print(f"MAE : {result['mae']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"R2  : {result['r2']:.4f}")

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
        "model_output": str(MODEL_OUTPUT),
        "prediction_output": str(PRED_OUTPUT),
    }

    with open(META_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n=== 저장 완료 ===")
    print(f"- {MODEL_OUTPUT}")
    print(f"- {META_OUTPUT}")
    print(f"- {PRED_OUTPUT}")


if __name__ == "__main__":
    main()
