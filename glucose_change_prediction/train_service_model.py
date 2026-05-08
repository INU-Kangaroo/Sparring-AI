import pandas as pd
import joblib
from catboost import CatBoostRegressor

from paths import DATASET_NAME, MODELS_DIR, PROCESSED_DIR
from service.data_utils import ALLOWED_MEAL_TYPES, get_cat_feature_indices, prepare_service_frame, require_columns
from service.metrics import evaluate_minutes, evaluate_regression
from service.targets import MODEL_TARGETS


INPUT_CSV_PATH = PROCESSED_DIR / f"{DATASET_NAME}.csv"
TRAIN_DATA_PATH = PROCESSED_DIR / f"{DATASET_NAME}_train.csv"
TEST_DATA_PATH = PROCESSED_DIR / f"{DATASET_NAME}_test.csv"

MODEL_OUTPUT = MODELS_DIR / "service_model.joblib"

TARGETS = MODEL_TARGETS
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


def print_dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    print("Training final CatBoost models")
    print(f"input csv: {INPUT_CSV_PATH}")
    print(f"rows total : {len(full_df)}")
    print(f"rows train : {len(train_df)}")
    print(f"rows test  : {len(test_df)}")
    print(f"subjects total : {full_df['subject'].nunique()}")
    print(f"subjects train : {train_df['subject'].nunique()}")
    print(f"subjects test  : {test_df['subject'].nunique()}")
    print(f"test subject list: {TEST_SUBJECTS}")


def evaluate_target(target: str, y_test, pred) -> dict:
    if target == "peak_minute":
        return evaluate_minutes(y_test, pred)
    return evaluate_regression(y_test, pred)


def train_target_model(
    target: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_features: list[int],
):
    # 타깃별 결측 제거 뒤 동일한 피처 구조로 학습과 평가 분리
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

    # 테스트 세트 기준으로 타깃별 성능 확인
    pred = model.predict(X_test)
    result = evaluate_target(target, y_test, pred)

    print(f"MAE : {result['mae']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"R2  : {result['r2']:.4f}")
    if target == "peak_minute":
        print(f"<=5m: {result['within_5min']:.4f}")
        print(f"<=10m: {result['within_10min']:.4f}")
        print(f"<=15m: {result['within_15min']:.4f}")

    return model, result, local_test.index, y_test.values, pred


def build_dataset_stats(train_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame) -> dict:
    return {
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
    }


def save_outputs(bundle: dict) -> None:
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, MODEL_OUTPUT)

    print("\n=== 저장 완료 ===")
    print(f"- {MODEL_OUTPUT}")


def main():
    train_df = pd.read_csv(TRAIN_DATA_PATH).copy()
    test_df = pd.read_csv(TEST_DATA_PATH).copy()

    # 서비스에서 실제로 받을 범주 체계에 맞춰 학습 데이터 정리
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
    print_dataset_summary(train_df, test_df, full_df)
    models = {}
    metrics = {}
    cat_features = get_cat_feature_indices(train_df, FEATURES)

    # 응답 필드별 모델을 따로 학습해 하나의 번들로 저장
    for target in TARGETS:
        model, result, _, _, _ = train_target_model(
            target,
            train_df,
            test_df,
            cat_features,
        )
        models[target] = model
        metrics[target] = result

    bundle = {
        "features": FEATURES,
        "targets": TARGETS,
        "models": models,
    }
    save_outputs(bundle)


if __name__ == "__main__":
    main()
