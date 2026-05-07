from pathlib import Path
from itertools import combinations
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR

DATA_PATH = BASE_DIR / "data" / "processed" / "last_final_no_activity_plus_clean_with_Gender.csv"
BIO_PATH = BASE_DIR.parent / "CGMacros" / "bio.csv"
OUTPUT_JSON = BASE_DIR / "models" / "compare_userinfo_feature_sets_results.json"

RANDOM_STATE = 42
TARGETS = ["delta_30", "delta_60", "delta_120", "peak_delta"]

BASE_FEATURES = [
    "meal_type",
    "total_kcal",
    "total_carbs",
    "effective_carbs",
    "total_protein",
    "total_fat",
    "total_fiber",
    "baseline_glucose",
]

USER_FEATURE_CANDIDATES = [
    "sex",
    "age",
    "bmi",
    "weight_kg",
    "height_cm",
]

CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "depth": 6,
    "learning_rate": 0.05,
    "iterations": 500,
    "random_seed": RANDOM_STATE,
    "verbose": False,
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_subject_split(df: pd.DataFrame, test_ratio: float = 0.3):
    if "subject" not in df.columns:
        raise ValueError("데이터셋에 'subject' 컬럼이 필요합니다.")

    subjects = sorted(df["subject"].dropna().unique().tolist())
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(subjects)

    n_test = max(1, int(round(len(subjects) * test_ratio)))
    test_subjects = set(subjects[:n_test])
    train_subjects = set(subjects[n_test:])

    train_df = df[df["subject"].isin(train_subjects)].copy()
    test_df = df[df["subject"].isin(test_subjects)].copy()

    return train_df, test_df, sorted(train_subjects), sorted(test_subjects)


def build_dataset_with_bio(data_path: Path, bio_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")
    if not bio_path.exists():
        raise FileNotFoundError(f"bio.csv not found: {bio_path}")

    df = pd.read_csv(data_path).copy()
    bio_df = pd.read_csv(bio_path).copy()

    bio_df["subject"] = bio_df["subject"].apply(
        lambda x: f"CGMacros-{int(float(x)):03d}" if pd.notna(x) else x
    )

    required_bio_cols = ["subject", "Age", "Gender", "BMI", "Body weight ", "Height "]
    missing_bio_cols = [c for c in required_bio_cols if c not in bio_df.columns]
    if missing_bio_cols:
        raise ValueError(f"bio.csv에 필요한 컬럼이 없습니다: {missing_bio_cols}")

    bio_use = bio_df[required_bio_cols].copy()
    bio_use["age"] = pd.to_numeric(bio_use["Age"], errors="coerce")
    bio_use["sex"] = bio_use["Gender"].astype(str).str.strip().str.upper()
    bio_use["bmi"] = pd.to_numeric(bio_use["BMI"], errors="coerce")
    bio_use["body_weight_lb"] = pd.to_numeric(bio_use["Body weight "], errors="coerce")
    bio_use["weight_kg"] = bio_use["body_weight_lb"] * 0.45359237
    bio_use["height_in"] = pd.to_numeric(bio_use["Height "], errors="coerce")
    bio_use["height_cm"] = bio_use["height_in"] * 2.54
    bio_use = bio_use[
        ["subject", "age", "sex", "bmi", "body_weight_lb", "weight_kg", "height_in", "height_cm"]
    ].drop_duplicates(subset=["subject"], keep="first")

    if "Gender" in df.columns:
        df = df.drop(columns=["Gender"])

    merged_df = df.merge(bio_use, on="subject", how="left")
    return merged_df


def train_and_eval_one_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target: str,
) -> Dict[str, float]:
    local_train = train_df.dropna(subset=features + [target]).copy()
    local_test = test_df.dropna(subset=features + [target]).copy()

    if len(local_train) == 0 or len(local_test) == 0:
        return {
            "mae": None,
            "rmse": None,
            "r2": None,
            "train_rows": len(local_train),
            "test_rows": len(local_test),
        }

    X_train = local_train[features].copy()
    y_train = local_train[target].copy()
    X_test = local_test[features].copy()
    y_test = local_test[target].copy()

    cat_features = [i for i, col in enumerate(features) if X_train[col].dtype == "object"]

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, cat_features=cat_features)

    pred = model.predict(X_test)

    result = evaluate(y_test, pred)
    result["train_rows"] = len(local_train)
    result["test_rows"] = len(local_test)
    return result


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    feature_sets = {"baseline": BASE_FEATURES.copy()}
    available_user_features = [c for c in USER_FEATURE_CANDIDATES if c in df.columns]

    for r in range(1, len(available_user_features) + 1):
        for combo in combinations(available_user_features, r):
            set_name = "__".join(combo)
            feature_sets[set_name] = BASE_FEATURES + list(combo)

    return feature_sets


def print_comparison(results: Dict):
    print("=" * 70)
    print("비교: baseline vs user info feature sets")
    print("=" * 70)

    for target in TARGETS:
        print(f"\n[{target}]")
        for set_name, target_result in results["results"].items():
            metrics = target_result[target]
            print(
                f"{set_name:<18} "
                f"MAE {metrics['mae']:.4f} | "
                f"RMSE {metrics['rmse']:.4f} | "
                f"R2 {metrics['r2']:.4f}"
            )

        baseline_rmse = results["results"]["baseline"][target]["rmse"]
        baseline_mae = results["results"]["baseline"][target]["mae"]

        print("-" * 70)
        for set_name, target_result in results["results"].items():
            if set_name == "baseline":
                continue
            metrics = target_result[target]
            print(
                f"{set_name:<18} "
                f"ΔRMSE {metrics['rmse'] - baseline_rmse:+.4f} | "
                f"ΔMAE {metrics['mae'] - baseline_mae:+.4f}"
            )


def main():
    df = build_dataset_with_bio(DATA_PATH, BIO_PATH)

    required_base = BASE_FEATURES + TARGETS + ["subject"]
    missing_required = [c for c in required_base if c not in df.columns]
    if missing_required:
        raise ValueError(f"데이터셋에 필요한 컬럼이 없습니다: {missing_required}")

    train_df, test_df, train_subjects, test_subjects = get_subject_split(df, test_ratio=0.3)
    feature_sets = build_feature_sets(df)

    results = {
        "data_path": str(DATA_PATH),
        "bio_path": str(BIO_PATH),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "feature_sets": feature_sets,
        "results": {},
    }

    for set_name, features in feature_sets.items():
        results["results"][set_name] = {}
        print(f"\n=== feature set: {set_name} ===")
        print("features:", features)

        for target in TARGETS:
            metrics = train_and_eval_one_target(
                train_df=train_df,
                test_df=test_df,
                features=features,
                target=target,
            )
            results["results"][set_name][target] = metrics

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_comparison(results)
    print("\n저장 완료:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
