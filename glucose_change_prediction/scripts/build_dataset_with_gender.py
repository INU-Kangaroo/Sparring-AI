from pathlib import Path
import argparse
import json
import sys

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from paths import BASE_DATASET_NAME, BIO_CSV, CGMACROS_DIR, DATASET_NAME, PROCESSED_DIR


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

INTERMEDIATE_CLEAN_CSV = PROCESSED_DIR / f"{BASE_DATASET_NAME}.csv"
RAW_CSV = PROCESSED_DIR / f"{BASE_DATASET_NAME}_raw.csv"
REMOVED_CSV = PROCESSED_DIR / f"{BASE_DATASET_NAME}_removed.csv"
SUBJECT_SUMMARY_CSV = PROCESSED_DIR / f"{BASE_DATASET_NAME}_subject_summary.csv"

FINAL_CSV = PROCESSED_DIR / f"{DATASET_NAME}.csv"
TRAIN_CSV = PROCESSED_DIR / f"{DATASET_NAME}_train.csv"
TEST_CSV = PROCESSED_DIR / f"{DATASET_NAME}_test.csv"
SPLIT_SUMMARY_JSON = PROCESSED_DIR / f"{DATASET_NAME}_split_summary.json"
INTERMEDIATE_FILES = [
    RAW_CSV,
    REMOVED_CSV,
    SUBJECT_SUMMARY_CSV,
    SPLIT_SUMMARY_JSON,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="중간 산출물도 같이 저장합니다.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    if "Amount Consumed " in df.columns and "Amount Consumed" not in df.columns:
        rename_map["Amount Consumed "] = "Amount Consumed"
    if "Intensity" in df.columns and "METs" not in df.columns:
        rename_map["Intensity"] = "METs"

    if rename_map:
        df = df.rename(columns=rename_map)

    required = [
        "Timestamp",
        "Libre GL",
        "Dexcom GL",
        "Meal Type",
        "Calories",
        "Carbs",
        "Protein",
        "Fat",
        "Fiber",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    return df


def add_glucose_column(df: pd.DataFrame) -> pd.DataFrame:
    df["glucose"] = pd.to_numeric(df["Dexcom GL"], errors="coerce").combine_first(
        pd.to_numeric(df["Libre GL"], errors="coerce")
    )
    return df


def normalize_meal_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    mapping = {
        "breakfast": "breakfast",
        "lunch": "lunch",
        "dinner": "dinner",
        "snack": "snack",
        "snacks": "snack",
        "snack 1": "snack",
        "snack1": "snack",
    }
    return mapping.get(s, s)


def extract_meal_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Calories", "Carbs", "Protein", "Fat", "Fiber"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    meal_mask = (
        df["Meal Type"].notna()
        & (
            df["Calories"].fillna(0).gt(0)
            | df["Carbs"].fillna(0).gt(0)
        )
    )

    meals = df.loc[meal_mask].copy()
    meals["meal_type"] = meals["Meal Type"].apply(normalize_meal_label)
    meals["Timestamp"] = pd.to_datetime(meals["Timestamp"], errors="coerce")
    meals = meals.dropna(subset=["Timestamp", "meal_type"])

    return meals


def drop_exact_meal_duplicates(meals: pd.DataFrame) -> pd.DataFrame:
    meals = meals.copy()

    dedup_cols = [
        "Timestamp",
        "meal_type",
        "Calories",
        "Carbs",
        "Protein",
        "Fat",
        "Fiber",
    ]

    meals = meals.drop_duplicates(subset=dedup_cols, keep="first").copy()
    return meals


def collapse_meals(meals: pd.DataFrame, subject_id: str, merge_minutes: int = 20) -> pd.DataFrame:
    meals = meals.sort_values("Timestamp").copy()

    for col in ["Calories", "Carbs", "Protein", "Fat", "Fiber"]:
        meals[col] = pd.to_numeric(meals[col], errors="coerce").fillna(0.0)

    rows = []
    current = None

    for _, row in meals.iterrows():
        row_ts = row["Timestamp"]
        if pd.isna(row_ts):
            continue

        if current is None:
            current = {
                "subject": subject_id,
                "meal_time": row_ts,
                "meal_type": row["meal_type"],
                "total_kcal": float(row["Calories"]),
                "total_carbs": float(row["Carbs"]),
                "total_protein": float(row["Protein"]),
                "total_fat": float(row["Fat"]),
                "total_fiber": float(row["Fiber"]),
            }
            continue

        same_type = row["meal_type"] == current["meal_type"]
        close_in_time = (row_ts - current["meal_time"]).total_seconds() / 60 <= merge_minutes

        if same_type and close_in_time:
            current["total_kcal"] += float(row["Calories"])
            current["total_carbs"] += float(row["Carbs"])
            current["total_protein"] += float(row["Protein"])
            current["total_fat"] += float(row["Fat"])
            current["total_fiber"] += float(row["Fiber"])
        else:
            rows.append(current)
            current = {
                "subject": subject_id,
                "meal_time": row_ts,
                "meal_type": row["meal_type"],
                "total_kcal": float(row["Calories"]),
                "total_carbs": float(row["Carbs"]),
                "total_protein": float(row["Protein"]),
                "total_fat": float(row["Fat"]),
                "total_fiber": float(row["Fiber"]),
            }

    if current is not None:
        rows.append(current)

    return pd.DataFrame(rows)


def nearest_glucose(df: pd.DataFrame, target_ts: pd.Timestamp, tol_min: int):
    window = df[
        (df["Timestamp"] >= target_ts - pd.Timedelta(minutes=tol_min))
        & (df["Timestamp"] <= target_ts + pd.Timedelta(minutes=tol_min))
    ].copy()

    if window.empty:
        return np.nan

    idx = (window["Timestamp"] - target_ts).abs().idxmin()
    return float(window.loc[idx, "glucose"])


def build_training_rows(subject_df: pd.DataFrame, meal_df: pd.DataFrame) -> pd.DataFrame:
    subject_df = subject_df.sort_values("Timestamp").copy()
    subject_df["Timestamp"] = pd.to_datetime(subject_df["Timestamp"], errors="coerce")
    subject_df["glucose"] = pd.to_numeric(subject_df["glucose"], errors="coerce")
    subject_df = subject_df.dropna(subset=["Timestamp", "glucose"]).copy()

    rows = []

    for _, meal in meal_df.iterrows():
        t = meal["meal_time"]

        pre30 = subject_df[
            (subject_df["Timestamp"] >= t - pd.Timedelta(minutes=30))
            & (subject_df["Timestamp"] < t)
        ].copy()

        if len(pre30.dropna(subset=["glucose"])) < 10:
            continue

        pre15 = pre30[pre30["Timestamp"] >= t - pd.Timedelta(minutes=15)].copy()
        if len(pre15.dropna(subset=["glucose"])) < 5:
            continue

        baseline = float(pre15["glucose"].mean())

        g30 = nearest_glucose(subject_df, t + pd.Timedelta(minutes=30), tol_min=12)
        g60 = nearest_glucose(subject_df, t + pd.Timedelta(minutes=60), tol_min=15)
        g120 = nearest_glucose(subject_df, t + pd.Timedelta(minutes=120), tol_min=20)

        post120 = subject_df[
            (subject_df["Timestamp"] > t)
            & (subject_df["Timestamp"] <= t + pd.Timedelta(minutes=120))
        ].copy()

        if post120.dropna(subset=["glucose"]).empty:
            continue

        peak_idx = post120["glucose"].idxmax()
        peak_row = post120.loc[peak_idx]
        peak_glucose = float(peak_row["glucose"])
        peak_minute = int(round((peak_row["Timestamp"] - t).total_seconds() / 60.0))

        if pd.isna(g30) or pd.isna(g60) or pd.isna(g120):
            continue

        hour = int(pd.to_datetime(t).hour)
        effective_carbs = max(float(meal["total_carbs"]) - float(meal["total_fiber"]), 0.0)

        rows.append({
            "subject": meal["subject"],
            "meal_time": t,
            "meal_type": meal["meal_type"],
            "hour": hour,
            "total_kcal": float(meal["total_kcal"]),
            "total_carbs": float(meal["total_carbs"]),
            "effective_carbs": float(effective_carbs),
            "total_protein": float(meal["total_protein"]),
            "total_fat": float(meal["total_fat"]),
            "total_fiber": float(meal["total_fiber"]),
            "baseline_glucose": float(baseline),
            "delta_30": float(g30 - baseline),
            "delta_60": float(g60 - baseline),
            "delta_120": float(g120 - baseline),
            "peak_delta": float(peak_glucose - baseline),
            "peak_minute": peak_minute,
        })

    return pd.DataFrame(rows)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    required_cols = [
        "subject",
        "meal_time",
        "meal_type",
        "hour",
        "total_kcal",
        "total_carbs",
        "effective_carbs",
        "total_protein",
        "total_fat",
        "total_fiber",
        "baseline_glucose",
        "delta_30",
        "delta_60",
        "delta_120",
        "peak_delta",
        "peak_minute",
    ]
    df = df.dropna(subset=required_cols).copy()

    expected_kcal = (
        4 * df["total_carbs"] +
        4 * df["total_protein"] +
        9 * df["total_fat"]
    )
    kcal_ratio = df["total_kcal"] / expected_kcal.replace(0, np.nan)

    keep_mask = (
        df["total_kcal"].between(30, 1180)
        & df["total_carbs"].between(0, 176)
        & df["effective_carbs"].between(0, 176)
        & df["total_protein"].between(3, 176)
        & df["total_fat"].between(0, 176)
        & df["total_fiber"].between(0, 176)
        & (df["total_fiber"] <= df["total_carbs"])
        & kcal_ratio.between(0.6, 1.8)
        & df["baseline_glucose"].between(40, 400)
        & df["hour"].between(0, 23)
        & df["peak_minute"].between(1, 120)
    )

    clean = df.loc[keep_mask].copy()
    removed = df.loc[~keep_mask].copy()

    return clean, removed


def build_base_dataset(save_intermediate: bool) -> pd.DataFrame:
    csv_files = sorted(
        p for p in CGMACROS_DIR.rglob("*.csv")
        if p.stem.startswith("CGMacros-")
    )

    if not csv_files:
        raise RuntimeError(f"No CGMacros CSV files found under: {CGMACROS_DIR}")

    print("Building base dataset from CGMacros")
    print(f"found csv files: {len(csv_files)}")

    all_rows = []
    summary_rows = []

    for csv_path in csv_files:
        subject_id = csv_path.stem

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[skip] {csv_path.name}: read error -> {e}")
            continue

        df = normalize_columns(df)
        df = add_glucose_column(df)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        meals = extract_meal_rows(df)
        meals_before = len(meals)

        meals = drop_exact_meal_duplicates(meals)
        meals_after_dedup = len(meals)

        collapsed = collapse_meals(meals, subject_id, merge_minutes=20)
        rows = build_training_rows(df, collapsed) if not collapsed.empty else pd.DataFrame()

        summary_rows.append({
            "subject": subject_id,
            "raw_rows": len(df),
            "meal_rows_before_dedup": meals_before,
            "meal_rows_after_dedup": meals_after_dedup,
            "collapsed_meals": len(collapsed),
            "train_rows": len(rows),
        })

        print(
            f"{subject_id}: raw={len(df)}, "
            f"meals_before={meals_before}, "
            f"meals_after_dedup={meals_after_dedup}, "
            f"collapsed={len(collapsed)}, "
            f"train={len(rows)}"
        )

        if not rows.empty:
            all_rows.append(rows)

    if not all_rows:
        raise RuntimeError("No training rows were created.")

    final_df = pd.concat(all_rows, ignore_index=True)
    clean_df, removed_df = clean_dataset(final_df)

    if save_intermediate:
        clean_df.to_csv(INTERMEDIATE_CLEAN_CSV, index=False)
        final_df.to_csv(RAW_CSV, index=False)
        removed_df.to_csv(REMOVED_CSV, index=False)
        pd.DataFrame(summary_rows).to_csv(SUBJECT_SUMMARY_CSV, index=False)

    if save_intermediate:
        print("\n=== 저장 완료 ===")
        print("-", INTERMEDIATE_CLEAN_CSV)
        print("-", RAW_CSV)
        print("-", REMOVED_CSV)
        print("-", SUBJECT_SUMMARY_CSV)

    print("\nBase dataset summary")
    print("raw rows:", len(final_df))
    print("clean rows:", len(clean_df))
    print("removed rows:", len(removed_df))
    print("subjects(raw):", final_df["subject"].nunique())
    print("subjects(clean):", clean_df["subject"].nunique())
    return clean_df


def load_gender_map() -> pd.DataFrame:
    if not BIO_CSV.exists():
        raise FileNotFoundError(f"bio 파일이 없습니다: {BIO_CSV}")

    bio_df = pd.read_csv(BIO_CSV).copy()

    required_cols = ["subject", "Gender"]
    missing_cols = [col for col in required_cols if col not in bio_df.columns]
    if missing_cols:
        raise KeyError(f"bio.csv에 필요한 컬럼이 없습니다: {missing_cols}")

    gender_df = bio_df[required_cols].copy()
    gender_df["subject"] = gender_df["subject"].apply(
        lambda x: f"CGMacros-{int(float(x)):03d}" if pd.notna(x) else x
    )
    gender_df["Gender"] = gender_df["Gender"].astype(str).str.strip().str.upper()
    gender_df = gender_df.drop_duplicates(subset=["subject"], keep="first")
    return gender_df


def attach_gender(final_df: pd.DataFrame) -> pd.DataFrame:
    print("\nAttaching gender column")

    final_df = final_df.copy()
    gender_df = load_gender_map()

    if "subject" not in final_df.columns:
        raise KeyError("'subject' 컬럼이 없습니다.")

    if "Gender" in final_df.columns:
        final_df = final_df.drop(columns=["Gender"])

    merged_df = final_df.merge(gender_df, on="subject", how="left")
    merged_df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")

    print(f"입력 rows  : {len(final_df)}")
    print(f"결과 rows  : {len(merged_df)}")
    print(f"저장 파일  : {FINAL_CSV}")
    print("Gender 분포:")
    print(merged_df["Gender"].value_counts(dropna=False))
    return merged_df


def split_dataset(df: pd.DataFrame, save_intermediate: bool):
    print("\nSplitting train/test")

    df = df.copy()

    if "subject" not in df.columns:
        raise KeyError("'subject' 컬럼이 없습니다.")

    test_mask = df["subject"].isin(TEST_SUBJECTS)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    found_test_subjects = sorted(test_df["subject"].dropna().unique().tolist())
    missing_test_subjects = sorted(list(set(TEST_SUBJECTS) - set(found_test_subjects)))
    extra_test_subjects = sorted(list(set(found_test_subjects) - set(TEST_SUBJECTS)))

    summary = {
        "input_path": str(FINAL_CSV),
        "train_path": str(TRAIN_CSV),
        "test_path": str(TEST_CSV),
        "total_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "total_subject_count": int(df["subject"].nunique()),
        "train_subject_count": int(train_df["subject"].nunique()),
        "test_subject_count": int(test_df["subject"].nunique()),
        "requested_test_subjects": TEST_SUBJECTS,
        "found_test_subjects": found_test_subjects,
        "missing_test_subjects": missing_test_subjects,
        "extra_test_subjects": extra_test_subjects,
    }

    print(f"train rows : {len(train_df)}")
    print(f"test rows  : {len(test_df)}")
    print(f"train file : {TRAIN_CSV}")
    print(f"test file  : {TEST_CSV}")
    if save_intermediate:
        with open(SPLIT_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"summary    : {SPLIT_SUMMARY_JSON}")

    if missing_test_subjects:
        print("[주의] test subject 목록 중 데이터셋에 없는 subject:")
        print(missing_test_subjects)


def remove_intermediate_files():
    for path in INTERMEDIATE_FILES:
        if path.exists():
            path.unlink()


def main(save_intermediate: bool = False):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    clean_df = build_base_dataset(save_intermediate=save_intermediate)
    merged_df = attach_gender(clean_df)
    split_dataset(merged_df, save_intermediate=save_intermediate)

    if not save_intermediate:
        remove_intermediate_files()

    print("\nDone")
    print(FINAL_CSV)
    print(TRAIN_CSV)
    print(TEST_CSV)


if __name__ == "__main__":
    args = parse_args()
    main(save_intermediate=args.save_intermediate)
