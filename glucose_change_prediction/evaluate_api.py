import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from paths import API_PREDICTIONS_NAME, DATASET_NAME, MODELS_DIR, PROCESSED_DIR
from service_data_utils import ALLOWED_MEAL_TYPES, prepare_service_frame

load_dotenv()

BASE_URL = os.environ["GLUCOSE_API_BASE_URL"]
PREDICT_PATH = "/predict-glucose"

TEST_CSV = PROCESSED_DIR / f"{DATASET_NAME}_test.csv"

OUTPUT_PRED_CSV = PROCESSED_DIR / f"{API_PREDICTIONS_NAME}.csv"
OUTPUT_METRICS_JSON = MODELS_DIR / "api_evaluation_metrics.json"

REQUEST_TIMEOUT = 10
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def evaluate_minutes(y_true, y_pred):
    diff = (y_true - y_pred).abs()
    return {
        "mae": float(diff.mean()),
        "rmse": float((mean_squared_error(y_true, y_pred) ** 0.5)),
        "exact_match": float((diff == 0).mean()),
        "within_5min": float((diff <= 5).mean()),
        "within_10min": float((diff <= 10).mean()),
        "within_15min": float((diff <= 15).mean()),
    }
def row_to_request_payload(row: pd.Series) -> dict:
    sex = str(row["sex"]).strip().upper()
    if sex not in {"M", "F"}:
        raise ValueError(f"잘못된 sex 값: {sex}")

    return {
        "baselineGlucose": float(row["baseline_glucose"]),
        "sex": sex,
        "meal": {
            "carbs": float(row["total_carbs"]),
            "protein": float(row["total_protein"]),
            "fat": float(row["total_fat"]),
            "fiber": float(row["total_fiber"]),
            "kcal": float(row["total_kcal"]),
            "mealType": str(row["meal_type"]).strip().lower(),
        }
    }


def call_server(payload: dict) -> dict:
    url = BASE_URL + PREDICT_PATH
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    required_keys = {"delta30", "delta60", "peakDelta", "peakMinute", "curve"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Response missing keys: {sorted(missing)}")

    return data


def extract_curve_delta120(curve):
    if not isinstance(curve, list):
        return None

    for point in curve:
        if isinstance(point, dict) and point.get("minute") == 120:
            return float(point.get("delta"))
    return None


def main():
    test_df = pd.read_csv(TEST_CSV).copy()
    test_df = prepare_service_frame(
        test_df,
        sex_missing_message="테스트 CSV에 'sex' 또는 'Gender' 컬럼이 없습니다.",
        meal_missing_message="테스트 CSV에 'meal_type' 컬럼이 없습니다.",
    )

    required_input_cols = [
        "baseline_glucose",
        "sex",
        "total_carbs",
        "total_protein",
        "total_fat",
        "total_fiber",
        "total_kcal",
        "meal_type",
    ]
    required_target_cols = [
        "delta_30",
        "delta_60",
        "delta_120",
        "peak_delta",
        "peak_minute",
    ]

    test_df = test_df.dropna(subset=required_input_cols + required_target_cols).copy()

    print("Running API evaluation")
    print("test rows:", len(test_df))
    print("test subjects:", test_df["subject"].nunique() if "subject" in test_df.columns else "N/A")

    pred_delta30 = []
    pred_delta60 = []
    pred_delta120 = []
    pred_peakDelta = []
    peak_minutes = []
    errors = []

    for idx, row in test_df.iterrows():
        payload = row_to_request_payload(row)

        try:
            result = call_server(payload)
            curve_delta120 = extract_curve_delta120(result["curve"])
            if curve_delta120 is None:
                raise ValueError("curve에서 minute=120 delta를 찾을 수 없습니다.")

            pred_delta30.append(float(result["delta30"]))
            pred_delta60.append(float(result["delta60"]))
            pred_delta120.append(curve_delta120)
            pred_peakDelta.append(float(result["peakDelta"]))
            peak_minutes.append(int(result["peakMinute"]))
            errors.append("")
        except Exception as e:
            pred_delta30.append(None)
            pred_delta60.append(None)
            pred_delta120.append(None)
            pred_peakDelta.append(None)
            peak_minutes.append(None)
            errors.append(str(e))
            print(f"[error] row index={idx}: {e}")

    out_df = test_df.copy()
    out_df["pred_delta30"] = pred_delta30
    out_df["pred_delta60"] = pred_delta60
    out_df["pred_delta120"] = pred_delta120
    out_df["pred_peakDelta"] = pred_peakDelta
    out_df["pred_peakMinute"] = peak_minutes
    out_df["server_error"] = errors

    valid_df = out_df[
        out_df["pred_delta30"].notna()
        & out_df["pred_delta60"].notna()
        & out_df["pred_delta120"].notna()
        & out_df["pred_peakDelta"].notna()
        & out_df["pred_peakMinute"].notna()
    ].copy()

    print("\nvalid rows:", len(valid_df))
    print("failed rows:", len(out_df) - len(valid_df))

    if valid_df.empty:
        raise RuntimeError("서버 응답이 모두 실패했습니다. 모델 경로와 서버 로그를 먼저 확인하세요.")

    metrics = {
        "server_base_env": "GLUCOSE_API_BASE_URL",
        "predict_path": PREDICT_PATH,
        "allowed_meal_types": sorted(ALLOWED_MEAL_TYPES),
        "test_rows_total": int(len(out_df)),
        "test_rows_valid": int(len(valid_df)),
        "test_subjects": int(valid_df["subject"].nunique()) if "subject" in valid_df.columns else None,
        "results": {},
    }
    target_map = {
        "delta30": ("delta_30", "pred_delta30", evaluate_regression),
        "delta60": ("delta_60", "pred_delta60", evaluate_regression),
        "delta120": ("delta_120", "pred_delta120", evaluate_regression),
        "peakDelta": ("peak_delta", "pred_peakDelta", evaluate_regression),
        "peakMinute": ("peak_minute", "pred_peakMinute", evaluate_minutes),
    }

    print("\n=== 서버 응답 기준 성능 ===")
    for name, (true_col, pred_col, evaluator) in target_map.items():
        result = evaluator(valid_df[true_col], valid_df[pred_col])
        metrics["results"][name] = result

        print(f"\n[{name}]")
        print(f"MAE : {result['mae']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        if "r2" in result:
            print(f"R2  : {result['r2']:.4f}")
        else:
            print(f"<=5m: {result['within_5min']:.4f}")
            print(f"<=10m: {result['within_10min']:.4f}")

    OUTPUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(OUTPUT_PRED_CSV, index=False)

    with open(OUTPUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== 저장 완료 ===")
    print("-", OUTPUT_PRED_CSV)
    print("-", OUTPUT_METRICS_JSON)


if __name__ == "__main__":
    main()
