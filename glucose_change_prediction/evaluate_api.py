import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from paths import API_PREDICTIONS_NAME, DATASET_NAME, MODELS_DIR, PROCESSED_DIR
from service.data_utils import ALLOWED_MEAL_TYPES, prepare_service_frame
from service.metrics import evaluate_minutes, evaluate_regression
from service.targets import API_EVALUATION_TARGETS, API_RESPONSE_REQUIRED_KEYS

load_dotenv()

BASE_URL = os.environ["GLUCOSE_API_BASE_URL"]
PREDICT_PATH = "/predict-glucose"

TEST_CSV = PROCESSED_DIR / f"{DATASET_NAME}_test.csv"

OUTPUT_PRED_CSV = PROCESSED_DIR / f"{API_PREDICTIONS_NAME}.csv"
OUTPUT_METRICS_JSON = MODELS_DIR / "api_evaluation_metrics.json"

REQUEST_TIMEOUT = 10

REQUIRED_INPUT_COLS = [
    "baseline_glucose",
    "sex",
    "total_carbs",
    "total_protein",
    "total_fat",
    "total_fiber",
    "total_kcal",
    "meal_type",
]
REQUIRED_TARGET_COLS = [
    "delta_30",
    "delta_60",
    "delta_120",
    "peak_delta",
    "peak_minute",
]


def make_payload(row: pd.Series) -> dict:
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


def request_predict(payload: dict) -> dict:
    url = BASE_URL + PREDICT_PATH
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    missing = API_RESPONSE_REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Response missing keys: {sorted(missing)}")

    return data


def pick_delta(curve, minute: int):
    if not isinstance(curve, list):
        return None

    for point in curve:
        if isinstance(point, dict) and point.get("minute") == minute:
            return float(point.get("delta"))
    return None


def load_test_frame() -> pd.DataFrame:
    test_df = pd.read_csv(TEST_CSV).copy()
    test_df = prepare_service_frame(
        test_df,
        sex_missing_message="테스트 CSV에 'sex' 또는 'Gender' 컬럼이 없습니다.",
        meal_missing_message="테스트 CSV에 'meal_type' 컬럼이 없습니다.",
    )
    return test_df.dropna(subset=REQUIRED_INPUT_COLS + REQUIRED_TARGET_COLS).copy()


def collect_predictions(test_df: pd.DataFrame) -> dict[str, list]:
    predictions = {
        "pred_delta30": [],
        "pred_delta60": [],
        "pred_delta120": [],
        "pred_peakDelta": [],
        "pred_peakMinute": [],
        "server_error": [],
    }

    for idx, row in test_df.iterrows():
        payload = make_payload(row)

        try:
            result = request_predict(payload)
            curve_delta30 = pick_delta(result["curve"], 30)
            curve_delta60 = pick_delta(result["curve"], 60)
            curve_delta120 = pick_delta(result["curve"], 120)

            if curve_delta30 is None:
                raise ValueError("curve에서 minute=30 delta를 찾을 수 없습니다.")
            if curve_delta60 is None:
                raise ValueError("curve에서 minute=60 delta를 찾을 수 없습니다.")
            if curve_delta120 is None:
                raise ValueError("curve에서 minute=120 delta를 찾을 수 없습니다.")

            predictions["pred_delta30"].append(float(curve_delta30))
            predictions["pred_delta60"].append(float(curve_delta60))
            predictions["pred_delta120"].append(float(curve_delta120))
            predictions["pred_peakDelta"].append(float(result["peakDelta"]))
            predictions["pred_peakMinute"].append(int(result["peakMinute"]))
            predictions["server_error"].append("")
        except Exception as e:
            predictions["pred_delta30"].append(None)
            predictions["pred_delta60"].append(None)
            predictions["pred_delta120"].append(None)
            predictions["pred_peakDelta"].append(None)
            predictions["pred_peakMinute"].append(None)
            predictions["server_error"].append(str(e))
            print(f"[error] row index={idx}: {e}")

    return predictions


def get_valid_rows(out_df: pd.DataFrame) -> pd.DataFrame:
    valid_mask = (
        out_df["pred_delta30"].notna()
        & out_df["pred_delta60"].notna()
        & out_df["pred_delta120"].notna()
        & out_df["pred_peakDelta"].notna()
        & out_df["pred_peakMinute"].notna()
    )
    return out_df[valid_mask].copy()


def build_metrics(valid_df: pd.DataFrame, out_df: pd.DataFrame) -> dict:
    metrics = {
        "server_base_env": "GLUCOSE_API_BASE_URL",
        "predict_path": PREDICT_PATH,
        "allowed_meal_types": sorted(ALLOWED_MEAL_TYPES),
        "test_rows_total": int(len(out_df)),
        "test_rows_valid": int(len(valid_df)),
        "test_subjects": int(valid_df["subject"].nunique()) if "subject" in valid_df.columns else None,
        "results": {},
    }

    for name, config in API_EVALUATION_TARGETS.items():
        evaluator = evaluate_minutes if config["metric_kind"] == "minutes" else evaluate_regression
        true_col = config["true_col"]
        pred_col = config["pred_col"]
        metrics["results"][name] = evaluator(valid_df[true_col], valid_df[pred_col])

    return metrics


def print_metrics(metrics: dict) -> None:
    print("\n=== 서버 응답 기준 성능 ===")
    for name, result in metrics["results"].items():
        print(f"\n[{name}]")
        print(f"MAE : {result['mae']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        if "r2" in result:
            print(f"R2  : {result['r2']:.4f}")
        if "within_5min" in result:
            print(f"<=5m: {result['within_5min']:.4f}")
            print(f"<=10m: {result['within_10min']:.4f}")
            print(f"<=15m: {result['within_15min']:.4f}")


def save_outputs(out_df: pd.DataFrame, metrics: dict) -> None:
    OUTPUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(OUTPUT_PRED_CSV, index=False)

    with open(OUTPUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== 저장 완료 ===")
    print("-", OUTPUT_PRED_CSV)
    print("-", OUTPUT_METRICS_JSON)


def main():
    test_df = load_test_frame()

    print("Running API evaluation")
    print("test rows:", len(test_df))
    print("test subjects:", test_df["subject"].nunique() if "subject" in test_df.columns else "N/A")

    predictions = collect_predictions(test_df)
    out_df = test_df.copy()
    for column, values in predictions.items():
        out_df[column] = values

    valid_df = get_valid_rows(out_df)

    print("\nvalid rows:", len(valid_df))
    print("failed rows:", len(out_df) - len(valid_df))

    if valid_df.empty:
        raise RuntimeError("서버 응답이 모두 실패했습니다. 모델 경로와 서버 로그를 먼저 확인하세요.")

    metrics = build_metrics(valid_df, out_df)
    print_metrics(metrics)
    save_outputs(out_df, metrics)


if __name__ == "__main__":
    main()
