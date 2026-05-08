from pathlib import Path
import os


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"

DATASET_NAME = "meal_glucose_with_gender"
BASE_DATASET_NAME = "meal_glucose_base"
MODEL_PREDICTIONS_NAME = "service_model_predictions"
MODEL_METRICS_NAME = "service_model_metrics"
API_PREDICTIONS_NAME = "api_evaluation_predictions"


def get_cgmacros_dir() -> Path:
    env_path = os.environ.get("CGMACROS_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (PROJECT_DIR.parent / "CGMacros").resolve()


CGMACROS_DIR = get_cgmacros_dir()
BIO_CSV = CGMACROS_DIR / "bio.csv"
