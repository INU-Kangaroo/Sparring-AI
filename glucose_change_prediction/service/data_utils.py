import pandas as pd


ALLOWED_MEAL_TYPES = {"breakfast", "lunch", "dinner"}
SEX_ALIASES = {
    "MALE": "M",
    "FEMALE": "F",
    "남": "M",
    "여": "F",
}


def normalize_sex(df: pd.DataFrame, *, missing_message: str) -> pd.DataFrame:
    out = df.copy()

    # 입력 데이터마다 다른 성별 컬럼 이름 흡수
    if "sex" not in out.columns and "Gender" in out.columns:
        out["sex"] = out["Gender"]

    if "sex" not in out.columns:
        raise ValueError(missing_message)

    out["sex"] = (
        out["sex"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(SEX_ALIASES)
    )
    return out[out["sex"].isin(["M", "F"])].copy()


def keep_supported_meals(df: pd.DataFrame, *, missing_message: str) -> pd.DataFrame:
    out = df.copy()

    if "meal_type" not in out.columns:
        raise ValueError(missing_message)

    # 서비스에서 사용하는 식사 유형만 남겨 학습과 추론 기준 일치
    out["meal_type"] = out["meal_type"].astype(str).str.strip().str.lower()
    return out[out["meal_type"].isin(ALLOWED_MEAL_TYPES)].copy()


def prepare_service_frame(
    df: pd.DataFrame,
    *,
    sex_missing_message: str,
    meal_missing_message: str,
) -> pd.DataFrame:
    # 서비스 모델이 기대하는 범주 체계로 입력 데이터 정리
    out = normalize_sex(df, missing_message=sex_missing_message)
    out = keep_supported_meals(out, missing_message=meal_missing_message)
    return out


def require_columns(df: pd.DataFrame, columns: list[str], label: str):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} 데이터셋에 필요한 컬럼이 없습니다: {missing}")


def get_cat_feature_indices(df: pd.DataFrame, features: list[str]) -> list[int]:
    return [i for i, col in enumerate(features) if df[col].dtype == "object"]
