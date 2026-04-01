import pandas as pd
from pathlib import Path

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

food_path = RAW_DIR / "foods.csv"
raw_rda_path = RAW_DIR / "raw_materials_rda.csv"
raw_nifs_path = RAW_DIR / "raw_materials_nifs.csv"

KEEP_RAW_COLS = [
    "식품코드", "식품명", "데이터구분명", "식품기원명",
    "식품대분류명", "대표식품명", "식품중분류명", "식품소분류명", "식품세분류명",
    "영양성분함량기준량",
    "에너지(kcal)", "단백질(g)", "지방(g)", "탄수화물(g)", "당류(g)", "식이섬유(g)", "나트륨(mg)",
    "폐기율(%)", "출처명", "수입여부", "원산지국명", "원산지역명", "생산·채취·포획월",
    "데이터생성방법명", "데이터기준일자"
]

KEEP_FOOD_COLS = [
    "식품코드", "식품명", "데이터구분명", "식품기원명",
    "식품대분류명", "대표식품명", "식품중분류명", "식품소분류명", "식품세분류명",
    "영양성분함량기준량",
    "에너지(kcal)", "단백질(g)", "지방(g)", "탄수화물(g)", "당류(g)", "식이섬유(g)", "나트륨(mg)",
    "식품중량", "출처명", "데이터생성방법명", "데이터기준일자"
]

NUMERIC_COLS_RAW = [
    "에너지(kcal)", "단백질(g)", "지방(g)", "탄수화물(g)", "당류(g)", "식이섬유(g)", "나트륨(mg)", "폐기율(%)", "생산·채취·포획월"
]

NUMERIC_COLS_FOOD = [
    "에너지(kcal)", "단백질(g)", "지방(g)", "탄수화물(g)", "당류(g)", "식이섬유(g)", "나트륨(mg)"
]

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")

def clean_text(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    return df

def clean_numeric(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# -------------------------
# 1) 원재료성식품 합치기
# -------------------------
raw_rda = load_csv(raw_rda_path)[KEEP_RAW_COLS].copy()
raw_nifs = load_csv(raw_nifs_path)[KEEP_RAW_COLS].copy()

raw_rda["source_file"] = "raw_materials_rda"
raw_nifs["source_file"] = "raw_materials_nifs"

raw_df = pd.concat([raw_rda, raw_nifs], ignore_index=True)

raw_df = clean_text(raw_df, [
    "식품코드", "식품명", "데이터구분명", "식품기원명",
    "식품대분류명", "대표식품명", "식품중분류명", "식품소분류명", "식품세분류명",
    "영양성분함량기준량", "출처명", "수입여부", "원산지국명", "원산지역명",
    "데이터생성방법명", "데이터기준일자", "source_file"
])
raw_df = clean_numeric(raw_df, NUMERIC_COLS_RAW)

# 완전 중복 제거
raw_df = raw_df.drop_duplicates()

# 식품명 + 기준량 + 출처명 기준 중복 1차 제거
raw_df = raw_df.drop_duplicates(subset=["식품명", "영양성분함량기준량", "출처명"])

raw_df.to_csv(PROCESSED_DIR / "clean_raw_materials.csv", index=False, encoding="utf-8-sig")

# -------------------------
# 2) 음식 데이터 정리
# -------------------------
food_df = load_csv(food_path)[KEEP_FOOD_COLS].copy()

food_df = clean_text(food_df, [
    "식품코드", "식품명", "데이터구분명", "식품기원명",
    "식품대분류명", "대표식품명", "식품중분류명", "식품소분류명", "식품세분류명",
    "영양성분함량기준량", "식품중량", "출처명", "데이터생성방법명", "데이터기준일자"
])
food_df = clean_numeric(food_df, NUMERIC_COLS_FOOD)

food_df = food_df.drop_duplicates()
food_df = food_df.drop_duplicates(subset=["식품명", "영양성분함량기준량", "출처명"])

food_df.to_csv(PROCESSED_DIR / "clean_foods.csv", index=False, encoding="utf-8-sig")

print("완료:")
print("-", PROCESSED_DIR / "clean_raw_materials.csv")
print("-", PROCESSED_DIR / "clean_foods.csv")
print("raw shape:", raw_df.shape)
print("food shape:", food_df.shape)