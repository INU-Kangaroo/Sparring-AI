import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

food_group_map_path = PROCESSED_DIR / "food_group_map.csv"
clean_foods_path = PROCESSED_DIR / "clean_foods.csv"

meal_plan_raw_path = PROCESSED_DIR / "meal_plan_raw_materials.csv"
meal_plan_foods_path = PROCESSED_DIR / "meal_plan_foods.csv"

# --------------------------------------------------
# 1) 원재료 추천용 DB 만들기
# --------------------------------------------------
raw_df = pd.read_csv(food_group_map_path)

usable_groups = ["곡류군", "어육류군", "채소군", "과일군", "우유군", "지방군"]
raw_usable = raw_df[raw_df["exchange_group"].isin(usable_groups)].copy()

# 너무 특수하거나 식단용으로 애매한 것 추가 제외
exclude_keywords_raw = [
    "추출", "분말", "가루", "수액", "성충", "유충", "번데기", "귀뚜라미", "메뚜기",
    "프로폴리스", "홍삼", "백삼", "수삼", "인삼", "솔잎", "삼백초"
]

def contains_exclude_keyword(name):
    name = str(name)
    return any(k in name for k in exclude_keywords_raw)

raw_usable = raw_usable[~raw_usable["식품명"].apply(contains_exclude_keyword)].copy()

raw_usable.to_csv(meal_plan_raw_path, index=False, encoding="utf-8-sig")

print("=" * 80)
print("[원재료 추천용 DB]")
print("저장:", meal_plan_raw_path)
print(raw_usable["exchange_group"].value_counts())
print("최종 개수:", len(raw_usable))


# --------------------------------------------------
# 2) 음식 추천용 DB 만들기
# --------------------------------------------------
food_df = pd.read_csv(clean_foods_path)

numeric_cols = [
    "에너지(kcal)", "단백질(g)", "지방(g)", "탄수화물(g)", "당류(g)", "식이섬유(g)", "나트륨(mg)"
]

for col in numeric_cols:
    food_df[col] = pd.to_numeric(food_df[col], errors="coerce")

text_cols = [
    "식품명", "식품대분류명", "식품중분류명", "식품소분류명", "식품세분류명", "대표식품명", "식품기원명"
]
for col in text_cols:
    if col in food_df.columns:
        food_df[col] = food_df[col].astype(str).str.strip()

# --------------------------------------------------
# 2-1) 제외할 음식 키워드
# --------------------------------------------------
exclude_keywords_food = [
    "튀김", "라면", "피자", "햄버거", "도넛", "케이크", "쿠키", "과자", "아이스크림",
    "사탕", "초콜릿", "탄산", "콜라", "사이다", "에이드", "쉐이크",
    "밀크티", "잼", "시럽", "캔디", "팝콘"
]

# --------------------------------------------------
# 2-2) 감점할 음식 키워드
# --------------------------------------------------
penalty_keywords_food = [
    "볶음", "부침", "전", "튀각", "덮밥", "짜장", "짬뽕", "국밥", "찌개", "탕", "국"
]

# --------------------------------------------------
# 2-3) 가점할 음식 키워드
# --------------------------------------------------
plus_keywords_food = [
    "현미", "잡곡", "오트", "귀리", "샐러드", "나물", "구이", "찜", "숙회",
    "삶은", "데친", "두부", "달걀", "계란", "닭가슴살"
]

def join_text(row):
    return " ".join([
        str(row.get("식품명", "")),
        str(row.get("대표식품명", "")),
        str(row.get("식품대분류명", "")),
        str(row.get("식품중분류명", "")),
        str(row.get("식품소분류명", "")),
        str(row.get("식품세분류명", "")),
    ])

def has_any_keyword(text, keywords):
    return any(k in text for k in keywords)

# --------------------------------------------------
# 2-4) 음식 제외 규칙
# --------------------------------------------------
def is_excluded_food(row):
    text = join_text(row)

    # 키워드 제외
    if has_any_keyword(text, exclude_keywords_food):
        return True

    # 영양값 기준 제외
    sugar = row.get("당류(g)")
    sodium = row.get("나트륨(mg)")
    fiber = row.get("식이섬유(g)")
    carb = row.get("탄수화물(g)")
    protein = row.get("단백질(g)")

    # 너무 당이 높으면 제외
    if pd.notna(sugar) and sugar >= 20:
        return True

    # 너무 나트륨 높으면 제외
    if pd.notna(sodium) and sodium >= 1000:
        return True

    # 탄수화물만 높고 단백질/식이섬유 거의 없으면 제외 후보
    if pd.notna(carb) and carb >= 50:
        if (pd.isna(protein) or protein < 3) and (pd.isna(fiber) or fiber < 2):
            return True

    return False

food_df["excluded"] = food_df.apply(is_excluded_food, axis=1)
food_usable = food_df[food_df["excluded"] == False].copy()

# --------------------------------------------------
# 2-5) 간단 점수 부여
# --------------------------------------------------
def score_food(row):
    score = 0
    text = join_text(row)

    kcal = row.get("에너지(kcal)")
    carb = row.get("탄수화물(g)")
    sugar = row.get("당류(g)")
    protein = row.get("단백질(g)")
    fiber = row.get("식이섬유(g)")
    sodium = row.get("나트륨(mg)")

    # 기본 가점
    if pd.notna(protein) and protein >= 8:
        score += 2
    if pd.notna(fiber) and fiber >= 2:
        score += 2
    if pd.notna(sugar) and sugar <= 5:
        score += 2
    if pd.notna(sodium) and sodium <= 500:
        score += 2
    if pd.notna(carb) and carb <= 35:
        score += 1

    # 너무 낮은 열량/너무 높은 열량 조정
    if pd.notna(kcal):
        if 80 <= kcal <= 350:
            score += 1
        elif kcal > 500:
            score -= 2

    # 키워드 가점
    if has_any_keyword(text, plus_keywords_food):
        score += 2

    # 키워드 감점
    if has_any_keyword(text, penalty_keywords_food):
        score -= 2

    # 영양 기반 감점
    if pd.notna(sugar) and sugar >= 10:
        score -= 2
    if pd.notna(sodium) and sodium >= 700:
        score -= 2
    if pd.notna(carb) and carb >= 45:
        score -= 1

    return score

food_usable["priority_score"] = food_usable.apply(score_food, axis=1)

# --------------------------------------------------
# 2-6) 식사 시간 힌트
# --------------------------------------------------
def infer_meal_time(row):
    text = join_text(row)

    if any(k in text for k in ["죽", "오트", "우유", "요거트", "계란", "달걀", "샌드위치"]):
        return "아침"
    if any(k in text for k in ["샐러드", "과일", "요거트", "두유"]):
        return "간식"
    if any(k in text for k in ["국", "탕", "찌개", "구이", "볶음", "밥", "덮밥", "반찬"]):
        return "점심/저녁"
    return "전체"

food_usable["meal_time_hint"] = food_usable.apply(infer_meal_time, axis=1)

# 점수순 정렬
food_usable = food_usable.sort_values(
    by=["priority_score", "단백질(g)", "식이섬유(g)"],
    ascending=[False, False, False]
).copy()

food_usable.to_csv(meal_plan_foods_path, index=False, encoding="utf-8-sig")

print("=" * 80)
print("[음식 추천용 DB]")
print("저장:", meal_plan_foods_path)
print("최종 개수:", len(food_usable))
print(food_usable[[
    "식품명", "식품대분류명", "식품중분류명",
    "에너지(kcal)", "탄수화물(g)", "당류(g)", "단백질(g)", "식이섬유(g)", "나트륨(mg)",
    "priority_score", "meal_time_hint"
]].head(30).to_string(index=False))