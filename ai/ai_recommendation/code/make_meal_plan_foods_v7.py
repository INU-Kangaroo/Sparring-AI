import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

in_path = PROCESSED_DIR / "clean_foods.csv"
out_path = PROCESSED_DIR / "meal_plan_foods_cleaned.csv"

df = pd.read_csv(in_path)

numeric_cols = [
    "에너지(kcal)", "단백질(g)", "지방(g)", "탄수화물(g)",
    "당류(g)", "식이섬유(g)", "나트륨(mg)"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

text_cols = [
    "식품명", "대표식품명", "식품대분류명",
    "식품중분류명", "식품소분류명", "식품세분류명"
]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

def join_text(row):
    return " ".join([
        str(row.get("식품명", "")),
        str(row.get("대표식품명", "")),
        str(row.get("식품대분류명", "")),
        str(row.get("식품중분류명", "")),
        str(row.get("식품소분류명", "")),
        str(row.get("식품세분류명", "")),
    ])

def has_any(text, keywords):
    text = str(text)
    return any(k in text for k in keywords)

exclude_big_categories = [
    "음료 및 차류",
    "유제품류 및 빙과류",
]

# 제거: 음료/간식/디저트/패스트푸드/튀김/복합형 밥/간편조리세트/치킨/탕평채
exclude_keywords = [
    "과일", "사과", "바나나", "딸기", "귤", "오렌지", "키위", "포도", "복숭아", "자몽",
    "주스", "쥬스", "스무디", "라떼", "커피", "차", "에이드", "쉐이크", "빙수", "샤베트",
    "요거트", "요구르트", "우유", "치즈",
    "과자", "쿠키", "비스킷", "케이크", "도넛", "초콜릿", "사탕", "젤리", "푸딩",
    "머핀", "와플", "토스트",
    "피자", "햄버거", "핫도그", "샌드위치",
    "라면", "컵라면",
    "튀김", "돈까스", "치킨까스", "가스", "탕수육", "강정", "후라이드",
    "덮밥", "초밥", "볶음밥", "비빔밥", "김밥", "주먹밥", "오므라이스", "리조또",
    "짜장밥", "자장밥", "카레라이스", "하이라이스", "잡채밥", "잡탕밥", "알밥", "국밥",
    "연어롤", "캘리포니아롤",
    "간편조리세트", "치킨", "탕평채"
]

allow_food_names = {
    "달걀부침(달걀후라이)",
    "달걀후라이",
    "달걀말이",
    "달걀말이_채소",
    "두부부침",
    "두부전",
    "굴전",
}

allow_rep_names = {
    "달걀부침(달걀후라이)",
    "달걀후라이",
    "달걀말이",
    "두부부침",
    "두부전",
    "굴전",
}

allow_big_categories = {
    "김치류",
}

vegetable_main_keywords = [
    "나물", "숙채", "생채", "겉절이", "김치",
    "시금치", "콩나물", "숙주", "애호박", "버섯",
    "배추", "양배추", "열무", "오이", "상추", "브로콜리",
    "토란대", "미나리", "고사리", "도라지", "취나물",
    "버섯구이", "버섯볶음", "채소볶음", "양상추샐러드", "샐러드_양상추"
]

vegetable_exclude_keywords = [
    "고기", "닭", "소고기", "돼지고기", "개고기",
    "주꾸미", "오징어", "문어", "낙지", "굴", "조개", "피조개",
    "참치", "연어", "물회", "해물",
    "두부김치", "마파두부", "잡채", "샐러드_닭가슴살", "닭가슴살샐러드",
    "두부", "양념두부", "탕평채"
]

protein_rep_keywords = [
    "구이", "찜", "조림", "수육", "불고기",
    "두부", "양념두부", "두부조림", "마파두부",
    "달걀", "계란", "닭", "소고기", "돼지고기",
    "생선", "연어", "고등어", "갈치", "참치", "오징어", "문어",
    "새우", "조개", "전복", "굴", "꼬막", "북어", "황태"
]

high_fat_keywords = [
    "족발", "갈비", "장어", "도가니", "곱창", "순대", "삼겹살"
]

# 제거: 복합 샐러드/토핑 샐러드/단백질 섞인 샐러드/밥 양념 변형
composite_salad_keywords = [
    "닭고기 샐러드", "달걀 샐러드", "감자샐러드", "단호박 샐러드",
    "멕시컨샐러드", "옥수수샐러드", "포테이토", "튜나샐러드",
    "쉬림프", "에그샐러드", "모짜렐라", "콥볼샐러드", "콥샐러드"
]

composite_rice_keywords = [
    "비빔 잡곡밥",
    "콩나물밥_소고기", "콩나물밥_돼지고기", "콩나물밥_양념장",
    "곤드레밥_양념장"
]

composite_other_keywords = [
    "돼지고기 피망잡채"
]

def detect_bucket(row):
    big = str(row.get("식품대분류명", ""))
    rep = str(row.get("대표식품명", ""))
    food = str(row.get("식품명", ""))
    mid = str(row.get("식품중분류명", ""))
    text = join_text(row)

    sugar = row.get("당류(g)")
    sodium = row.get("나트륨(mg)")
    fat = row.get("지방(g)")
    carb = row.get("탄수화물(g)")
    protein = row.get("단백질(g)")
    fiber = row.get("식이섬유(g)")

    if ("달걀말이" in food or "달걀말이" in rep) and ("햄" in food or "햄" in rep or "햄" in mid or "햄" in text):
        return "제외", "햄 들어간 달걀말이 제외"

    if has_any(text, composite_salad_keywords):
        return "제외", "복합 샐러드 제외"

    if has_any(text, composite_rice_keywords):
        return "제외", "복합 밥류 제외"

    if has_any(text, composite_other_keywords):
        return "제외", "복합 음식 제외"

    if "탕평채" in text:
        return "제외", "탕평채 제외"

    if big in allow_big_categories or food in allow_food_names or rep in allow_rep_names:
        if big == "김치류":
            return "채소군후보", "김치류"
        if food in allow_food_names or rep in allow_rep_names:
            return "어육류군후보", "허용 전/부침"

    if big in exclude_big_categories:
        return "제외", "제외 대분류"

    if has_any(text, exclude_keywords):
        return "제외", "제외 키워드"

    if "스프" in text:
        return "제외", "스프 제외"

    if big in ["국 및 탕류", "찌개 및 전골류"]:
        return "제외", "국물류 제외"

    if big == "전·적 및 부침류":
        return "제외", "전/부침 제외"

    if has_any(text, ["햄", "베이컨", "치킨텐더", "시저"]):
        return "제외", "가공육/패스트샐러드 제외"

    if has_any(text, high_fat_keywords):
        return "제외", "고지방 육류 제외"

    if pd.notna(sugar) and sugar >= 10:
        return "제외", "당류 높음"

    if pd.notna(sodium) and sodium >= 800:
        return "제외", "나트륨 높음"

    if big in ["밥류", "죽 및 스프류"]:
        return "곡류군후보", "대분류"

    if has_any(rep, [
        "밥", "죽", "누룽지", "미음",
        "현미밥", "잡곡밥", "보리밥", "귀리밥", "율무밥",
        "콩나물밥", "무밥", "굴밥", "채소밥", "곤드레밥", "곤드레나물밥", "영양돌솥밥"
    ]):
        return "곡류군후보", "대표식품명"

    if big in ["구이류", "찜류", "조림류"]:
        if has_any(text, ["두부", "양념두부", "마파두부"]):
            if pd.notna(fat) and fat >= 15:
                return "제외", "지방 높아 제외"
            return "어육류군후보", "두부 음식"

        if pd.notna(fat) and fat >= 15:
            return "제외", "지방 높아 제외"
        if pd.notna(protein) and protein >= 8:
            return "어육류군후보", "대분류+단백질"

    if has_any(rep, protein_rep_keywords):
        if has_any(text, high_fat_keywords):
            return "제외", "고지방 육류 제외"
        if pd.notna(fat) and fat >= 15:
            return "제외", "지방 높아 제외"
        if pd.notna(protein) and protein >= 5:
            return "어육류군후보", "대표식품명"

    if big in ["나물·숙채류", "생채·무침류"]:
        if has_any(rep, vegetable_exclude_keywords) or has_any(text, vegetable_exclude_keywords):
            return "제외", "채소군 제외 키워드"
        if has_any(rep, vegetable_main_keywords) or has_any(text, vegetable_main_keywords):
            return "채소군후보", "대분류"

    if has_any(rep, vegetable_main_keywords):
        if has_any(rep, vegetable_exclude_keywords) or has_any(text, vegetable_exclude_keywords):
            return "제외", "채소군 제외 키워드"
        return "채소군후보", "대표식품명"

    if big == "볶음류":
        if has_any(text, ["두부", "양념두부", "마파두부"]):
            if pd.notna(fat) and fat >= 12:
                return "제외", "볶음류 지방 높음"
            if pd.notna(protein) and protein >= 5:
                return "어육류군후보", "두부 볶음류"

        if has_any(rep, vegetable_exclude_keywords) or has_any(text, vegetable_exclude_keywords):
            if pd.notna(protein) and protein >= 8 and pd.notna(fat) and fat < 12:
                return "어육류군후보", "볶음류"
            return "제외", "볶음류 제외"

        if has_any(rep, vegetable_main_keywords) or has_any(text, vegetable_main_keywords):
            if pd.notna(fat) and fat < 12:
                return "채소군후보", "볶음류"

        if pd.notna(fat) and fat >= 12:
            return "제외", "볶음류 지방 높음"

        if pd.notna(protein) and protein >= 8 and pd.notna(fat) and fat < 12:
            return "어육류군후보", "볶음류"

        if pd.notna(fiber) and fiber >= 2 and pd.notna(protein) and protein < 8:
            return "채소군후보", "볶음류"

        return "제외", "볶음류 제외"

    if big == "면 및 만두류":
        return "제외", "면/만두류 제외"

    return "제외", "식단용 후보 아님"

def score_food(row):
    bucket = row.get("food_bucket")
    rep = str(row.get("대표식품명", ""))
    text = join_text(row)

    kcal = row.get("에너지(kcal)")
    carb = row.get("탄수화물(g)")
    sugar = row.get("당류(g)")
    protein = row.get("단백질(g)")
    fiber = row.get("식이섬유(g)")
    sodium = row.get("나트륨(mg)")
    fat = row.get("지방(g)")

    if bucket == "제외":
        return -999

    score = 0

    if pd.notna(sugar) and sugar <= 5:
        score += 2
    if pd.notna(sodium) and sodium <= 500:
        score += 2
    if pd.notna(kcal) and 50 <= kcal <= 350:
        score += 1

    if bucket == "곡류군후보":
        if pd.notna(carb) and 20 <= carb <= 45:
            score += 3
        if has_any(text, ["현미", "잡곡", "귀리", "오트", "보리", "율무"]):
            score += 3

    elif bucket == "어육류군후보":
        if pd.notna(protein) and protein >= 10:
            score += 3
        if pd.notna(fat) and fat <= 10:
            score += 2
        elif pd.notna(fat) and fat <= 14:
            score += 1
        if pd.notna(fiber) and fiber >= 2:
            score += 1
        if "소불고기" in rep or "소불고기" in text:
            score -= 1

    elif bucket == "채소군후보":
        score += 3
        if pd.notna(fiber) and fiber >= 2:
            score += 3
        if pd.notna(protein) and protein <= 8:
            score += 1

    return score

bucket_result = df.apply(detect_bucket, axis=1, result_type="expand")
bucket_result.columns = ["food_bucket", "bucket_reason"]

df["food_bucket"] = bucket_result["food_bucket"]
df["bucket_reason"] = bucket_result["bucket_reason"]

usable_df = df[df["food_bucket"] != "제외"].copy()
usable_df["priority_score"] = usable_df.apply(score_food, axis=1)

usable_df = usable_df.sort_values(
    by=["food_bucket", "priority_score", "단백질(g)", "식이섬유(g)"],
    ascending=[True, False, False, False]
).copy()

usable_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("저장 완료:", out_path)
print("\nfood_bucket 분포")
print(usable_df["food_bucket"].value_counts())
print("\n최종 개수:", len(usable_df))

for bucket in ["곡류군후보", "어육류군후보", "채소군후보"]:
    print("\n" + "=" * 80)
    print(f"[{bucket}]")
    temp = usable_df[usable_df["food_bucket"] == bucket].head(20)
    if len(temp) > 0:
        print(temp[[
            "식품명", "대표식품명", "식품대분류명", "식품중분류명",
            "에너지(kcal)", "탄수화물(g)", "당류(g)", "단백질(g)",
            "지방(g)", "식이섬유(g)", "나트륨(mg)",
            "food_bucket", "bucket_reason", "priority_score"
        ]].to_string(index=False))