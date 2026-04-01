import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

raw_path = PROCESSED_DIR / "clean_raw_materials.csv"
out_path = PROCESSED_DIR / "food_group_map.csv"

df = pd.read_csv(raw_path)

def map_exchange_group(row):
    big = str(row.get("식품대분류명", "")).strip()
    mid = str(row.get("식품중분류명", "")).strip()
    sub = str(row.get("식품소분류명", "")).strip()
    name = str(row.get("식품명", "")).strip()

    text = " ".join([big, mid, sub, name])

    # 곡류군
    if any(k in text for k in [
        "곡류", "쌀", "보리", "현미", "밀", "귀리", "옥수수",
        "감자", "고구마", "전분", "떡", "국수", "면"
    ]):
        return "곡류군"

    # 과일군
    if any(k in text for k in [
        "과일", "사과", "배", "바나나", "귤", "오렌지", "포도",
        "딸기", "복숭아", "감", "키위", "망고", "파인애플"
    ]):
        return "과일군"

    # 우유군
    if any(k in text for k in [
        "우유", "유제품", "요구르트", "요거트", "치즈", "분유"
    ]):
        return "우유군"

    # 지방군
    if any(k in text for k in [
        "유지", "기름", "참기름", "들기름", "올리브유", "버터",
        "마요네즈", "견과", "아몬드", "호두", "잣", "땅콩", "종실"
    ]):
        return "지방군"

    # 어육류군
    if any(k in text for k in [
        "육류", "닭", "소고기", "돼지고기", "쇠고기", "계란", "난류",
        "두류", "콩", "두부", "생선", "어패류", "수산물", "오징어",
        "문어", "새우", "게", "조개", "참치", "고등어", "연어", "멸치"
    ]):
        return "어육류군"

    # 채소군
    if any(k in text for k in [
        "채소", "버섯", "해조", "나물", "상추", "배추", "양배추",
        "브로콜리", "오이", "당근", "시금치", "깻잎", "토마토", "파", "양파", "마늘"
    ]):
        return "채소군"

    return "기타"

df["exchange_group"] = df.apply(map_exchange_group, axis=1)

# 필요한 컬럼만 일부 확인용
print(df["exchange_group"].value_counts(dropna=False))

df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"저장 완료: {out_path}")