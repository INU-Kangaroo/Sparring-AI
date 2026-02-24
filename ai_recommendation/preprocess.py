from collections import Counter
from typing import Dict, List


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    h = height_cm / 100.0
    return round(weight_kg / (h * h), 2)


def decide_goals(user: dict) -> List[str]:
    goals = []
    if user["blood_glucose"] >= 140:
        goals.append("혈당관리")
    if user["sbp"] >= 140 or user["dbp"] >= 90:
        goals.append("혈압관리")
    if user["bmi"] >= 25:
        goals.append("감량")
    elif user["bmi"] < 18.5:
        goals.append("증량")
    if not goals:
        goals.append("유지")
    return goals


def enrich_food_features(food: dict) -> dict:
    kcal = food.get("kcal") or 0.0
    carb_g = food.get("carb") or 0.0
    food["carb_kcal_ratio"] = ((carb_g * 4.0) / kcal) if kcal > 0 else 0.0

    tag_score = 0.0
    if food.get("carb") is not None and food["carb"] < 30:
        tag_score += 0.4
    if food.get("sodium") is not None and food["sodium"] < 400:
        tag_score += 0.3
    if food.get("protein") is not None and food["protein"] > 15:
        tag_score += 0.3

    food["tag_score"] = tag_score
    return food


def drop_unknown_nutrition(foods: List[dict], goals: List[str]) -> List[dict]:
    out = []
    for f in foods:
        if f.get("kcal") is None:
            continue
        if "혈당관리" in goals and (f.get("carb") is None or f.get("sugar") is None):
            continue
        if "혈압관리" in goals and f.get("sodium") is None:
            continue
        out.append(f)
    return out


def hard_filter_for_diabetes(foods: List[dict]) -> List[dict]:
    return [f for f in foods if not f.get("is_liquid")]


def rule_based_filter(foods: List[dict], goals: List[str]) -> List[dict]:
    out = []
    for f in foods:
        carb_raw = f.get("carb") if f.get("carb") is not None else 0.0
        carb_eff = f.get("carb_effective", carb_raw)
        sodium = f.get("sodium") if f.get("sodium") is not None else 0.0
        kcal = f.get("kcal") if f.get("kcal") is not None else 0.0

        if any(x in f.get("name", "") for x in ["간", "곱창", "순대", "햄", "소시지"]):
            continue

        if "혈당관리" in goals and carb_eff > 30:
            continue
        if "혈압관리" in goals and sodium > 800:
            continue
        if "감량" in goals and kcal > 500:
            continue

        out.append(f)

    return out


def compute_preferred_types(history: List[str]) -> Dict[str, float]:
    if not history:
        return {}
    counts = Counter([h.split("_")[0] for h in history])
    max_count = max(counts.values())
    return {k: 0.5 * (v / max_count) for k, v in counts.items()}


def food_type(food: dict) -> str:
    rep = (food.get("rep_name") or "").strip()
    if rep:
        return rep
    name = food.get("name", "")
    return name.split("_")[0] if "_" in name else name


def build_reasons(food: dict, goals: List[str]) -> List[str]:
    r = []
    carb = food.get("carb_effective", food.get("carb"))
    sugar = food.get("sugar_effective", food.get("sugar"))
    sodium = food.get("sodium_effective", food.get("sodium"))
    protein = food.get("protein_effective", food.get("protein"))
    fiber = food.get("fiber_effective", food.get("fiber"))

    if "혈당관리" in goals:
        if carb is not None and carb <= 25:
            r.append("탄수화물이 낮아 혈당 관리에 유리해요")
        if sugar is not None and sugar <= 8:
            r.append("당류가 낮아 혈당 상승 부담이 적어요")

    if "혈압관리" in goals:
        if sodium is not None and sodium <= 400:
            r.append("나트륨이 낮아 저염 식단에 좋아요")

    if protein is not None and protein >= 12:
        r.append("단백질이 있어 포만감에 도움이 돼요")
    if fiber is not None and fiber >= 3:
        r.append("식이섬유가 있어 혈당 안정에 도움될 수 있어요")

    if any(x in food.get("name", "") for x in ["생선", "두부", "콩"]):
        r.append("두부/콩/생선 등 건강한 단백질 식품 계열이에요")

    if not r:
        r.append("현재 목표에 맞춰 무난하게 선택하기 좋아요")

    return r