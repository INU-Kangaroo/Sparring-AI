import csv
import os
import random
from datetime import date
from typing import Dict, List, Tuple, Optional

from preprocess import (
    calculate_bmi,
    decide_goals,
    enrich_food_features,
    drop_unknown_nutrition,
    hard_filter_for_diabetes,
    rule_based_filter,
    compute_preferred_types,
    build_reasons,
    food_type,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOODS_CSV_PATH = os.path.join(BASE_DIR, "data", "foods.csv")

EXCLUDE_BIG_CATS = {"장류, 양념류"}
BREAD_SLICE_G = 35

BREAKFAST_BLOCK_KEYWORDS = [
    "피자", "햄버거", "라면", "떡볶이", "튀김", "치킨", "핫도그",
    "아이스크림", "케이크", "초콜릿", "쿠키", "과자", "도넛",
    "콜라", "사이다", "에너지드링크",
]


def safe_float(v):
    try:
        v = str(v).strip()
        return float(v) if v else None
    except Exception:
        return None


def load_foods() -> List[dict]:
    if not os.path.exists(FOODS_CSV_PATH):
        raise FileNotFoundError(FOODS_CSV_PATH)

    foods: List[dict] = []
    with open(FOODS_CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=",")
        for r in reader:
            base = (r.get("영양성분함량기준량") or "").strip()
            if not (base.startswith("100g") or base.startswith("100ml")):
                continue

            name = (r.get("식품명") or "").strip()
            if not name:
                continue

            cat_big = (r.get("식품대분류명") or "").strip()
            sugar = safe_float(r.get("당류(g)"))
            carb = safe_float(r.get("탄수화물(g)"))

            if cat_big == "과일류" and carb is None and sugar is not None:
                carb = sugar

            foods.append(
                {
                    "name": name,
                    "rep_name": (r.get("대표식품명") or "").strip(),
                    "cat_big": cat_big,
                    "cat_mid": (r.get("식품중분류명") or "").strip(),
                    "kcal": safe_float(r.get("에너지(kcal)")),
                    "carb": carb,
                    "sugar": sugar,
                    "protein": safe_float(r.get("단백질(g)")),
                    "fat": safe_float(r.get("지방(g)")),
                    "fiber": safe_float(r.get("식이섬유(g)")),
                    "sodium": safe_float(r.get("나트륨(mg)")),
                    "is_liquid": base.startswith("100ml"),
                }
            )
    return foods


def is_breakfast_blocked(food: dict) -> bool:
    name = food.get("name") or ""
    return any(k in name for k in BREAKFAST_BLOCK_KEYWORDS)


def is_fruit(food: dict) -> bool:
    name = (food.get("name") or "").strip()
    rep = (food.get("rep_name") or "").strip()
    big = (food.get("cat_big") or "").strip()
    mid = (food.get("cat_mid") or "").strip()

    return (
        big == "과일류"
        or name.startswith("과일_")
        or ("과일" in name)
        or ("과일" in rep)
        or ("과일" in mid)
    )


def is_bread(food: dict) -> bool:
    name = (food.get("name") or "")
    big = (food.get("cat_big") or "")

    bread_keywords = [
        "빵", "식빵", "토스트", "바게트", "사워", "사워도우", "효모", "천연발효",
        "호밀", "통밀", "베이글", "크루아상", "번", "도넛", "브리오슈", "머핀"
    ]
    if any(k in name for k in bread_keywords):
        return True

    big_keywords = ["제과", "빵", "베이커리"]
    if any(k in big for k in big_keywords):
        return True

    return False


def apply_effective_nutrition(foods: List[dict]) -> List[dict]:
    for f in foods:
        factor = (BREAD_SLICE_G / 100.0) if is_bread(f) else 1.0

        for key in ["kcal", "carb", "protein", "fat", "sodium", "sugar", "fiber"]:
            v = f.get(key)
            if v is None:
                continue
            f[f"{key}_effective"] = v * factor

        if f.get("carb_effective") is None and f.get("carb") is not None:
            f["carb_effective"] = f["carb"] * factor

    return foods


def weighted_sample_topk(scored: List[Tuple[dict, float]], n: int, *, top_k: int = 60, seed: Optional[int] = None) -> List[dict]:
    pool = scored[: max(n, top_k)]
    if not pool:
        return []

    rng = random.Random(seed)
    scores = [s for _, s in pool]
    min_s = min(scores)
    weights = [(s - min_s) + 1e-6 for s in scores]

    chosen: List[dict] = []
    items = list(pool)
    wts = list(weights)

    for _ in range(min(n, len(items))):
        total = sum(wts)
        r = rng.random() * total
        acc = 0.0
        idx = 0
        for i, w in enumerate(wts):
            acc += w
            if acc >= r:
                idx = i
                break
        chosen.append(items.pop(idx)[0])
        wts.pop(idx)

    return chosen


def is_breakfast_fruit_fallback(food: dict) -> bool:
    if food.get("is_liquid"):
        return False
    if is_breakfast_blocked(food):
        return False

    name = food.get("name") or ""
    big = food.get("cat_big") or ""

    carb = float(food.get("carb_effective", food.get("carb") or 0.0) or 0.0)
    sugar = float(food.get("sugar_effective", food.get("sugar") or 0.0) or 0.0)

    if sugar > 8:
        return False
    if carb > 30:
        return False

    if "아이스크림" in name:
        return False

    allow_keywords = ["요거트", "그릭", "견과", "샐러드", "채소", "토마토", "오이", "두부", "달걀", "계란", "두유"]
    if any(k in name for k in allow_keywords):
        return True

    if big == "유제품류 및 빙과류" and any(k in name for k in ["요거트", "그릭"]):
        return True

    return False


def meal_time_adjustment(food: dict, meal_time: str, goals: List[str]) -> float:
    carb = float(food.get("carb_effective", food.get("carb") or 0.0) or 0.0)
    sugar = float(food.get("sugar_effective", food.get("sugar") or 0.0) or 0.0)
    protein = float(food.get("protein_effective", food.get("protein") or 0.0) or 0.0)
    fiber = float(food.get("fiber_effective", food.get("fiber") or 0.0) or 0.0)
    sodium = float(food.get("sodium_effective", food.get("sodium") or 0.0) or 0.0)

    adj = 0.0

    if meal_time == "아침":
        if is_breakfast_blocked(food):
            adj -= 2.0

        if sugar <= 8:
            adj += 0.20
        if sugar >= 18:
            adj -= 0.25
        if carb <= 30:
            adj += 0.15
        if carb >= 55:
            adj -= 0.25

        if protein >= 8:
            adj += 0.10
        if fiber >= 2:
            adj += 0.08

        if is_fruit(food):
            adj += 0.18
            if sugar >= 15:
                adj -= 0.15

        if is_bread(food):
            adj += 0.08

    elif meal_time == "점심":
        if sugar <= 10:
            adj += 0.12
        if sugar >= 22:
            adj -= 0.18
        if carb <= 40:
            adj += 0.10
        if carb >= 65:
            adj -= 0.18
        if protein >= 12:
            adj += 0.10
        if fiber >= 3:
            adj += 0.08

    elif meal_time == "저녁":
        if sugar <= 8:
            adj += 0.15
        if sugar >= 18:
            adj -= 0.25
        if carb <= 35:
            adj += 0.12
        if carb >= 55:
            adj -= 0.25
        if sodium <= 500:
            adj += 0.06
        if sodium >= 900:
            adj -= 0.10
        if protein >= 12:
            adj += 0.08
        if fiber >= 3:
            adj += 0.08

    else:  # 간식
        if sugar <= 8:
            adj += 0.20
        if sugar >= 15:
            adj -= 0.30
        if carb <= 30:
            adj += 0.06
        if carb >= 45:
            adj -= 0.15
        if protein >= 8:
            adj += 0.06
        if fiber >= 3:
            adj += 0.06

        if is_fruit(food) or (food.get("cat_big") == "유제품류 및 빙과류"):
            adj += 0.10

    if meal_time in ("아침", "저녁") and sodium >= 900:
        adj -= 0.06

    if "혈당관리" in goals:
        adj *= 1.15

    return adj


def personalize_message(goals: List[str]) -> str:
    if "혈당관리" in goals and "혈압관리" in goals:
        return "맞춤 추천 (혈당·혈압 함께 관리해요)"
    if "혈당관리" in goals:
        return "맞춤 추천 (혈당 관리를 도와드려요)"
    if "혈압관리" in goals:
        return "맞춤 추천 (저염 식단으로 도와드려요)"
    if "감량" in goals:
        return "맞춤 추천 (가벼운 식사로 도와드려요)"
    if "증량" in goals:
        return "맞춤 추천 (충분히 먹을 수 있게 도와드려요)"
    return "맞춤 추천"


def age_based_default_kcal(sex: str, age_years: int) -> float:
    if age_years < 13:
        return 1800.0 if sex == "남" else 1600.0
    if 13 <= age_years <= 18:
        return 2800.0 if sex == "남" else 2200.0
    return 2500.0 if sex == "남" else 2000.0


def estimate_target_kcal(input_data: dict, goals: List[str]) -> float:
    if input_data.get("target_kcal") is not None:
        return float(input_data["target_kcal"])

    sex = input_data.get("sex")
    age = int(input_data.get("age_years") or 0)
    w = float(input_data.get("weight_kg") or 0)
    h = float(input_data.get("height_cm") or 0)

    ACT = 1.375

    tdee = None
    try:
        if sex in ("남", "여") and age > 0 and w > 0 and h > 0:
            if sex == "남":
                bmr = 10 * w + 6.25 * h - 5 * age + 5
            else:
                bmr = 10 * w + 6.25 * h - 5 * age - 161
            tdee = bmr * ACT
    except Exception:
        tdee = None

    if tdee is None:
        tdee = age_based_default_kcal(sex, age)

    if "감량" in goals:
        tdee -= 300.0
    elif "증량" in goals:
        tdee += 300.0

    if sex == "남":
        tdee = max(tdee, 1500.0)
    else:
        tdee = max(tdee, 1200.0)

    return round(tdee, 0)


# -----------------------------
# ✅ 개인 맞춤 target_* 자동 계산
# -----------------------------
def sodium_target_by_age_sex(age: int, sex: str) -> float:
    # ✅ 나트륨 목표치 고정 (mg/day)
    if age >= 65:
        return 1500.0
    return 1800.0


def protein_g_per_kg(age: int, goals: List[str]) -> float:
    """
    개인별 단백질 g/kg.
    - 청소년: 성장 고려
    - 성인: 기본 0.9
    - 감량: 근손실 방지/포만감 위해 상향
    - 증량: 상향
    """
    if age <= 18:
        base = 1.0
    else:
        base = 0.9

    if "감량" in goals:
        base = max(base, 1.2)
    if "증량" in goals:
        base = max(base, 1.1)

    return base


def fat_ratio_by_goals(goals: List[str]) -> float:
    """
    지방 비율(칼로리 비중).
    - 혈당관리면 탄수를 과다하게 잡지 않도록 지방 비율을 살짝 올림
    """
    fat_ratio = 0.30
    if "혈당관리" in goals:
        fat_ratio = 0.35
    return fat_ratio


def compute_personal_targets(input_data: dict, computed_target_kcal: float, goals: List[str]) -> dict:
    """
    target_*가 None일 때만 개인 맞춤 기본값을 채워준다.
    - kcal: computed_target_kcal
    - protein_g: 체중(kg) * g/kg
    - fat_g: kcal * fat_ratio / 9
    - carb_g: (kcal - protein_kcal - fat_kcal) / 4
    - sodium_mg: 연령대 기본 목표치
    """
    age = int(input_data.get("age_years") or 0)
    sex = input_data.get("sex")
    w = float(input_data.get("weight_kg") or 0)

    # kcal
    target_kcal = float(input_data["target_kcal"]) if input_data.get("target_kcal") is not None else float(computed_target_kcal)

    # protein
    p_gkg = protein_g_per_kg(age, goals)
    target_protein_g = w * p_gkg

    # fat
    fat_ratio = fat_ratio_by_goals(goals)
    target_fat_g = (target_kcal * fat_ratio) / 9.0

    # carb = remaining kcal
    protein_kcal = target_protein_g * 4.0
    fat_kcal = target_fat_g * 9.0
    remaining_kcal_for_carb = max(0.0, target_kcal - protein_kcal - fat_kcal)
    target_carb_g = remaining_kcal_for_carb / 4.0

    # sodium
    target_sodium_mg = sodium_target_by_age_sex(age, sex)

    out = {}
    if input_data.get("target_kcal") is None:
        out["target_kcal"] = round(target_kcal, 0)
    if input_data.get("target_protein_g") is None:
        out["target_protein_g"] = round(target_protein_g, 1)
    if input_data.get("target_fat_g") is None:
        out["target_fat_g"] = round(target_fat_g, 1)
    if input_data.get("target_carb_g") is None:
        out["target_carb_g"] = round(target_carb_g, 1)
    if input_data.get("target_sodium_mg") is None:
        out["target_sodium_mg"] = round(target_sodium_mg, 0)

    return out


def calc_remaining(input_data: dict, computed_target_kcal: float, goals: List[str]) -> Dict[str, Optional[float]]:
    """
    ✅ 사용자가 target_*를 안 주면 개인 맞춤으로 자동 생성해서 remaining 계산
    """
    filled_targets = compute_personal_targets(input_data, computed_target_kcal, goals)
    merged = dict(input_data)
    merged.update(filled_targets)

    def rem(target, eaten):
        if target is None:
            return None
        return max(0.0, float(target) - float(eaten or 0.0))

    return {
        "kcal": rem(merged.get("target_kcal"), merged.get("eaten_kcal")),
        "carb_g": rem(merged.get("target_carb_g"), merged.get("eaten_carb_g")),
        "protein_g": rem(merged.get("target_protein_g"), merged.get("eaten_protein_g")),
        "fat_g": rem(merged.get("target_fat_g"), merged.get("eaten_fat_g")),
        "sodium_mg": rem(merged.get("target_sodium_mg"), merged.get("eaten_sodium_mg")),
    }


def remaining_based_adjustment(food: dict, remaining: Dict[str, Optional[float]]) -> float:
    adj = 0.0

    kcal = float(food.get("kcal_effective", food.get("kcal") or 0.0) or 0.0)
    carb = float(food.get("carb_effective", food.get("carb") or 0.0) or 0.0)
    protein = float(food.get("protein_effective", food.get("protein") or 0.0) or 0.0)
    sodium = float(food.get("sodium_effective", food.get("sodium") or 0.0) or 0.0)

    rk = remaining.get("kcal")
    if rk is not None and rk > 0:
        over = max(0.0, kcal - rk)
        adj -= (over / rk) * 1.2
        if rk < 300 and kcal > 250:
            adj -= 0.8

    rc = remaining.get("carb_g")
    if rc is not None and rc > 0:
        over = max(0.0, carb - rc)
        adj -= (over / rc) * 1.4

    rs = remaining.get("sodium_mg")
    if rs is not None and rs > 0:
        over = max(0.0, sodium - rs)
        adj -= (over / rs) * 0.8

    rp = remaining.get("protein_g")
    if rp is not None and rp > 0:
        fill = min(protein, rp)
        adj += (fill / max(rp, 1.0)) * 0.9

    return adj


def pick_top_n(
    scored: List[Tuple[dict, float]],
    goals: List[str],
    n: int,
    meal_time: str,
    breakfast_fruit_or_fallback_required: int = 8,
    breakfast_bread_required: int = 1,
) -> Tuple[List[str], Dict[str, List[str]]]:
    rec: List[str] = []
    reasons_out: Dict[str, List[str]] = {}
    used_types = set()

    def add_item(f: dict) -> bool:
        t = food_type(f)
        if t in used_types:
            return False
        used_types.add(t)
        menu = f"{f['name']} (저염·무염 조리 가정)"
        rec.append(menu)
        reasons_out[menu] = build_reasons(f, goals)
        return True

    # -------------------------
    # ✅ seed 선택(원하는 것 1개만 남기기)
    # (A) 하루 기준 고정
    # base_seed = int(date.today().strftime("%Y%m%d"))

    # (B) 누를 때마다 변경
    base_seed = random.randint(1, 1_000_000_000)
    # -------------------------

    if meal_time == "아침":
        filtered = [(f, s) for (f, s) in scored if not is_breakfast_blocked(f)]

        need_fruit = min(breakfast_fruit_or_fallback_required, n)
        remaining_after = max(0, n - need_fruit)
        need_bread = min(breakfast_bread_required, remaining_after)

        fruit_or_fallback = [(f, s) for (f, s) in filtered if (is_fruit(f) or is_breakfast_fruit_fallback(f))]
        bread_scored = [(f, s) for (f, s) in filtered if is_bread(f)]
        others = [(f, s) for (f, s) in filtered if not (is_fruit(f) or is_breakfast_fruit_fallback(f) or is_bread(f))]

        for f in weighted_sample_topk(fruit_or_fallback, need_fruit, top_k=80, seed=base_seed + 1):
            if len(rec) >= n:
                break
            add_item(f)

        for f in weighted_sample_topk(bread_scored, need_bread, top_k=80, seed=base_seed + 2):
            if len(rec) >= n:
                break
            add_item(f)

        remaining_cnt = n - len(rec)
        for f in weighted_sample_topk(others, remaining_cnt, top_k=80, seed=base_seed + 3):
            if len(rec) >= n:
                break
            add_item(f)

        if len(rec) < n:
            for f, _s in filtered:
                if len(rec) >= n:
                    break
                add_item(f)

        return rec, reasons_out

    chosen = weighted_sample_topk(scored, n, top_k=80, seed=base_seed + 9)
    for f in chosen:
        add_item(f)
        if len(rec) >= n:
            break

    if len(rec) < n:
        for f, _s in scored:
            if len(rec) >= n:
                break
            add_item(f)

    return rec, reasons_out


def recommend(input_data: dict) -> dict:
    foods = load_foods()

    bmi = calculate_bmi(input_data["height_cm"], input_data["weight_kg"])
    user = {
        "blood_glucose": input_data["blood_glucose"],
        "sbp": input_data["sbp"],
        "dbp": input_data["dbp"],
        "bmi": bmi,
    }
    goals = decide_goals(user)
    preferred_types = compute_preferred_types(input_data.get("history", []))

    computed_target_kcal = estimate_target_kcal(input_data, goals)
    remaining = calc_remaining(input_data, computed_target_kcal, goals)

    foods = [enrich_food_features(f) for f in foods]
    foods = apply_effective_nutrition(foods)

    foods = drop_unknown_nutrition(foods, goals)

    if "혈당관리" in goals:
        foods = hard_filter_for_diabetes(foods)

    foods = rule_based_filter(foods, goals)
    foods = [f for f in foods if (f.get("cat_big") or "") not in EXCLUDE_BIG_CATS]

    eaten_set = set([str(x).strip() for x in (input_data.get("eaten_foods") or []) if str(x).strip()])
    if eaten_set:
        foods = [f for f in foods if f.get("name") not in eaten_set]

    scored_base: List[Tuple[dict, float]] = []
    for f in foods:
        score = f["tag_score"]

        sugar = float(f.get("sugar_effective", f.get("sugar") or 0.0) or 0.0)
        sodium = float(f.get("sodium_effective", f.get("sodium") or 0.0) or 0.0)
        kcal = float(f.get("kcal_effective", f.get("kcal") or 0.0) or 0.0)

        if "혈당관리" in goals:
            score -= f.get("carb_kcal_ratio", 0.0) * 6.0
            score -= sugar * 0.05

        if "혈압관리" in goals:
            score -= sodium * 0.0012

        if "감량" in goals:
            score -= kcal * 0.001

        if any(x in f.get("name", "") for x in ["생선", "두부", "콩"]):
            score += 0.5

        name_type = f["name"].split("_")[0] if "_" in f["name"] else f["name"]
        if name_type in preferred_types:
            score += preferred_types[name_type]

        score += remaining_based_adjustment(f, remaining)

        scored_base.append((f, score))

    top_n = int(input_data.get("top_n") or 10)
    meal_times = [input_data["meal_time"]] if input_data.get("meal_time") else ["아침", "점심", "저녁", "간식"]

    meal_recommendations: Dict[str, List[str]] = {}
    reasons: Dict[str, List[str]] = {}

    for mt in meal_times:
        scored_mt: List[Tuple[dict, float]] = []
        for f, base_score in scored_base:
            s = base_score + meal_time_adjustment(f, mt, goals)
            scored_mt.append((f, s))

        scored_mt.sort(key=lambda x: x[1], reverse=True)

        rec, rmap = pick_top_n(scored_mt, goals, n=top_n, meal_time=mt)
        meal_recommendations[mt] = rec
        reasons.update(rmap)

    return {
        "goals": goals,
        "message": personalize_message(goals),
        "meal_recommendations": meal_recommendations,
        "reasons": reasons,
        "remaining": remaining,
        "computed_target_kcal": computed_target_kcal,
    }