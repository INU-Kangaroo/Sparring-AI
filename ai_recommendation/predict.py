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

# 빵은 100g 기준 데이터를 “1장(약 35g)” 기준으로 환산
BREAD_SLICE_G = 35

# 아침에서 제외할 “확실히 부적절”한 것들
BREAKFAST_BLOCK_KEYWORDS = [
    "피자", "햄버거", "라면", "떡볶이", "튀김", "치킨", "핫도그",
    "아이스크림", "케이크", "초콜릿", "쿠키", "과자", "도넛",
    "콜라", "사이다", "에너지드링크",
]

# ---------------------------
# ✅ 간식 정책
# ---------------------------
DEFAULT_SNACK_MAX_KCAL = 250.0   # 사용자가 snack_max_kcal 안 주면 기본 상한
SNACK_MIN_CAP_KCAL = 80.0       # 남은 칼로리 매우 적어도 최소 상한
SNACK_DYNAMIC_RATIO = 0.40      # 남은 칼로리의 40%까지만 간식 허용(식사 여지 남기기)

# 간식 후보 키워드 (케이크 포함 허용)
SNACK_KEYWORDS = [
    # 과일
    "과일", "사과", "바나나", "딸기", "블루베리", "포도", "키위", "귤", "오렌지", "자몽",
    # 빵/베이커리
    "빵", "식빵", "토스트", "바게트", "사워", "사워도우", "효모", "천연발효",
    "호밀", "통밀", "베이글", "크루아상", "머핀", "번", "브리오슈",
    # 바/간편
    "에너지바", "프로틴바", "단백질바", "시리얼바", "그래놀라", "오트", "오트바",
    # 디저트/과자 (허용)
    "초콜릿", "쿠키", "과자", "비스킷", "웨하스", "젤리", "사탕",
    "케이크", "티라미수", "도넛", "크림", "크림빵", "슈", "크로플", "와플",
    # 유제품/요거트
    "요거트", "요구르트", "그릭", "치즈", "우유",
    # 견과
    "견과", "아몬드", "호두", "캐슈", "피스타치오", "땅콩",
    # 가벼운 단백질 간식
    "연두부", "두부", "순두부", "달걀", "계란", "삶은",
]

# 간식에서 “식사대용/패스트푸드”는 제외(피자/라면 등)
SNACK_DISALLOW_KEYWORDS = [
    "피자", "햄버거", "라면", "떡볶이", "치킨", "핫도그",
    "김밥", "볶음밥", "덮밥", "국밥", "짜장", "짬뽕", "버거"
]

# “식사(메인요리) 느낌” 강한 키워드 (간식에서 배제)
MAIN_DISH_BLOCK_KEYWORDS = [
    "탕", "찌개", "찜", "구이", "볶음", "전골", "국", "곰탕", "매운탕",
    "족발", "수육", "스테이크", "갈비", "삼겹", "불고기", "닭찜",
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

            # 과일류에서 carb 누락 시 sugar로 보정
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


def is_main_dish_like(food: dict) -> bool:
    name = (food.get("name") or "")
    return any(k in name for k in MAIN_DISH_BLOCK_KEYWORDS)


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


def weighted_sample_topk(
    scored: List[Tuple[dict, float]],
    n: int,
    *,
    top_k: int = 60,
    seed: Optional[int] = None
) -> List[dict]:
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

    allow_keywords = ["요거트", "그릭", "견과", "샐러드", "채소", "토마토", "오이", "두부", "연두부", "달걀", "계란", "두유"]
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
        # 간식은 “혈당 급등 방지”를 더 우선
        if sugar <= 8:
            adj += 0.10
        if sugar >= 20:
            adj -= 0.25
        if carb <= 30:
            adj += 0.05
        if carb >= 55:
            adj -= 0.20
        if protein >= 8:
            adj += 0.08
        if fiber >= 3:
            adj += 0.06

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


def sodium_target_by_age_sex(age: int, sex: str) -> float:
    if age >= 65:
        return 1500.0
    return 1800.0


def protein_g_per_kg(age: int, goals: List[str]) -> float:
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
    fat_ratio = 0.30
    if "혈당관리" in goals:
        fat_ratio = 0.35
    return fat_ratio


def compute_personal_targets(input_data: dict, computed_target_kcal: float, goals: List[str]) -> dict:
    age = int(input_data.get("age_years") or 0)
    sex = input_data.get("sex")
    w = float(input_data.get("weight_kg") or 0)

    target_kcal = float(input_data["target_kcal"]) if input_data.get("target_kcal") is not None else float(computed_target_kcal)

    target_protein_g = w * protein_g_per_kg(age, goals)
    target_fat_g = (target_kcal * fat_ratio_by_goals(goals)) / 9.0

    protein_kcal = target_protein_g * 4.0
    fat_kcal = target_fat_g * 9.0
    remaining_kcal_for_carb = max(0.0, target_kcal - protein_kcal - fat_kcal)
    target_carb_g = remaining_kcal_for_carb / 4.0

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


# ✅✅✅ 여기부터 “혈당 완충 점수” 추가
def glycemic_buffer_score(food: dict, goals: List[str]) -> float:
    """
    혈당이 튀지 않게(또는 천천히 오르게) 만드는 구조 점수
    - 단백질/지방/식이섬유 가산
    - 당류/탄수비율 감점
    """
    protein = float(food.get("protein_effective", food.get("protein") or 0.0) or 0.0)
    fat = float(food.get("fat_effective", food.get("fat") or 0.0) or 0.0)
    fiber = float(food.get("fiber_effective", food.get("fiber") or 0.0) or 0.0)
    sugar = float(food.get("sugar_effective", food.get("sugar") or 0.0) or 0.0)
    carb_ratio = float(food.get("carb_kcal_ratio") or 0.0)

    score = 0.0
    score += protein * 0.4
    score += fat * 0.2
    score += fiber * 0.5
    score -= sugar * 0.6
    score -= carb_ratio * 5.0

    # 혈당관리일 때만 더 강하게 반영
    if "혈당관리" in goals:
        score *= 1.4

    return score
# ✅✅✅ 여기까지 추가


# ---------------------------
# ✅ 간식 후보 판정 + 칼로리 상한
# ---------------------------
def is_snack_like(food: dict, goals: List[str]) -> bool:
    name = (food.get("name") or "")
    big = (food.get("cat_big") or "")

    # 메인요리 느낌은 간식에서 제외
    if is_main_dish_like(food):
        return False

    # 음료는 기본 제외 (커피 튐 방지)
    # 단, 두유/프로틴 계열 + 저당이면 간식으로 허용
    if food.get("is_liquid"):
        sugar = float(food.get("sugar_effective", food.get("sugar") or 0.0) or 0.0)
        if any(k in name for k in ["프로틴", "단백질", "두유"]) and sugar <= 8:
            return True
        return False

    # 간식에서 확실히 제외할 것
    if any(k in name for k in SNACK_DISALLOW_KEYWORDS):
        return False

    # 과일/빵/유제품은 강하게 간식 후보
    if is_fruit(food) or is_bread(food):
        return True
    if "유제품" in big:
        return True

    # 그 외는 키워드로 판단
    return any(k in name for k in SNACK_KEYWORDS)


def compute_snack_cap_kcal(input_data: dict, remaining: Dict[str, Optional[float]]) -> float:
    user_cap = input_data.get("snack_max_kcal")
    base_cap = float(user_cap) if user_cap is not None else DEFAULT_SNACK_MAX_KCAL

    rk = remaining.get("kcal")
    if rk is None:
        return base_cap

    dynamic_cap = rk * SNACK_DYNAMIC_RATIO
    return max(SNACK_MIN_CAP_KCAL, min(base_cap, dynamic_cap))


def snack_health_gate(food: dict, goals: List[str], snack_cap: float, *, mode: str) -> bool:
    """
    mode:
      - "strict": 혈당관리면 당/탄수 기준도 적용
      - "loose": kcal만 지키고(간식 후보군만) 채우기
    """
    kcal = float(food.get("kcal_effective", food.get("kcal") or 0.0) or 0.0)
    if kcal <= 0 or kcal > snack_cap:
        return False

    if mode == "loose":
        return True

    if "혈당관리" in goals:
        sugar = food.get("sugar_effective", food.get("sugar"))
        carb = food.get("carb_effective", food.get("carb"))
        if sugar is None or carb is None:
            return False
        sugar = float(sugar or 0.0)
        carb = float(carb or 0.0)

        # “케이크 허용”을 위해 너무 빡세지 않게(원하면 조절)
        if sugar > 18:
            return False
        if carb > 45:
            return False

    return True


def pick_top_n(
    scored: List[Tuple[dict, float]],
    goals: List[str],
    n: int,
    meal_time: str,
    *,
    input_data: Optional[dict] = None,
    remaining: Optional[Dict[str, Optional[float]]] = None,
    breakfast_fruit_or_fallback_required: int = 8,
    breakfast_bread_required: int = 1,
) -> Tuple[List[str], Dict[str, List[str]]]:
    rec: List[str] = []
    reasons_out: Dict[str, List[str]] = {}

    used_names = set()
    used_types = set()

    def add_item(f: dict) -> bool:
        menu = f"{f['name']}"
        if menu in used_names:
            return False

        # ✅ 간식은 타입 중복 제한을 풀어야 10개 채움
        if meal_time == "간식":
            used_names.add(menu)
            rec.append(menu)
            reasons_out[menu] = build_reasons(f, goals)
            return True

        # 다른 끼니는 타입 다양성 유지
        t = food_type(f)
        if t in used_types:
            return False
        used_types.add(t)
        used_names.add(menu)
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

    if meal_time == "간식":
        _input = input_data or {}
        _remaining = remaining or {}
        snack_cap = compute_snack_cap_kcal(_input, _remaining)

        # 1) strict 먼저
        strict_pool: List[Tuple[dict, float]] = []
        for f, s in scored:
            if not is_snack_like(f, goals):
                continue
            if not snack_health_gate(f, goals, snack_cap, mode="strict"):
                continue
            strict_pool.append((f, s))

        for f in weighted_sample_topk(strict_pool, n, top_k=220, seed=base_seed + 7):
            if len(rec) >= n:
                break
            add_item(f)

        # 2) 부족하면 loose로 채우기 (kcal만 지킴)
        if len(rec) < n:
            loose_pool: List[Tuple[dict, float]] = []
            for f, s in scored:
                if not is_snack_like(f, goals):
                    continue
                if not snack_health_gate(f, goals, snack_cap, mode="loose"):
                    continue
                loose_pool.append((f, s))

            for f in weighted_sample_topk(loose_pool, n - len(rec), top_k=320, seed=base_seed + 8):
                if len(rec) >= n:
                    break
                add_item(f)

        return rec, reasons_out

    # 아침은 “과일/대체 8개 + 빵 1개” 우선
    if meal_time == "아침":
        filtered = [(f, s) for (f, s) in scored if not is_breakfast_blocked(f)]

        need_fruit = min(breakfast_fruit_or_fallback_required, n)
        remaining_after = max(0, n - need_fruit)
        need_bread = min(breakfast_bread_required, remaining_after)

        fruit_or_fallback = [(f, s) for (f, s) in filtered if (is_fruit(f) or is_breakfast_fruit_fallback(f))]
        bread_scored = [(f, s) for (f, s) in filtered if is_bread(f)]
        others = [(f, s) for (f, s) in filtered if not (is_fruit(f) or is_breakfast_fruit_fallback(f) or is_bread(f))]

        for f in weighted_sample_topk(fruit_or_fallback, need_fruit, top_k=120, seed=base_seed + 1):
            if len(rec) >= n:
                break
            add_item(f)

        for f in weighted_sample_topk(bread_scored, need_bread, top_k=120, seed=base_seed + 2):
            if len(rec) >= n:
                break
            add_item(f)

        remaining_cnt = n - len(rec)
        for f in weighted_sample_topk(others, remaining_cnt, top_k=120, seed=base_seed + 3):
            if len(rec) >= n:
                break
            add_item(f)

        if len(rec) < n:
            for f, _s in filtered:
                if len(rec) >= n:
                    break
                add_item(f)

        return rec, reasons_out

    # 점심/저녁은 일반 weighted 샘플
    chosen = weighted_sample_topk(scored, n, top_k=160, seed=base_seed + 9)
    for f in chosen:
        if len(rec) >= n:
            break
        add_item(f)

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

    foods = [f for f in foods if (f.get("cat_big") or "") not in EXCLUDE_BIG_CATS]

    eaten_set = set([str(x).strip() for x in (input_data.get("eaten_foods") or []) if str(x).strip()])
    if eaten_set:
        foods = [f for f in foods if f.get("name") not in eaten_set]

    # ✅ base score 계산
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

        # ✅✅✅ 혈당 덜 튀게(완충 구조) 점수 반영
        score += glycemic_buffer_score(f, goals)

        scored_base.append((f, score))

    top_n = int(input_data.get("top_n") or 10)
    meal_times = [input_data["meal_time"]] if input_data.get("meal_time") else ["아침", "점심", "저녁", "간식"]

    meal_recommendations: Dict[str, List[str]] = {}
    reasons: Dict[str, List[str]] = {}

    for mt in meal_times:
        # 끼니별 후보군: 기본 rule_based_filter 적용(혈당 튀는 고탄수는 여기서 많이 걸러짐)
        candidates_foods = rule_based_filter([f for f, _ in scored_base], goals)

        cand_set = set(id(x) for x in candidates_foods)
        candidates_scored = [(f, s) for (f, s) in scored_base if id(f) in cand_set]

        scored_mt: List[Tuple[dict, float]] = []
        for f, base_score in candidates_scored:
            s = base_score + meal_time_adjustment(f, mt, goals)
            scored_mt.append((f, s))

        scored_mt.sort(key=lambda x: x[1], reverse=True)

        rec, rmap = pick_top_n(
            scored_mt,
            goals,
            n=top_n,
            meal_time=mt,
            input_data=input_data,
            remaining=remaining,
        )
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