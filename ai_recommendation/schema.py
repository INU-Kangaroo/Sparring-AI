from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

MealTime = Literal["아침", "점심", "저녁", "간식"]
Sex = Literal["남", "여"]


class RecommendationInput(BaseModel):
    blood_glucose: float
    sbp: float
    dbp: float
    height_cm: float
    weight_kg: float

    sex: Sex
    age_years: int

    history: Optional[List[str]] = []
    meal_time: MealTime
    top_n: Optional[int] = 10

    eaten_foods: Optional[List[str]] = []
    eaten_kcal: Optional[float] = 0
    eaten_carb_g: Optional[float] = 0
    eaten_protein_g: Optional[float] = 0
    eaten_fat_g: Optional[float] = 0
    eaten_sodium_mg: Optional[float] = 0

    # 사용자가 주면 최우선, 없으면 서버가 사용자 맞춤으로 자동 계산해서 remaining 계산에 사용
    target_kcal: Optional[float] = None
    target_carb_g: Optional[float] = None
    target_protein_g: Optional[float] = None
    target_fat_g: Optional[float] = None
    target_sodium_mg: Optional[float] = None


class RecommendationResult(BaseModel):
    goals: List[str]
    message: str = ""
    meal_recommendations: Dict[str, List[str]] = Field(default_factory=dict)
    reasons: Dict[str, List[str]] = Field(default_factory=dict)

    remaining: Dict[str, Optional[float]] = Field(default_factory=dict)

    # 사용자가 target_kcal을 안 주면 서버가 계산한 값
    computed_target_kcal: Optional[float] = None