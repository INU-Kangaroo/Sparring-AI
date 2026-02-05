from pydantic import BaseModel
from typing import List

class GlucoseInput(BaseModel):
    # 최근 혈당 기록 (최소 3개 이상)
    glucose_history: List[float]

    # 식사 / 운동
    carb_intake: float
    meal_type: int
    steps: int
    intensity: float

    # 혈압 (실제 측정값)
    systolic_bp: int
    diastolic_bp: int

    # 개인 정보
    age: int
    sex: int
    weight: float

    # 시간 (ISO 문자열)
    timestamp: str

    # 기타 요인
    alcohol: int
    medication: int
    caffeine: int
