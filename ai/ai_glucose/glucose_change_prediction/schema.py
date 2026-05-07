from pydantic import BaseModel, Field, field_validator
from typing import Literal, List


class MealInput(BaseModel):
    carbs: float = Field(..., ge=0, le=1000)
    protein: float = Field(..., ge=0, le=1000)
    fat: float = Field(..., ge=0, le=1000)
    fiber: float = Field(..., ge=0, le=1000)
    kcal: float = Field(..., gt=0, le=5000)
    mealType: Literal["breakfast", "lunch", "dinner"]

    @field_validator("mealType", mode="before")
    @classmethod
    def normalize_meal_type(cls, v):
        s = str(v).strip().lower()
        allowed = {"breakfast", "lunch", "dinner"}
        if s not in allowed:
            raise ValueError("mealType must be one of: breakfast, lunch, dinner")
        return s

    @field_validator("fiber")
    @classmethod
    def validate_fiber_vs_carbs(cls, v, info):
        carbs = info.data.get("carbs")
        if carbs is not None and v > carbs:
            raise ValueError("fiber cannot be greater than carbs")
        return v


class PredictMealResponseRequest(BaseModel):
    baselineGlucose: float = Field(..., ge=40, le=400)
    sex: Literal["M", "F"]
    meal: MealInput

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, v):
        s = str(v).strip().upper()
        if s not in {"M", "F"}:
            raise ValueError("sex must be 'M' or 'F'")
        return s


class CurvePoint(BaseModel):
    minute: int
    delta: float


class PredictMealResponseResponse(BaseModel):
    delta30: float
    delta60: float
    peakDelta: float
    peakMinute: int
    curve: List[CurvePoint]