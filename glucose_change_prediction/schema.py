from typing import Literal

from pydantic import BaseModel, Field, field_validator


ALLOWED_MEAL_TYPES = {"breakfast", "lunch", "dinner"}
ALLOWED_SEX = {"M", "F"}


class Meal(BaseModel):
    carbs: float = Field(..., ge=0, le=1000)
    protein: float = Field(..., ge=0, le=1000)
    fat: float = Field(..., ge=0, le=1000)
    fiber: float = Field(..., ge=0, le=1000)
    kcal: float = Field(..., gt=0, le=5000)
    mealType: Literal["breakfast", "lunch", "dinner"]

    @field_validator("mealType", mode="before")
    @classmethod
    def normalize_meal_type(cls, value):
        meal_type = str(value).strip().lower()
        if meal_type not in ALLOWED_MEAL_TYPES:
            raise ValueError("mealType must be one of: breakfast, lunch, dinner")
        return meal_type

    @field_validator("fiber")
    @classmethod
    def validate_fiber(cls, value, info):
        carbs = info.data.get("carbs")
        if carbs is not None and value > carbs:
            raise ValueError("fiber cannot be greater than carbs")
        return value


class PredictRequest(BaseModel):
    baselineGlucose: float = Field(..., ge=40, le=400)
    sex: Literal["M", "F"]
    meal: Meal

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, value):
        sex = str(value).strip().upper()
        if sex not in ALLOWED_SEX:
            raise ValueError("sex must be 'M' or 'F'")
        return sex


class CurvePoint(BaseModel):
    minute: int
    delta: float


class PredictResponse(BaseModel):
    peakDelta: float
    peakMinute: int
    curve: list[CurvePoint]
