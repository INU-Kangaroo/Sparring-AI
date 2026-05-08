from pydantic import BaseModel, Field, field_validator


MEAL_TYPES = {"breakfast", "lunch", "dinner"}
SEX_VALUES = {"M", "F"}


class Meal(BaseModel):
    carbs: float = Field(..., ge=0, le=1000)
    protein: float = Field(..., ge=0, le=1000)
    fat: float = Field(..., ge=0, le=1000)
    fiber: float = Field(..., ge=0, le=1000)
    kcal: float = Field(..., gt=0, le=5000)
    mealType: str

    @field_validator("mealType", mode="before")
    @classmethod
    def check_meal_type(cls, value):
        value = str(value).strip().lower()
        if value not in MEAL_TYPES:
            raise ValueError("mealType must be one of: breakfast, lunch, dinner")
        return value

    @field_validator("fiber")
    @classmethod
    def check_fiber(cls, value, info):
        carbs = info.data.get("carbs")
        if carbs is not None and value > carbs:
            raise ValueError("fiber cannot be greater than carbs")
        return value


class PredictRequest(BaseModel):
    baselineGlucose: float = Field(..., ge=40, le=400)
    sex: str
    meal: Meal

    @field_validator("sex", mode="before")
    @classmethod
    def check_sex(cls, value):
        value = str(value).strip().upper()
        if value not in SEX_VALUES:
            raise ValueError("sex must be 'M' or 'F'")
        return value


class CurvePoint(BaseModel):
    minute: int
    delta: float


class PredictResponse(BaseModel):
    peakDelta: float
    peakMinute: int
    curve: list[CurvePoint]
