from pydantic import BaseModel, Field, field_validator


MEAL_TYPES = {"breakfast", "lunch", "dinner"}
SEX_VALUES = {"M", "F"}


class Meal(BaseModel):
    carbs: float = Field(..., ge=0)
    protein: float = Field(..., ge=0)
    fat: float = Field(..., ge=0)
    fiber: float = Field(..., ge=0)
    kcal: float = Field(..., gt=0)
    mealType: str

    @field_validator("mealType", mode="before")
    @classmethod
    def check_meal_type(cls, value):
        # 식사 유형 표기를 학습된 범주 형태로 통일
        value = str(value).strip().lower()
        if value not in MEAL_TYPES:
            raise ValueError("mealType must be one of: breakfast, lunch, dinner")
        return value

    @field_validator("fiber")
    @classmethod
    def check_fiber(cls, value, info):
        # 순탄수 계산이 음수가 되지 않도록 최소 관계만 유지
        carbs = info.data.get("carbs")
        if carbs is not None and value > carbs:
            raise ValueError("fiber cannot be greater than carbs")
        return value


class PredictRequest(BaseModel):
    baselineGlucose: float = Field(..., gt=0)
    sex: str
    meal: Meal

    @field_validator("sex", mode="before")
    @classmethod
    def check_sex(cls, value):
        # 성별 표기를 모델이 학습한 값으로 정규화
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
