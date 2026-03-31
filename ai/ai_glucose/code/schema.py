from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class GlucoseHistoryItem(BaseModel):
    glucose_level: float = Field(..., alias="glucoseLevel", ge=20, le=600)
    measured_at: str = Field(..., alias="measuredAt")
    measurement_label: str = Field(default="기타", alias="measurementLabel")

    @field_validator("measurement_label")
    @classmethod
    def validate_measurement_label(cls, v: Optional[str]) -> str:
        allowed = {"공복", "식전", "식후", "기타"}
        if v is None:
            return "기타"
        if v not in allowed:
            raise ValueError(f"measurementLabel must be one of {sorted(allowed)}")
        return v

    model_config = {
        "populate_by_name": True
    }


class InsulinEventItem(BaseModel):
    event_type: Literal["basal", "bolus"] = Field(..., alias="eventType")
    dose: float = Field(..., ge=0, le=100, description="인슐린 투여량")
    used_at: str = Field(..., alias="usedAt")
    insulin_type: str = Field(..., alias="insulinType")

    @field_validator("insulin_type")
    @classmethod
    def validate_insulin_type(cls, v: Optional[str]) -> str:
        if v is None:
            raise ValueError("insulinType is required for each insulin event")
        s = str(v).strip()
        if not s:
            raise ValueError("insulinType is required for each insulin event")
        return s

    model_config = {
        "populate_by_name": True
    }


class PredictGlucoseRequest(BaseModel):
    timestamp: str = Field(..., description="예측 요청 시각 (ISO8601)")

    glucose_history: List[GlucoseHistoryItem] = Field(
        ...,
        alias="glucoseHistory",
        min_length=1,
        description="최근 혈당 기록 목록"
    )

    carb_intake: float = Field(
        ...,
        alias="carbIntake",
        ge=0,
        le=300,
        description="섭취 예정 또는 섭취한 탄수화물(g)"
    )

    meal_type: int = Field(
        0,
        alias="mealType",
        ge=0,
        le=4,
        description="0=unknown, 1=breakfast, 2=lunch, 3=dinner, 4=snack"
    )

    @field_validator("meal_type", mode="before")
    @classmethod
    def normalize_meal_type(cls, v):
        if isinstance(v, str):
            mapping = {
                "unknown": 0,
                "breakfast": 1,
                "lunch": 2,
                "dinner": 3,
                "snack": 4,
            }
            key = v.strip().lower()
            if key in mapping:
                return mapping[key]
        return v

    steps: int = Field(
        default=0,
        ge=0,
        le=100000,
        description="최근 60분 걸음 수"
    )

    intensity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="최근 활동 강도 (0.0~1.0)"
    )

    insulin_bolus: float = Field(
        default=0.0,
        alias="insulinBolus",
        ge=0,
        le=100,
        description="레거시 호환용 bolus 입력값"
    )

    insulin_type: Optional[str] = Field(
        default="Unknown",
        alias="insulinType",
        description="레거시 호환용 인슐린 유형"
    )

    insulin_events: List[InsulinEventItem] = Field(
        default_factory=list,
        alias="insulinEvents",
        description="인슐린 이벤트 목록"
    )
    insulin_basal: float = Field(default=0.0, alias="insulinBasal", ge=0, le=100)
    bolus_dose_60m: float = Field(default=0.0, alias="bolusDose60m", ge=0, le=100)
    bolus_dose_120m: float = Field(default=0.0, alias="bolusDose120m", ge=0, le=100)
    bolus_carb_input_30m: float = Field(default=0.0, alias="bolusCarbInput30m", ge=0, le=300)
    bolus_carb_input_60m: float = Field(default=0.0, alias="bolusCarbInput60m", ge=0, le=300)
    bolus_carb_input_120m: float = Field(default=0.0, alias="bolusCarbInput120m", ge=0, le=300)
    temp_basal_active: bool = Field(default=False, alias="tempBasalActive")
    temp_basal_value: float = Field(default=0.0, alias="tempBasalValue", ge=0, le=100)
    insulin_total_60m: float = Field(default=0.0, alias="insulinTotal60m", ge=0, le=200)
    insulin_total_120m: float = Field(default=0.0, alias="insulinTotal120m", ge=0, le=200)
    insulin_onboard_proxy: float = Field(default=0.0, alias="insulinOnboardProxy", ge=0, le=200)
    basal_bolus_ratio: float = Field(default=0.0, alias="basalBolusRatio", ge=0, le=10000)
    is_insulin_user: float = Field(default=1.0, alias="isInsulinUser", ge=0, le=1)

    prediction_offset_minutes: int = Field(
        default=120,
        alias="predictionOffsetMinutes",
        ge=10,
        le=240,
        description="대표 예측 시점(분)"
    )

    step_minutes: int = Field(
        default=10,
        alias="stepMinutes",
        ge=10,
        le=30,
        description="forecast 간격(분)"
    )

    horizon_minutes: int = Field(
        default=120,
        alias="horizonMinutes",
        ge=10,
        le=240,
        description="전체 예측 범위(분)"
    )

    debug: bool = Field(default=False)

    @field_validator("glucose_history")
    @classmethod
    def validate_glucose_history(cls, v: List[GlucoseHistoryItem]) -> List[GlucoseHistoryItem]:
        if len(v) < 1:
            raise ValueError("glucoseHistory must contain at least 1 item")
        return v
    @field_validator("insulin_type", mode="before")
    @classmethod
    def normalize_insulin_type(cls, v):
        if v is None:
            return "Unknown"
        s = str(v).strip()
        if not s:
            return "Unknown"
        return s

    @field_validator("insulin_events")
    @classmethod
    def validate_insulin_events(cls, v: List[InsulinEventItem]) -> List[InsulinEventItem]:
        return v

    @model_validator(mode="after")
    def validate_insulin_source(self):
        has_events = len(self.insulin_events) > 0
        has_direct = any([
            self.insulin_bolus > 0,
            self.insulin_basal > 0,
            self.bolus_dose_60m > 0,
            self.bolus_dose_120m > 0,
            self.temp_basal_active,
            self.temp_basal_value > 0,
            self.insulin_total_60m > 0,
            self.insulin_total_120m > 0,
            self.insulin_onboard_proxy > 0,
        ])
        has_identity = self.is_insulin_user > 0 or (self.insulin_type is not None and str(self.insulin_type).strip().lower() not in {"", "unknown", "none"})
        if not has_events and not has_direct and not has_identity:
            raise ValueError("Provide insulinEvents or direct insulin feature fields.")
        return self

    @field_validator("prediction_offset_minutes")
    @classmethod
    def validate_prediction_offset(cls, v: int) -> int:
        if v != 120:
            raise ValueError("predictionOffsetMinutes must be 120")
        return v

    @field_validator("step_minutes")
    @classmethod
    def validate_step_minutes(cls, v: int) -> int:
        if v != 10:
            raise ValueError("stepMinutes must be 10")
        return v

    @field_validator("horizon_minutes")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v != 120:
            raise ValueError("horizonMinutes must be 120")
        return v

    model_config = {
        "populate_by_name": True
    }


class ForecastPoint(BaseModel):
    time: str
    predicted_glucose: float = Field(..., alias="predictedGlucose")
    step: int
    offset_minutes: int = Field(..., alias="offsetMinutes")

    model_config = {
        "populate_by_name": True
    }


class PeakInfo(BaseModel):
    peak_glucose: float = Field(..., alias="peakGlucose")
    peak_time: str = Field(..., alias="peakTime")
    peak_offset_minutes: int = Field(..., alias="peakOffsetMinutes")

    model_config = {
        "populate_by_name": True
    }


class PredictGlucoseResponse(BaseModel):
    predicted_glucose: float = Field(..., alias="predictedGlucose")
    prediction_offset_minutes: int = Field(..., alias="predictionOffsetMinutes")
    predicted_time: str = Field(..., alias="predictedTime")

    forecast: List[ForecastPoint]
    milestones: Optional[Dict[str, ForecastPoint]] = None
    peak: Optional[PeakInfo] = None
    debug: Optional[Dict[str, Any]] = None

    model_config = {
        "populate_by_name": True
    }


GlucosePredictRequest = PredictGlucoseRequest
GlucosePredictResponse = PredictGlucoseResponse
