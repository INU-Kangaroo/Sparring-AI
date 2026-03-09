from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GlucosePredictRequest(BaseModel):
    timestamp: str = Field(..., description="ISO8601 e.g. 2026-01-27T03:00:00.000Z")

    glucose_history: List[float] = Field(..., min_items=1)

    carb_intake: float = Field(..., ge=0)
    meal_type: int = Field(..., description="1 breakfast, 2 lunch, 3 dinner, 4 snack")

    steps: float = Field(0, ge=0)
    intensity: float = Field(0, ge=0, le=1)

    age: float = 0
    sex: int = 0
    weight: float = 0

    alcohol: int = 0
    medication: int = 0
    caffeine: int = 0

    is_insulin_user: int = 0
    insulin_bolus: float = 0
    insulin_basal: float = 0
    carb_ratio: float = 0
    insulin_sensitivity: float = 0

    prediction_offset_minutes: int = Field(60, ge=5, le=240)
    step_minutes: int = Field(5, ge=1, le=30)
    horizon_minutes: int = Field(60, ge=5, le=240)

    debug: bool = False


class ForecastPoint(BaseModel):
    time: str
    predicted_glucose: float
    step: int
    offset_minutes: int


class PeakInfo(BaseModel):
    peak_glucose: float
    peak_time: str
    peak_offset_minutes: int


class GlucosePredictResponse(BaseModel):
    predicted_glucose: float
    prediction_offset_minutes: int
    predicted_time: str

    forecast: Optional[List[ForecastPoint]] = None
    milestones: Optional[Dict[str, ForecastPoint]] = None
    peak: Optional[PeakInfo] = None

    debug: Optional[Dict[str, Any]] = None