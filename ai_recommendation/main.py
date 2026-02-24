from fastapi import FastAPI, HTTPException
from schema import RecommendationInput, RecommendationResult
from predict import recommend

app = FastAPI(title="Food Recommendation AI")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResult)
def recommend_api(data: RecommendationInput):
    # 기본 값 검증
    if data.height_cm <= 0 or data.weight_kg <= 0:
        raise HTTPException(status_code=400, detail="height_cm and weight_kg must be positive")

    if data.blood_glucose <= 0:
        raise HTTPException(status_code=400, detail="blood_glucose must be positive")

    if data.sbp <= 0 or data.dbp <= 0:
        raise HTTPException(status_code=400, detail="blood pressure values must be positive")

    if data.age_years <= 0:
        raise HTTPException(status_code=400, detail="age_years must be positive")

    if data.top_n is not None and (data.top_n <= 0 or data.top_n > 50):
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 50")

    # 음수 방지
    for k in ["eaten_kcal", "eaten_carb_g", "eaten_protein_g", "eaten_fat_g", "eaten_sodium_mg"]:
        v = getattr(data, k)
        if v is not None and v < 0:
            raise HTTPException(status_code=400, detail=f"{k} must be >= 0")

    for k in ["target_kcal", "target_carb_g", "target_protein_g", "target_fat_g", "target_sodium_mg"]:
        v = getattr(data, k)
        if v is not None and v < 0:
            raise HTTPException(status_code=400, detail=f"{k} must be >= 0")

    try:
        return recommend(data.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Data file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")