# from fastapi import FastAPI, HTTPException
# from schema import GlucoseInput
# from predict import predict_glucose

# app = FastAPI()

# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "model_loaded": True
#     }

# @app.post("/predict")
# def predict(data: GlucoseInput):
#     result = predict_glucose(data.dict())
#     return {"predicted_glucose": result}


from fastapi import FastAPI, HTTPException
from schema import GlucoseInput
from predict import predict_glucose

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "glucose-prediction"
        }

@app.post("/glucose/predict")
def predict(data: GlucoseInput):

    # 혈당 히스토리 길이 검사
    if len(data.glucose_history) < 3:
        raise HTTPException(
            status_code=409,
            detail="glucose_history must contain at least 3 values"
        )

    # 비정상 값 검사
    if any(g <= 0 for g in data.glucose_history):
        raise HTTPException(
            status_code=400,
            detail="glucose values must be positive"
        )

    if data.age <= 0 or data.weight <= 0:
        raise HTTPException(
            status_code=400,
            detail="age and weight must be positive"
        )

    try:
        result = predict_glucose(data.dict())
        return {
            "predicted_glucose": float(result)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
