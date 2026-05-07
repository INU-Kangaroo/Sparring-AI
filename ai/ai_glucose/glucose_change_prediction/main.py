from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_meal_response
from schema import PredictMealResponseRequest, PredictMealResponseResponse


app = FastAPI(
    title="Meal Glucose Response API",
    description="식사 기반 혈당 반응 예측 API",
    version="8.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-glucose", response_model=PredictMealResponseResponse)
def predict_glucose_endpoint(req: PredictMealResponseRequest):
    try:
        req_dict = req.model_dump()
        result = predict_meal_response(req_dict)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")