from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_glucose
from schema import PredictGlucoseRequest, PredictGlucoseResponse


app = FastAPI(
    title="Glucose Prediction API",
    description="탄수화물 섭취 후 혈당 예측 API",
    version="1.0.0",
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


@app.post("/predict-glucose", response_model=PredictGlucoseResponse)
def predict_glucose_endpoint(req: PredictGlucoseRequest):
    try:
        req_dict = req.model_dump(by_alias=False)
        result = predict_glucose(req_dict)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")