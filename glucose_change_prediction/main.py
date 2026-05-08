from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_meal_response
from schema import PredictRequest, PredictResponse


app = FastAPI()

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


@app.post("/predict-glucose", response_model=PredictResponse)
def predict_glucose(request: PredictRequest):
    try:
        return predict_meal_response(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Model file error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc
