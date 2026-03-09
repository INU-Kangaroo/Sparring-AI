from fastapi import FastAPI, HTTPException
from schema import GlucosePredictRequest, GlucosePredictResponse
from predict import predict_glucose, load_model_meta_calib

app = FastAPI(title="Glucose Prediction API", version="1.0.0")


@app.on_event("startup")
def _startup():
    # 서버 시작할 때 한 번만 로드해서 캐시 채우기(실패하면 즉시 알 수 있음)
    load_model_meta_calib()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-glucose", response_model=GlucosePredictResponse)
def predict_endpoint(req: GlucosePredictRequest):
    try:
        return predict_glucose(req.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"prediction failed: {e}")