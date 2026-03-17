from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schema import GlucosePredictRequest, GlucosePredictResponse
from predict import predict_glucose, load_model_meta_calib

app = FastAPI(title="Glucose Prediction API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


@app.on_event("startup")
def _startup():
    # 서버 시작 시 모델 로드
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