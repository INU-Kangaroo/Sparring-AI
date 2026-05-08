from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_meal_response
from schema import PredictRequest, PredictResponse


app = FastAPI()

# 외부 클라이언트에서 바로 호출할 수 있게 기본 CORS 허용
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
    # 요청 스키마를 모델 입력 형태로 넘겨 예측 결과 반환
    return predict_meal_response(request.model_dump())
