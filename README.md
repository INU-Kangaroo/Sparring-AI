# Glucose Change Prediction<br>(INU Capstone Design - Team Kangaroo)

식사 영양 정보, 기저 혈당, 성별을 입력받아 식후 혈당 반응을 예측하는 머신러닝 모델 및 API 서버입니다.  
이 저장소의 산출물은 앱 서비스에 연결할 수 있는 예측 모델과 FastAPI 기반 API 서버입니다.  
모델은 식후 혈당 변화량 곡선을 예측하고, 최고 상승 시점과 최고 상승량을 함께 제공합니다.

## Project Overview

이 프로젝트는 다음 정보를 바탕으로 식후 혈당 반응을 예측합니다.

- 기저 혈당
- 성별
- 식사 유형
- 탄수화물, 단백질, 지방, 식이섬유, 칼로리

예측 결과는 단일 수치가 아니라 시간 흐름을 반영한 형태로 제공됩니다.

- `peakDelta`: 식후 120분 내 최고 혈당 상승량
- `peakMinute`: 식후 120분 내 최고 혈당 상승 시점
- `curve`: 시간대별 혈당 변화량 곡선

`curve`는 변화량 기준으로 반환되며, `minute=0`, `30`, `60`, `120`과 피크 시점의 변화량을 포함합니다.

## Dataset

모델 학습에는 전처리된 CGMacros 기반 데이터셋을 사용합니다.  
데이터셋은 subject 기준으로 학습 데이터와 테스트 데이터로 분리되어 있습니다.

데이터셋 이름:

- 전체 데이터셋: `meal_glucose_with_gender.csv`
- 학습 데이터셋: `meal_glucose_with_gender_train.csv`
- 테스트 데이터셋: `meal_glucose_with_gender_test.csv`

데이터 규모:

- 전체 샘플 수: `1362`
- 학습 샘플 수: `1083`
- 테스트 샘플 수: `279`
- 전체 대상자 수: `45`
- 학습 대상자 수: `36`
- 테스트 대상자 수: `9`

서비스 모델 학습 시에는 `meal_type`이 `breakfast`, `lunch`, `dinner`에 해당하고 `sex` 값이 유효한 샘플만 사용했습니다.

- 학습에 실제 사용된 전체 샘플 수: `1205`
- 학습에 실제 사용된 학습 샘플 수: `953`
- 학습에 실제 사용된 테스트 샘플 수: `252`

## Input Features

서비스 모델은 아래 9개 feature를 사용합니다.

- `meal_type`
- `total_kcal`
- `total_carbs`
- `effective_carbs`
- `total_protein`
- `total_fat`
- `total_fiber`
- `baseline_glucose`
- `sex`

`meal_type`은 `breakfast`, `lunch`, `dinner` 세 가지 범주를 사용합니다.

영양 정보(`total_kcal`, `total_carbs`, `effective_carbs`, `total_protein`, `total_fat`, `total_fiber`)는 음식 영양 데이터베이스에서 제공되는 값을 사용합니다.

## Prediction Targets

모델은 아래 5개 target을 각각 회귀 방식으로 직접 학습합니다.

- `delta_30`
- `delta_60`
- `delta_120`
- `peak_delta`
- `peak_minute`

`peak_delta`와 `peak_minute`는 식후 120분 구간에서 실제 최고 혈당 상승량과 그 시점을 기준으로 만든 target입니다.
서비스 응답의 `peakDelta`와 `curve`는 변화량 기준 예측 결과입니다.

## Model

서비스 모델은 `CatBoostRegressor`를 사용하는 예측 모델입니다.  
`delta_30`, `delta_60`, `delta_120`, `peak_delta`, `peak_minute`를 각각 별도의 회귀 모델로 학습하고, 이 5개 모델을 하나의 `joblib` 파일에 함께 저장해 API에서 사용합니다.

- 최종 모델 파일: `models/service_model.joblib`
- 메타 정보: `models/service_model_meta.json`
- 오프라인 성능 파일: `models/service_model_metrics.json`
- API 성능 파일: `models/api_evaluation_metrics.json`

학습 입력은 `meal_glucose_with_gender_train.csv`, 평가 입력은 `meal_glucose_with_gender_test.csv`를 사용합니다.  
학습과 평가는 `meal_type`이 `breakfast`, `lunch`, `dinner`인 샘플과 유효한 `sex` 값을 가진 샘플만 포함합니다.

학습 스크립트:

- [train_service_model.py](glucose_change_prediction/train_service_model.py)

서비스 실행 코드:

- [main.py](glucose_change_prediction/main.py)
- [predict.py](glucose_change_prediction/predict.py)
- [schema.py](glucose_change_prediction/schema.py)

## Offline Performance

테스트 데이터셋 기준 오프라인 모델 성능은 다음과 같습니다.

| Target | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| `delta_30` | 20.69 | 27.45 | 0.2112 |
| `delta_60` | 34.96 | 44.40 | 0.1640 |
| `delta_120` | 27.22 | 38.23 | 0.3882 |
| `peak_delta` | 25.62 | 35.01 | 0.3917 |
| `peak_minute` | 29.16 | 34.54 | 0.0121 |

`models/service_model_metrics.json`은 테스트 데이터셋 기준 오프라인 평가 지표를 정리한 파일이며, 위 수치는 이 파일에 저장된 결과를 기준으로 작성했습니다.
`peakMinute`의 오프라인 기준 `±5분` 이내 비율은 약 `9.9%`, `±10분` 이내 비율은 약 `20.6%`, `±15분` 이내 비율은 약 `28.6%`입니다.

## API Performance

배포된 API 서버를 실제 호출해 측정한 성능은 다음과 같습니다.
아래 지표는 응답의 `peakDelta`, `peakMinute`, `curve`를 기준으로 계산한 평가 결과입니다.

| Target | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| `delta30` | 22.45 | 29.00 | 0.1194 |
| `delta60` | 37.23 | 47.80 | 0.0312 |
| `delta120` | 27.42 | 37.98 | 0.3962 |
| `peakDelta` | 25.62 | 35.01 | 0.3918 |
| `peakMinute` | 29.17 | 34.55 | 0.0118 |

`models/api_evaluation_metrics.json`은 배포 API를 실제 호출해 측정한 평가 지표를 정리한 파일이며, 위 수치는 이 파일에 저장된 결과를 기준으로 작성했습니다.
`peakMinute`의 API 기준 `±5분` 이내 비율은 약 `11.1%`, `±10분` 이내 비율은 약 `21.0%`, `±15분` 이내 비율은 약 `29.8%`입니다.

## API

엔드포인트:

- `GET /health`
- `POST /predict-glucose`

요청 예시:

```json
{
  "baselineGlucose": 95,
  "sex": "M",
  "meal": {
    "carbs": 60,
    "protein": 25,
    "fat": 18,
    "fiber": 6,
    "kcal": 520,
    "mealType": "lunch"
  }
}
```

응답 예시:

```json
{
  "peakDelta": 39.2,
  "peakMinute": 75,
  "curve": [
    { "minute": 0, "delta": 0.0 },
    { "minute": 30, "delta": 13.7 },
    { "minute": 60, "delta": 31.2 },
    { "minute": 75, "delta": 39.2 },
    { "minute": 120, "delta": 29.3 }
  ]
}
```

## Project Structure

- [main.py](glucose_change_prediction/main.py): FastAPI 서버 진입점
- [predict.py](glucose_change_prediction/predict.py): 모델 로딩 및 예측 로직
- [schema.py](glucose_change_prediction/schema.py): 요청/응답 스키마
- [service_data_utils.py](glucose_change_prediction/service_data_utils.py): 공통 서비스 데이터 전처리 유틸
- [train_service_model.py](glucose_change_prediction/train_service_model.py): 앱 서비스용 최종 모델 학습
- [evaluate_api.py](glucose_change_prediction/evaluate_api.py): API 응답 성능 검증
- [scripts/build_dataset_with_gender.py](glucose_change_prediction/scripts/build_dataset_with_gender.py): 최종 데이터셋 생성 스크립트
- [models](glucose_change_prediction/models): 모델 및 평가 결과

## Installation

```bash
pip install -r requirements.txt
```

## Run

API 서버 실행:

```bash
uvicorn main:app --reload
```

모델 학습:

```bash
python train_service_model.py
```

API 성능 검증:

```bash
python evaluate_api.py
```

`.env` 예시:

```env
GLUCOSE_API_BASE_URL=http://<host>:<port>
```

## Tech Stack

- Python
- FastAPI
- CatBoost
- pandas
- scikit-learn
- joblib
