# Glucose Change Prediction<br>(INU Capstone Design - Team Kangaroo)

식사 영양 정보, 기저 혈당, 성별을 입력받아 식후 혈당 반응을 예측하는 머신러닝 모델 및 API 서버입니다.  
이 저장소의 최종 산출물은 앱 서비스에 연결해 사용할 수 있는 예측 모델이며, FastAPI 기반 API 서버와 함께 구성되어 있습니다.  
학습된 CatBoost 모델은 혈당 변화량과 반응 곡선을 반환합니다.

## Project Overview

이 프로젝트는 다음 정보를 바탕으로 식후 혈당 반응을 예측합니다.

- 기저 혈당
- 성별
- 식사 유형
- 탄수화물, 단백질, 지방, 식이섬유, 칼로리

예측 결과는 단일 수치가 아니라 시간 흐름을 반영한 형태로 제공됩니다.

- `delta30`: 식후 30분 혈당 변화량
- `delta60`: 식후 60분 혈당 변화량
- `peakDelta`: 최고 혈당 상승량
- `peakMinute`: 최고점 도달 시점
- `curve`: 시간대별 혈당 반응 곡선

## Dataset

모델 학습에는 전처리된 CGMacros 기반 데이터셋을 사용했습니다.  
학습 과정에서는 전체 데이터셋을 만든 뒤, subject 기준으로 학습 데이터와 테스트 데이터를 별도로 분리해 사용했습니다.

데이터셋 이름:

- 전체 데이터셋: `last_final_no_activity_plus_clean_with_Gender.csv`
- 학습 데이터셋: `last_final_no_activity_plus_clean_with_Gender_train.csv`
- 테스트 데이터셋: `last_final_no_activity_plus_clean_with_Gender_test.csv`

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

최종 서비스 모델은 아래 9개 feature를 사용합니다.

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

## Why These Features

최종 서비스 모델은 앱에서 실제로 사용자가 쉽게 입력할 수 있는 정보만 사용하는 것을 원칙으로 했습니다.  
사용자가 직접 입력해야 하는 항목이 많아질수록 입력 부담이 커지고, 서비스 사용성도 떨어지기 때문에 현재 버전에서는 입력 단계를 단순하게 유지했습니다.

영양 정보(`total_kcal`, `total_carbs`, `effective_carbs`, `total_protein`, `total_fat`, `total_fiber`)는 사용자가 직접 계산해서 넣는 값이 아니라, 앱이 보유한 음식 영양 데이터베이스를 통해 제공되는 값을 사용합니다.  
즉, 사용자 입장에서는 식사 정보를 고르면 되고, 모델에는 해당 음식의 영양 정보가 자동으로 들어가는 구조를 전제로 합니다.

사용자 정보 중에서는 `sex`만 최종 feature로 채택했습니다.  
`age`, `bmi`, `height_cm`, `weight_kg` 등 다른 사용자 정보와 그 조합도 모두 비교했지만, 전체적으로는 `sex`만 사용하는 구성이 가장 안정적으로 좋은 성능을 보였습니다.  
세부적으로는 4개 예측 target 중 `delta_30`, `delta_60`, `peak_delta`에서는 `sex` 단독 조합이 가장 좋았고, `delta_120`에서는 `sex + bmi` 조합이 가장 좋았습니다.  
다만 서비스 모델은 전체적인 성능, 입력 단순성, 사용자 입력 부담을 함께 고려해 최종적으로 성별만 반영했습니다.

상세 비교 결과는 `compare_userinfo_feature_sets.py`와 `models/compare_userinfo_feature_sets_results.json`에서 확인할 수 있습니다.

## Prediction Targets

모델은 아래 4개 target을 각각 회귀 방식으로 학습합니다.

- `delta_30`
- `delta_60`
- `delta_120`
- `peak_delta`

서비스 응답에서는 이 예측값을 기반으로 혈당 반응 곡선을 함께 반환합니다.

## Model

최종 모델은 `CatBoostRegressor` 기반입니다.  
각 target마다 별도의 회귀 모델을 학습하고, 이를 하나의 모델 번들로 저장해 앱 서비스에서 사용합니다.

- 최종 모델 파일: `models/final_catboost_service_shape_with_sex.joblib`
- 메타 정보: `models/final_catboost_service_shape_with_sex_meta.json`
- 예측 결과 저장 파일: `data/processed/final_catboost_service_shape_with_sex_predictions.csv`

학습 스크립트:

- [final_train_model_service_shape.py](glucose_change_prediction/final_train_model_service_shape.py)

서비스 실행 코드:

- [main.py](glucose_change_prediction/main.py)
- [predict.py](glucose_change_prediction/predict.py)
- [schema.py](glucose_change_prediction/schema.py)

## Model Performance

테스트 데이터셋 기준 오프라인 모델 성능은 다음과 같습니다.

| Target | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| `delta_30` | 20.69 | 27.45 | 0.2112 |
| `delta_60` | 34.96 | 44.40 | 0.1640 |
| `delta_120` | 27.22 | 38.23 | 0.3882 |
| `peak_delta` | 25.62 | 35.01 | 0.3917 |

위 수치는 현재 프로젝트에서 다시 학습해 생성한 `models/final_catboost_service_shape_with_sex_meta.json`의 테스트 결과를 기준으로 작성했습니다.

배포된 API 서버 실측 성능은 다음과 같습니다.

| Target | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| `delta30` | 20.37 | 27.24 | 0.2228 |
| `delta60` | 34.36 | 43.78 | 0.1873 |
| `delta120` | 28.69 | 42.80 | 0.2334 |
| `peakDelta` | 27.85 | 38.78 | 0.2537 |

위 수치는 배포 API를 실제 호출해 생성한 `models/test_server_api_catboost_service_shape_metrics.json`을 기준으로 작성했습니다.

서버 응답 기준 상세 결과는 아래 파일에 저장되어 있습니다.

- `models/test_server_api_catboost_service_shape_metrics.json`

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
  "delta30": 12.4,
  "delta60": 24.8,
  "peakDelta": 27.1,
  "peakMinute": 75,
  "curve": [
    { "minute": 0, "delta": 0.0 },
    { "minute": 30, "delta": 12.4 },
    { "minute": 60, "delta": 24.8 },
    { "minute": 75, "delta": 27.1 },
    { "minute": 120, "delta": 18.6 }
  ]
}
```

## Project Structure

- [main.py](glucose_change_prediction/main.py): FastAPI 서버 진입점
- [predict.py](glucose_change_prediction/predict.py): 모델 로딩 및 예측 로직
- [schema.py](glucose_change_prediction/schema.py): 요청/응답 스키마
- [final_train_model_service_shape.py](glucose_change_prediction/final_train_model_service_shape.py): 앱 서비스용 최종 모델 학습
- [final_api_test_service_shape.py](glucose_change_prediction/final_api_test_service_shape.py): API 응답 성능 검증
- [compare_userinfo_feature_sets.py](glucose_change_prediction/compare_userinfo_feature_sets.py): 사용자 정보 feature 조합 비교 실험
- [scripts/build_last_final_no_activity_plus_with_gender.py](glucose_change_prediction/scripts/build_last_final_no_activity_plus_with_gender.py): 최종 데이터셋 생성 스크립트
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
python final_train_model_service_shape.py
```

API 성능 검증:

```bash
python final_api_test_service_shape.py
```

`.env` 예시:

```env
GLUCOSE_API_BASE_URL=http://<host>:<port>
```

배포 서버 기준 API 성능 검증:

```bash
python final_api_test_service_shape.py
```

## Tech Stack

- Python
- FastAPI
- CatBoost
- pandas
- scikit-learn
- joblib
