# import pandas as pd
# from xgboost import XGBRegressor
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error

# # =========================
# # 1. 데이터 로드
# # =========================
# df = pd.read_csv("data/glucobench.csv")

# # =========================
# # 2. 시간 정렬
# # =========================
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# df = df.sort_values("timestamp")

# # =========================
# # 3. 이전 혈당 / 현재 혈당 / 변화량
# # =========================
# df["glucose_prev"] = df["glucose"].shift(1)
# df["glucose_prev_2"] = df["glucose"].shift(2)
# df["delta_glucose"] = df["glucose_prev"] - df["glucose_prev_2"]

# # =========================
# # 4. target (다음 혈당)
# # =========================
# df["target_glucose"] = df["glucose"].shift(-6)

# # =========================
# # 5. 시간 파생 변수
# # =========================
# df["hour"] = df["timestamp"].dt.hour
# df["weekday"] = df["timestamp"].dt.weekday

# # =========================
# # 6. meal_type 숫자화
# # =========================
# meal_map = {
#     "none": 0,
#     "breakfast": 1,
#     "lunch": 2,
#     "dinner": 3,
#     "snack": 4
# }
# df["meal_type"] = df["meal_type"].map(meal_map).fillna(0).astype(int)

# # =========================
# # 7. 성별 숫자화
# # =========================
# df["sex"] = df["sex"].map({"M": 0, "F": 1}).fillna(0).astype(int)

# # =========================
# # 8. 운동 강도 숫자화
# # =========================
# intensity_map = {
#     "none": 0.0,
#     "low": 0.3,
#     "moderate": 0.6,
#     "high": 0.9
# }
# df["intensity"] = df["exercise_intensity"].map(intensity_map).fillna(0.0)

# # =========================
# # 9. proxy_bp 생성 (PDF 설계 반영)
# # =========================
# df["proxy_bp_raw"] = (
#     0.5 * df["heart_rate"].fillna(0) +
#     0.3 * df["stress_level"].fillna(0) +
#     0.2 * df["gsr"].fillna(0)
# )

# # qcut 안전 적용
# if df["proxy_bp_raw"].nunique() >= 3:
#     df["proxy_bp"] = pd.qcut(
#         df["proxy_bp_raw"],
#         q=3,
#         labels=[0, 1, 2],
#         duplicates="drop"
#     ).astype(int)
# else:
#     df["proxy_bp"] = (df["proxy_bp_raw"] > df["proxy_bp_raw"].mean()).astype(int)

# # =========================
# # 10. medication / caffeine 처리
# # =========================
# df["medication"] = df["medication_other"].notnull().astype(int)
# df["caffeine"] = 0  # 데이터셋에 없으므로 더미 변수

# # =========================
# # 11. 사용할 feature 정의 (FastAPI와 동일)
# # =========================
# FEATURE_COLS = [
#     "glucose_prev",
#     "delta_glucose",
#     "carbs",
#     "meal_type",
#     "exercise_steps",
#     "intensity",
#     "proxy_bp",
#     "age",
#     "sex",
#     "weight",
#     "hbA1c",
#     "carb_ratio",
#     "insulin_sensitivity",
#     "hour",
#     "weekday",
#     "alcohol",
#     "medication",
#     "caffeine"
# ]

# # =========================
# # 12. 학습 데이터 정리
# # =========================
# df = df[FEATURE_COLS + ["target_glucose"]].dropna()

# X = df[FEATURE_COLS]
# y = df["target_glucose"]

# print(X.dtypes)  # 디버깅용

# # =========================
# # 13. Train / Validation 분리 (시계열)
# # =========================
# X_train, X_val, y_train, y_val = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     shuffle=False
# )

# # =========================
# # 14. 모델 학습
# # =========================
# model = XGBRegressor(
#     n_estimators=200,
#     max_depth=4,
#     learning_rate=0.05,
#     random_state=42
# )

# model.fit(X_train, y_train)

# # =========================
# # 15. Validation 성능 평가
# # =========================
# val_pred = model.predict(X_val)
# mae = mean_absolute_error(y_val, val_pred)

# print(f"Validation MAE: {mae:.2f} mg/dL")

# # =========================
# # 16. 모델 저장
# # =========================
# joblib.dump(model, "model/xgb_model.pkl")
# print("모델 학습 완료 및 저장")




import pandas as pd
from xgboost import XGBRegressor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# =========================
# 1. 데이터 로드
# =========================
df = pd.read_csv("data/glucobench.csv")

# =========================
# 2. 시간 정렬
# =========================
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# =========================
# 3. 혈당 시계열 파생
# =========================
df["glucose_lag_1"] = df["glucose"].shift(1)
df["glucose_lag_3"] = df["glucose"].shift(3)
df["glucose_lag_6"] = df["glucose"].shift(6)
df["glucose_roll_mean"] = df["glucose"].rolling(3).mean()
df["glucose_delta_6"] = df["glucose_lag_1"] - df["glucose_lag_6"]

# =========================
# 4. target (다음 시점 혈당)
# =========================
df["target_glucose"] = df["glucose"].shift(-1)

# =========================
# 5. 시간 파생
# =========================
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

# =========================
# 6. 범주/수치 변환
# =========================
df["meal_type"] = df["meal_type"].map({
    "none": 0, "breakfast": 1, "lunch": 2, "dinner": 3, "snack": 4
}).fillna(0)

df["sex"] = df["sex"].map({"M": 0, "F": 1}).fillna(0)

df["intensity"] = df["exercise_intensity"].map({
    "none": 0.0, "low": 0.3, "moderate": 0.6, "high": 0.9
}).fillna(0.0)

# =========================
# 7. proxy_bp (학습용)
# =========================
df["proxy_bp"] = pd.qcut(
    0.5 * df["heart_rate"].fillna(0)
    + 0.3 * df["stress_level"].fillna(0)
    + 0.2 * df["gsr"].fillna(0),
    q=3,
    labels=[0, 1, 2],
    duplicates="drop"
).astype(int)

# =========================
# 8. 컬럼 정리
# =========================
df["carb_intake"] = df["carbs"]
df["steps"] = df["exercise_steps"]
df["medication"] = df["medication_other"].notnull().astype(int)
df["caffeine"] = 0

FEATURE_COLS = [
    "glucose_lag_1",
    "glucose_lag_3",
    "glucose_lag_6",
    "glucose_roll_mean",
    "glucose_delta_6",
    "carb_intake",
    "meal_type",
    "steps",
    "intensity",
    "proxy_bp",
    "age",
    "sex",
    "weight",
    "hour",
    "weekday",
    "alcohol",
    "medication",
    "caffeine"
]

df = df[FEATURE_COLS + ["target_glucose"]].dropna()

X = df[FEATURE_COLS]
y = df["target_glucose"]

# =========================
# 9. Train / Validation
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_val)
print(f"Validation MAE: {mean_absolute_error(y_val, pred):.2f} mg/dL")

joblib.dump(model, "model/xgb_model.pkl")
print("모델 학습 완료")
