import joblib
from preprocess import preprocess_input

model = joblib.load("model/xgb_model.pkl")

def predict_glucose(input_data: dict) -> float:
    X = preprocess_input(input_data)
    pred = model.predict(X)
    return float(pred[0])
