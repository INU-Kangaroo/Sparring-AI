import numpy as np
from datetime import datetime

def map_bp_to_proxy(systolic: int, diastolic: int) -> int:
    if systolic >= 140 or diastolic >= 90:
        return 2  # 높음
    elif systolic >= 120 or diastolic >= 80:
        return 1  # 보통
    else:
        return 0  # 낮음

def preprocess_input(data: dict):
    history = data["glucose_history"]

    # 혈당 시계열 파생
    g1 = history[-1]
    g3 = history[-3] if len(history) >= 3 else history[0]
    g6 = history[-6] if len(history) >= 6 else history[0]

    roll_mean = np.mean(history[-3:])
    delta_6 = g1 - g6

    # 시간 파생
    ts = datetime.fromisoformat(data["timestamp"])
    hour = ts.hour
    weekday = ts.weekday()

    # 혈압 상태 변환
    proxy_bp = map_bp_to_proxy(
        data["systolic_bp"],
        data["diastolic_bp"]
    )

    features = [
        g1,            # glucose_lag_1
        g3,            # glucose_lag_3
        g6,            # glucose_lag_6
        roll_mean,     # glucose_roll_mean
        delta_6,       # glucose_delta_6
        data["carb_intake"],
        data["meal_type"],
        data["steps"],
        data["intensity"],
        proxy_bp,
        data["age"],
        data["sex"],
        data["weight"],
        hour,
        weekday,
        data["alcohol"],
        data["medication"],
        data["caffeine"],
    ]

    return np.array(features, dtype=float).reshape(1, -1)
