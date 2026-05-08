import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_minutes(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    diff = np.abs(y_true_arr - y_pred_arr)
    return {
        "mae": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "exact_match": float(np.mean(diff == 0)),
        "within_5min": float(np.mean(diff <= 5)),
        "within_10min": float(np.mean(diff <= 10)),
        "within_15min": float(np.mean(diff <= 15)),
    }
