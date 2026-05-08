MODEL_TARGETS = [
    "delta_30",
    "delta_60",
    "delta_120",
    "peak_delta",
    "peak_minute",
]

API_RESPONSE_REQUIRED_KEYS = {"peakDelta", "peakMinute", "curve"}

API_EVALUATION_TARGETS = {
    "delta30": {
        "true_col": "delta_30",
        "pred_col": "pred_delta30",
        "metric_kind": "regression",
    },
    "delta60": {
        "true_col": "delta_60",
        "pred_col": "pred_delta60",
        "metric_kind": "regression",
    },
    "delta120": {
        "true_col": "delta_120",
        "pred_col": "pred_delta120",
        "metric_kind": "regression",
    },
    "peakDelta": {
        "true_col": "peak_delta",
        "pred_col": "pred_peakDelta",
        "metric_kind": "regression",
    },
    "peakMinute": {
        "true_col": "peak_minute",
        "pred_col": "pred_peakMinute",
        "metric_kind": "minutes",
    },
}
