def soften_curve_near_peak(
    peak_minute: int,
    delta30: float,
    delta60: float,
    delta120: float,
    peak_delta: float,
) -> tuple[float, float, float]:
    adjusted = {
        30: float(delta30),
        60: float(delta60),
        120: float(delta120),
    }

    for anchor_minute in (30, 60, 120):
        distance = abs(peak_minute - anchor_minute)
        if distance == 0 or distance > 60:
            continue

        proximity = 1.0 - (distance / 60.0)
        proximity_weight = proximity * proximity
        if anchor_minute == 30:
            anchor_bias = 0.03
            strength = min(0.45, 0.04 + proximity_weight * 0.24 + anchor_bias)
        elif anchor_minute == 60:
            anchor_bias = 0.12
            strength = min(0.72, 0.08 + proximity_weight * 0.45 + anchor_bias)
        else:
            anchor_bias = 0.04
            strength = min(0.38, 0.05 + proximity_weight * 0.18 + anchor_bias)

        current_value = adjusted[anchor_minute]
        lifted_value = current_value + (peak_delta - current_value) * strength
        min_gap = 0.8 if distance <= 10 else 1.2 if distance <= 20 else 1.8
        adjusted[anchor_minute] = min(peak_delta - min_gap, lifted_value)

    return adjusted[30], adjusted[60], adjusted[120]


def build_curve_points(
    delta30: float,
    delta60: float,
    delta120: float,
    peak_delta: float,
    peak_minute: int,
) -> list[dict[str, float | int]]:
    points = {0: 0.0, 30: delta30, 60: delta60, 120: delta120}
    if peak_minute not in points:
        points[peak_minute] = peak_delta
    else:
        points[peak_minute] = max(points[peak_minute], peak_delta)

    return [
        {"minute": minute, "delta": round(points[minute], 1)}
        for minute in sorted(points)
    ]
