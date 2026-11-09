"""Turn raw AI4I telemetry into the model's input features.

The same function runs in training and in the API, so the two can never drift apart.

Engineered features follow the physics of each failure mode in the dataset:
- heat dissipation failure: depends on how much hotter the process is than the air
- power failure: mechanical power = torque x angular speed
- overstrain failure: worn tools under high torque, so torque x tool wear
"""
import numpy as np

RAW = ["type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
FEATURES = ["type_code", "air_temp_k", "process_temp_k", "rpm", "torque_nm",
            "tool_wear_min", "temp_diff_k", "power_w", "strain"]
TYPE_CODE = {"L": 0.0, "M": 1.0, "H": 2.0}   # product quality variant: low, medium, high

# column names in the UCI CSV
UCI_COLUMNS = {
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
}


def row_features(r):
    """One reading (dict with RAW keys) -> list of floats in FEATURES order."""
    temp_diff = r["process_temp_k"] - r["air_temp_k"]
    power = r["torque_nm"] * r["rpm"] * 2.0 * np.pi / 60.0
    strain = r["torque_nm"] * r["tool_wear_min"]
    return [TYPE_CODE[r["type"]], r["air_temp_k"], r["process_temp_k"], r["rpm"],
            r["torque_nm"], r["tool_wear_min"], temp_diff, power, strain]


def frame_features(df):
    """DataFrame with RAW columns -> float32 matrix (n, len(FEATURES))."""
    out = np.empty((len(df), len(FEATURES)), dtype=np.float32)
    out[:, 0] = df["type"].map(TYPE_CODE).to_numpy()
    out[:, 1] = df["air_temp_k"]
    out[:, 2] = df["process_temp_k"]
    out[:, 3] = df["rpm"]
    out[:, 4] = df["torque_nm"]
    out[:, 5] = df["tool_wear_min"]
    out[:, 6] = df["process_temp_k"] - df["air_temp_k"]
    out[:, 7] = df["torque_nm"] * df["rpm"] * 2.0 * np.pi / 60.0
    out[:, 8] = df["torque_nm"] * df["tool_wear_min"]
    return out


def validate(r):
    """Reject readings no real sensor would produce. Returns a list of problems."""
    problems = []
    if r.get("type") not in TYPE_CODE:
        problems.append("type must be L, M or H")
    limits = {"air_temp_k": (250, 350), "process_temp_k": (250, 400), "rpm": (0, 5000),
              "torque_nm": (0, 200), "tool_wear_min": (0, 500)}
    for k, (lo, hi) in limits.items():
        v = r.get(k)
        if not isinstance(v, (int, float)) or not (lo <= v <= hi):
            problems.append(f"{k} must be a number in [{lo}, {hi}]")
    return problems
