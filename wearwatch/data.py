"""Load the UCI AI4I 2020 Predictive Maintenance dataset, or make a synthetic copy for tests.

Real data: download ai4i2020.csv from the UCI Machine Learning Repository into data/.
Synthetic data only exists so the tests run offline. Never report its numbers.
"""
import numpy as np
import pandas as pd

from .features import UCI_COLUMNS


def load_ai4i(path="data/ai4i2020.csv"):
    df = pd.read_csv(path)
    missing = [c for c in UCI_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"not the AI4I file, missing columns: {missing}")
    return df.rename(columns=UCI_COLUMNS)[list(UCI_COLUMNS.values())]


def synthetic_ai4i(n=4000, seed=0):
    """Same columns and roughly the same rules as AI4I, so the code paths get exercised."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    proc = air + 10 + rng.normal(0, 1, n)
    rpm = rng.normal(1540, 180, n).clip(1100, 2900)
    torque = rng.normal(40, 10, n).clip(3, 77)
    wear = rng.integers(0, 254, n).astype(float)
    typ = rng.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1])
    power = torque * rpm * 2 * np.pi / 60
    fail = ((proc - air < 8.6) & (rpm < 1380)) | (power < 3500) | (power > 9000) \
        | (torque * wear > 11000) | (rng.random(n) < 0.002)
    return pd.DataFrame({"type": typ, "air_temp_k": air, "process_temp_k": proc, "rpm": rpm,
                         "torque_nm": torque, "tool_wear_min": wear, "failure": fail.astype(int)})
