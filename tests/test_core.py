import numpy as np
import pandas as pd

from wearwatch.features import row_features, frame_features, validate, FEATURES
from wearwatch.alerts import AlertTracker
from wearwatch.data import synthetic_ai4i
from wearwatch.train import run, pick_threshold
from wearwatch.service import Predictor

READING = {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0, "rpm": 1500.0,
           "torque_nm": 40.0, "tool_wear_min": 100.0}


def test_row_and_frame_features_agree():
    df = pd.DataFrame([READING, {**READING, "type": "H", "torque_nm": 60.0}])
    X = frame_features(df)
    assert X.shape == (2, len(FEATURES)) and X.dtype == np.float32
    assert np.allclose(X[0], np.array(row_features(READING), dtype=np.float32))


def test_power_feature_is_physical():
    f = dict(zip(FEATURES, row_features(READING)))
    assert abs(f["power_w"] - 40 * 1500 * 2 * np.pi / 60) < 1e-6   # about 6283 W
    assert f["temp_diff_k"] == 10.0 and f["strain"] == 4000.0


def test_validate_rejects_bad_readings():
    assert validate(READING) == []
    assert validate({**READING, "type": "X"})
    assert validate({**READING, "rpm": -5})
    assert validate({**READING, "torque_nm": "high"})


def test_alert_needs_k_in_a_row_and_clears_with_hysteresis():
    t = AlertTracker(threshold=0.5, k=3)
    seq = [0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1]
    events = [t.update("m1", p) for p in seq]
    assert events.index("raise") == 5          # only on the 3rd high reading in a row
    assert events.count("raise") == 1
    assert events[-1] == "clear" and t.active() == []


def test_alerts_are_per_machine():
    t = AlertTracker(threshold=0.5, k=2)
    for p in [0.9, 0.9]:
        t.update("a", p)
        t.update("b", 0.1)
    assert t.active() == ["a"]


def test_threshold_meets_recall_target():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    p = np.array([.1, .2, .3, .35, .5, .7, .9, .6, .8, .05])
    t = pick_threshold(y, p, 0.8)
    assert ((p >= t) & (y == 1)).sum() / y.sum() >= 0.8
    assert t == 0.5


def test_model_beats_baseline_on_synthetic():
    _, report, test = run(synthetic_ai4i(3000), target_recall=0.9)
    assert report["test"]["pr_auc"] > 3 * report["baseline_pr_auc"]
    assert report["test"]["recall"] >= 0.7
    assert set(test.columns) >= {"failure", "torque_nm"}


class FakeSession:
    """Stands in for onnxruntime: probability rises with torque."""
    def run(self, _, feeds):
        x = feeds["input"]
        p = np.clip((x[:, 4] - 30) / 40, 0, 1)
        return [(p > 0.5).astype(np.int64), np.stack([1 - p, p], axis=1)]


def test_predictor_end_to_end_with_fake_session():
    pr = Predictor(session=FakeSession(), threshold=0.5, k=2)
    low = {**READING, "machine_id": "m1", "torque_nm": 32.0}
    high = {**READING, "machine_id": "m1", "torque_nm": 68.0}
    assert pr.predict(low)["alert_event"] is None
    assert pr.predict(high)["alert_event"] is None
    out = pr.predict(high)
    assert out["alert_event"] == "raise" and out["alerting"]
    assert pr.predict({**READING, "rpm": -1})["ok"] is False
    m = pr.metrics()
    assert m["predictions"] == 3 and m["rejected"] == 1 and m["machines_alerting"] == ["m1"]
    assert m["latency_ms_p50"] >= 0
