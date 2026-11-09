"""ONNX export and API tests. Skipped automatically if skl2onnx or fastapi isn't installed."""
import json
import numpy as np
import pytest

from wearwatch.data import synthetic_ai4i
from wearwatch.features import frame_features
from wearwatch.train import run


def test_onnx_matches_sklearn(tmp_path):
    pytest.importorskip("skl2onnx")
    ort = pytest.importorskip("onnxruntime")
    from wearwatch.export import to_onnx, parity
    model, _, test = run(synthetic_ai4i(2000))
    path = to_onnx(model, str(tmp_path / "m.onnx"))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    assert parity(model, sess, frame_features(test)) < 1e-4


def test_api_predict_and_metrics(tmp_path, monkeypatch):
    pytest.importorskip("skl2onnx")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from wearwatch.export import to_onnx
    from wearwatch.train import save_report
    model, report, _ = run(synthetic_ai4i(2000))
    to_onnx(model, str(tmp_path / "model.onnx"))
    save_report(report, str(tmp_path / "report.json"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.onnx"))
    monkeypatch.setenv("REPORT_PATH", str(tmp_path / "report.json"))
    import importlib
    import wearwatch.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/predict", json={"type": "L", "air_temp_k": 300, "process_temp_k": 310,
                                 "rpm": 1500, "torque_nm": 40, "tool_wear_min": 50,
                                 "machine_id": "m1"})
    assert r.status_code == 200 and 0 <= r.json()["failure_prob"] <= 1
    assert c.post("/predict", json={"type": "Z"}).status_code == 422
    assert c.get("/metrics").json()["predictions"] == 1
