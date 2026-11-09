"""Inference core used by the API: ONNX Runtime session, alert tracker and latency telemetry.

Kept separate from the web framework so it can be unit-tested without a server.
"""
import json
import time
from collections import deque

import numpy as np

from .features import row_features, validate
from .alerts import AlertTracker


class Predictor:
    def __init__(self, model_path="models/model.onnx", report_path="models/report.json",
                 k=3, session=None, threshold=None):
        if session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1          # predictable latency on a small edge box
            session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        if threshold is None:
            with open(report_path) as f:
                threshold = json.load(f)["threshold"]
        self.session = session
        self.alerts = AlertTracker(threshold=threshold, k=k)
        self.latency_ms = deque(maxlen=5000)
        self.count = 0
        self.rejected = 0

    def predict(self, reading):
        problems = validate(reading)
        if problems:
            self.rejected += 1
            return {"ok": False, "problems": problems}
        x = np.array([row_features(reading)], dtype=np.float32)
        t0 = time.perf_counter()
        prob = float(self.session.run(None, {"input": x})[1][0, 1])
        self.latency_ms.append((time.perf_counter() - t0) * 1000.0)
        self.count += 1
        event = self.alerts.update(reading.get("machine_id", "unknown"), prob)
        return {"ok": True, "failure_prob": round(prob, 4), "alert_event": event,
                "alerting": reading.get("machine_id", "unknown") in self.alerts.active()}

    def metrics(self):
        lat = np.array(self.latency_ms) if self.latency_ms else np.array([np.nan])
        return {"predictions": self.count, "rejected": self.rejected,
                "latency_ms_p50": round(float(np.nanpercentile(lat, 50)), 4),
                "latency_ms_p95": round(float(np.nanpercentile(lat, 95)), 4),
                "alerts_raised": self.alerts.raised, "alerts_cleared": self.alerts.cleared,
                "machines_alerting": self.alerts.active()}
