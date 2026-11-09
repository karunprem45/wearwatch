"""FastAPI app: POST /predict, GET /health, GET /metrics."""
import os
from fastapi import FastAPI, HTTPException

from .service import Predictor

app = FastAPI(title="WearWatch")
predictor = Predictor(model_path=os.getenv("MODEL_PATH", "models/model.onnx"),
                      report_path=os.getenv("REPORT_PATH", "models/report.json"),
                      k=int(os.getenv("ALERT_K", "3")))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: dict):
    out = predictor.predict(reading)
    if not out["ok"]:
        raise HTTPException(status_code=422, detail=out["problems"])
    return out


@app.get("/metrics")
def metrics():
    return predictor.metrics()
