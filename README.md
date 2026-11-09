# WearWatch

Predicts machine failures from sensor telemetry, serves the model with ONNX Runtime behind a FastAPI service, and turns noisy predictions into stable maintenance alerts. Everything runs with Docker Compose.

## Data

The **AI4I 2020 Predictive Maintenance** dataset from the UCI Machine Learning Repository: 10,000 milling-machine readings with air and process temperature, rotational speed, torque, tool wear and a machine-failure label. Download `ai4i2020.csv` and put it in `data/`.

The synthetic data in `wearwatch/data.py` only lets the tests run offline. **Never report its numbers.**

## How it fits together

```
ai4i2020.csv -> features.py -> train.py (random forest, threshold for 90% recall)
                                  |
                              export.py -> model.onnx  (parity-checked against scikit-learn)
                                  |
   replay.py --HTTP--> api.py -> service.py (ONNX Runtime) -> alerts.py (k-in-a-row + hysteresis)
   (docker compose: two services, model mounted read-only)
```

| File | What it does |
|---|---|
| `features.py` | Raw reading to features. Adds temperature difference, mechanical power and torque x tool wear, one per failure mechanism. Rejects impossible sensor values |
| `train.py` | 60/20/20 stratified split, random forest with balanced class weights, alert threshold picked on validation for 90% recall, PR-AUC against a baseline on test |
| `export.py` | Exports to ONNX with skl2onnx and checks ONNX Runtime matches scikit-learn |
| `alerts.py` | Raises an alert after k high readings in a row on one machine, clears after k low ones |
| `service.py` | ONNX Runtime session, validation, alerts and latency telemetry (p50/p95) |
| `api.py` | FastAPI: `POST /predict`, `GET /health`, `GET /metrics` |
| `scripts/train_export.py` | Train, evaluate, export, parity check and latency benchmark |
| `scripts/replay.py` | Streams held-out readings to the API as 5 simulated machines |
| `docker-compose.yml` | API (limited to 1 CPU and 512 MB, like a small edge box) plus the replay client |

## Run it, in this order

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -v                                  # all tests, including ONNX parity and the API
python -m scripts.train_export --synthetic # code check only
python -m scripts.train_export             # real data: writes models/ and data/test_split.csv

docker compose up --build                  # API + replay; watch the alerts
curl localhost:8000/metrics                # in another terminal while it runs
docker compose down
```

No Docker? Run the API with `uvicorn wearwatch.api:app --port 8000` and the replay with `python -m scripts.replay` in a second terminal.

## What to learn from each part (interview prep)

- **Why not accuracy?** About 3% of readings are failures. Predicting "never fails" scores about 97% accuracy and catches nothing. PR-AUC and recall at a chosen threshold measure what matters.
- **Why pick the threshold on validation, not test?** Picking it on test leaks test information into the model and makes the result look better than it is.
- **Why ONNX?** One portable model file, no scikit-learn in the serving image, and runtimes built for fast inference on CPU, GPU or edge devices. TensorRT is NVIDIA's runtime that can run ONNX models on their GPUs; this project stops at ONNX Runtime on CPU.
- **Why the parity check?** An export that silently changes predictions is a production bug nobody notices until machines fail.
- **Why k-in-a-row alerts with hysteresis?** One noisy reading shouldn't page a technician, and an alert shouldn't flicker on and off.
- **Why mount the model as a volume?** Updating the model is a file swap and restart, with no image rebuild.
- **Why limit the API to 1 CPU?** It mimics the small compute on an edge box, so the latency numbers mean something.

## Limits (say these if asked)

- AI4I rows are independent readings, not real time series. The replay deals them round-robin to 5 simulated machines to exercise the alert logic.
- AI4I is a synthetic dataset built to resemble real machine data.
- CPU only. No TensorRT and no GPU.
