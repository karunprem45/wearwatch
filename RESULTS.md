# WearWatch results log (real AI4I data only)

## Tests
- `pytest -v`: ____ passed, ____ skipped

## Model (`python -m scripts.train_export`)
- Rows / failure rate:
- Threshold (picked on validation for 90% recall):
- Test PR-AUC: ____ vs baseline ____
- Test recall: ____  precision: ____
- Top 3 features:

## ONNX
- Parity max |p_sklearn - p_onnx|:
- Single-reading latency, scikit-learn: p50 ____ ms, p95 ____ ms
- Single-reading latency, ONNX Runtime: p50 ____ ms, p95 ____ ms
- Batch throughput (rows/s), scikit-learn ____ vs ONNX ____

## Docker Compose run
- Readings replayed / errors:
- Alerts raised / cleared:
- API p50 / p95 latency from /metrics:

## One thing that broke and how I fixed it
(Your interview story, in your own words.)
