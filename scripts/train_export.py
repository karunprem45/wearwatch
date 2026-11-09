"""Train, evaluate, export to ONNX, check parity and benchmark latency.

  python -m scripts.train_export                         # real data in data/ai4i2020.csv
  python -m scripts.train_export --synthetic             # code check only, never report
"""
import argparse
import json
import time
import numpy as np

from wearwatch.data import load_ai4i, synthetic_ai4i
from wearwatch.features import frame_features
from wearwatch.train import run, save_report
from wearwatch.export import to_onnx, parity


def bench(fn, X, single_rows=500):
    """p50/p95 latency for one-row calls, and rows per second for one big batch."""
    lat = []
    for i in range(min(single_rows, len(X))):
        t0 = time.perf_counter()
        fn(X[i:i + 1])
        lat.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    fn(X)
    rps = len(X) / (time.perf_counter() - t0)
    return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "batch_rows_per_s": float(rps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--target-recall", type=float, default=0.9)
    args = ap.parse_args()
    if args.synthetic:
        print("SYNTHETIC DATA: code check only. Do not report these numbers.\n")
        df = synthetic_ai4i()
    else:
        df = load_ai4i()
    print(f"rows: {len(df)}, failure rate: {df['failure'].mean():.2%}")

    model, report, test = run(df, args.target_recall)
    print(json.dumps(report, indent=2))

    path = to_onnx(model)
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    Xte = frame_features(test)
    report["parity_max_abs_diff"] = parity(model, sess, Xte)
    print(f"\nONNX parity, max |p_sklearn - p_onnx|: {report['parity_max_abs_diff']:.2e}")

    model.set_params(n_jobs=1)   # same single thread as ONNX Runtime, for a fair comparison
    report["latency_sklearn"] = bench(lambda X: model.predict_proba(X), Xte)
    report["latency_onnx"] = bench(lambda X: sess.run(None, {"input": X}), Xte)
    for k in ["latency_sklearn", "latency_onnx"]:
        r = report[k]
        print(f"{k:16s} p50 {r['p50_ms']:.3f} ms  p95 {r['p95_ms']:.3f} ms  "
              f"batch {r['batch_rows_per_s']:.0f} rows/s")
    save_report(report)
    test.to_csv("data/test_split.csv", index=False)
    print("\nsaved models/model.onnx, models/report.json, data/test_split.csv")


if __name__ == "__main__":
    main()
