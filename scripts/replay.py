"""Replay held-out telemetry against the running API, like a fleet of machines streaming readings.

AI4I rows are independent products, not real time series. Rows are dealt round-robin to
N simulated machines so the alert logic has streams to work on. Say this if asked.
"""
import argparse
import os
import time
import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000"))
    ap.add_argument("--csv", default="data/test_split.csv")
    ap.add_argument("--machines", type=int, default=5)
    ap.add_argument("--rate", type=float, default=50.0, help="readings per second")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for _ in range(30):                       # wait for the API to come up
        try:
            if requests.get(f"{args.api}/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(1)
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    events, errors = [], 0
    t0 = time.perf_counter()
    for i, row in enumerate(df.to_dict("records")):
        reading = {k: row[k] for k in ["type", "air_temp_k", "process_temp_k", "rpm",
                                       "torque_nm", "tool_wear_min"]}
        reading["machine_id"] = f"machine-{i % args.machines}"
        r = requests.post(f"{args.api}/predict", json=reading, timeout=5)
        if not r.ok:
            errors += 1
            continue
        out = r.json()
        if out["alert_event"]:
            events.append((i, reading["machine_id"], out["alert_event"], out["failure_prob"]))
            print(f"reading {i:5d} {reading['machine_id']}: {out['alert_event'].upper()} "
                  f"(p={out['failure_prob']:.2f})")
        time.sleep(max(0.0, (i + 1) / args.rate - (time.perf_counter() - t0)))
    print(f"\nsent {len(df)} readings, {errors} errors, {len(events)} alert events")
    print("server metrics:", requests.get(f"{args.api}/metrics", timeout=5).json())


if __name__ == "__main__":
    main()
