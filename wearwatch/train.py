"""Train and evaluate the failure classifier.

Failures are rare (about 3% in AI4I), so accuracy is useless: predicting "no failure"
every time scores about 97%. We report PR-AUC and pick the alert threshold on a
validation split to hit a recall target, then check it once on the untouched test split.
"""
import json
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .features import frame_features, FEATURES


def split(df, seed=0):
    """60/20/20 stratified train/validation/test split."""
    tr, rest = train_test_split(df, test_size=0.4, stratify=df["failure"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=rest["failure"], random_state=seed)
    return tr, va, te


def make_model(seed=0):
    return RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def pick_threshold(y, p, target_recall=0.9):
    """Highest threshold whose recall is still at least target_recall."""
    best = 0.0
    for t in np.unique(p):
        if recall_score(y, p >= t, zero_division=0) >= target_recall:
            best = max(best, float(t))
    return best


def evaluate(y, p, threshold):
    pred = p >= threshold
    return {"pr_auc": float(average_precision_score(y, p)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "alerts": int(pred.sum()), "failures": int(y.sum()), "n": int(len(y))}


def run(df, target_recall=0.9, seed=0):
    tr, va, te = split(df, seed)
    Xtr, ytr = frame_features(tr), tr["failure"].to_numpy()
    Xva, yva = frame_features(va), va["failure"].to_numpy()
    Xte, yte = frame_features(te), te["failure"].to_numpy()

    model = make_model(seed).fit(Xtr, ytr)
    thr = pick_threshold(yva, model.predict_proba(Xva)[:, 1], target_recall)
    test = evaluate(yte, model.predict_proba(Xte)[:, 1], thr)

    base = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    baseline_pr_auc = float(average_precision_score(yte, base.predict_proba(Xte)[:, 1]))

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda kv: -kv[1])
    report = {"threshold": thr, "target_recall": target_recall, "test": test,
              "baseline_pr_auc": baseline_pr_auc,
              "top_features": [[k, round(float(v), 4)] for k, v in imp[:5]]}
    return model, report, te


def save_report(report, path="models/report.json"):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
