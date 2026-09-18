"""
compare_targets_voladj.py -- EXPERIMENTAL, NOT PRODUCTION.

Runs the IDENTICAL training/evaluation code (same chronological split,
embargo, feature columns, scaler, classifier, hyperparameters, random seed,
metrics) against two tables:
  1. Data/persistence_training_table.csv           (baseline, untouched)
  2. Data/persistence_training_table_voladj.csv     (experimental)

The only difference between the two runs is which table's `usable`/`label`
columns are used -- everything else in this script is one code path shared
by both, so the comparison is apples-to-apples by construction.

Run: `python compare_targets_voladj.py` (from src/).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                              brier_score_loss, precision_recall_fscore_support, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaling import fit_transform_fold
import persistence_common

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(_ROOT, "Data", "persistence_training_table.csv")
VOLADJ_PATH = os.path.join(_ROOT, "Data", "persistence_training_table_voladj.csv")
OUT_DIR = os.path.join(_ROOT, "Data", "voladj_experiment")

HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
EMBARGO_BARS = 48
FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]
CLF_PARAMS = dict(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_and_prep(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df["log1p_duration"] = persistence_common.duration_feature(df["duration"].values)
    return df


def chronological_split(df):
    embargo_start = HOLDOUT_START - pd.Timedelta(minutes=5 * EMBARGO_BARS)
    train = df[df.index < embargo_start]
    holdout = df[df.index >= HOLDOUT_START]
    return train, holdout


def run_one(name, path):
    log(f"=== {name} ===")
    df = load_and_prep(path)
    usable = df[df["usable"] == True].copy()  # noqa: E712
    train, holdout = chronological_split(usable)
    y_train, y_holdout = train["label"].astype(int).values, holdout["label"].astype(int).values

    train_scaled, holdout_scaled, scaler = fit_transform_fold(train[FEATURE_COLUMNS], holdout[FEATURE_COLUMNS])
    model = GradientBoostingClassifier(**CLF_PARAMS)
    model.fit(train_scaled.values, y_train)
    p_holdout = model.predict_proba(holdout_scaled.values)[:, 1]
    y_pred = (p_holdout >= 0.5).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_holdout, y_pred, average="binary", zero_division=0)
    metrics = {
        "total_rows": len(df), "usable_rows": len(usable),
        "label_1": int((usable["label"] == 1).sum()), "label_0": int((usable["label"] == 0).sum()),
        "positive_pct": float((usable["label"] == 1).mean() * 100),
        "negative_pct": float((usable["label"] == 0).mean() * 100),
        "nan_pct": float((len(df) - len(usable)) / len(df) * 100),
        "train_n": len(train), "holdout_n": len(holdout),
        "roc_auc": float(roc_auc_score(y_holdout, p_holdout)),
        "pr_auc": float(average_precision_score(y_holdout, p_holdout)),
        "accuracy": float((y_pred == y_holdout).mean()),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_holdout, y_pred)),
        "brier": float(brier_score_loss(y_holdout, p_holdout)),
    }
    log(f"  usable={metrics['usable_rows']:,} train={metrics['train_n']:,} holdout={metrics['holdout_n']:,} "
        f"AUC={metrics['roc_auc']:.4f} Brier={metrics['brier']:.4f}")

    holdout_out = holdout.copy()
    holdout_out["p_pred"] = p_holdout
    holdout_out["y_pred"] = y_pred
    return metrics, holdout_out, df


def bootstrap_auc_ci(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys, ps = y[idx], p[idx]
        if len(set(ys)) > 1:
            aucs.append(roc_auc_score(ys, ps))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), aucs


def regime_breakdown(holdout_out, label_prefix):
    rows = []
    # by trend_sign (direction)
    for sign, name in [(1, "Uptrend"), (-1, "Downtrend")]:
        sub = holdout_out[holdout_out["trend_sign"] == sign]
        if len(sub) > 50 and sub["label"].nunique() > 1:
            auc = roc_auc_score(sub["label"].astype(int), sub["p_pred"])
            hit = float((sub["y_pred"] == sub["label"]).mean())
            rows.append({"condition": name, "n": len(sub), "auc": auc, "hit_rate": hit})
    # by state
    for state, sub in holdout_out.groupby("state"):
        if len(sub) > 50 and sub["label"].nunique() > 1:
            auc = roc_auc_score(sub["label"].astype(int), sub["p_pred"])
            hit = float((sub["y_pred"] == sub["label"]).mean())
            rows.append({"condition": f"state_{state}", "n": len(sub), "auc": auc, "hit_rate": hit})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    base_metrics, base_holdout, base_full = run_one("BASELINE (fixed epsilon)", BASELINE_PATH)
    vol_metrics, vol_holdout, vol_full = run_one("VOLATILITY-ADJUSTED", VOLADJ_PATH)

    runtime = time.time() - t0

    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    comp = pd.DataFrame({"Baseline": base_metrics, "Vol-Adjusted": vol_metrics})
    comp["Difference"] = comp["Vol-Adjusted"] - comp["Baseline"]
    print(comp.to_string())

    print("\n=== Bootstrap 95% CI for holdout AUC (2000 resamples each) ===")
    b_lo, b_hi, b_aucs = bootstrap_auc_ci(base_holdout["label"].astype(int).values, base_holdout["p_pred"].values)
    v_lo, v_hi, v_aucs = bootstrap_auc_ci(vol_holdout["label"].astype(int).values, vol_holdout["p_pred"].values)
    print(f"Baseline:      AUC={base_metrics['roc_auc']:.4f}  95% CI=[{b_lo:.4f}, {b_hi:.4f}]")
    print(f"Vol-Adjusted:  AUC={vol_metrics['roc_auc']:.4f}  95% CI=[{v_lo:.4f}, {v_hi:.4f}]")
    overlap = not (v_lo > b_hi or b_lo > v_hi)
    print(f"CIs overlap: {overlap}  (if True, difference is not clearly distinguishable from noise)")

    print("\n=== Regime/volatility breakdown -- BASELINE ===")
    base_regime = regime_breakdown(base_holdout, "baseline")
    print(base_regime.to_string(index=False))
    print("\n=== Regime/volatility breakdown -- VOL-ADJUSTED ===")
    vol_regime = regime_breakdown(vol_holdout, "voladj")
    print(vol_regime.to_string(index=False))

    print("\n=== Vol-regime terciles (using the EXPERIMENTAL table's own past_volatility, holdout only) ===")
    vh = vol_holdout.copy()
    vh["vol_tercile"] = pd.qcut(vh["past_volatility"], 3, labels=["Low", "Medium", "High"])
    for name, sub in vh.groupby("vol_tercile", observed=True):
        if len(sub) > 50 and sub["label"].nunique() > 1:
            auc = roc_auc_score(sub["label"].astype(int), sub["p_pred"])
            print(f"  {name}-vol: n={len(sub):,}  AUC={auc:.4f}  pos_rate={sub['label'].mean():.3f}")

    comp.to_csv(os.path.join(OUT_DIR, "comparison_table.csv"))
    base_regime.to_csv(os.path.join(OUT_DIR, "baseline_regime_breakdown.csv"), index=False)
    vol_regime.to_csv(os.path.join(OUT_DIR, "voladj_regime_breakdown.csv"), index=False)
    json.dump({"baseline": base_metrics, "voladj": vol_metrics,
               "baseline_auc_ci": [b_lo, b_hi], "voladj_auc_ci": [v_lo, v_hi], "ci_overlap": overlap,
               "runtime_s": runtime}, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2, default=str)
    log(f"\nRuntime: {runtime:.1f}s. Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
