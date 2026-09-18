"""
v4d_gate_threshold_calibration.py -- per-fold, training-only Trend Gate
threshold calibration. Pure Stage-B/cache-replay: reads only
Data/v2_cache/trend_threshold_experiment/adaptive_fold_threshold_master_oos.csv
(the already-fixed adaptive HMM-threshold OOS dataset, ±0.30 default / ±0.15
fallback), which already contains the real walk-forward LR's `lr_probability`
per row. No HMM computation, no refitting of the real walk-forward LR --
that model and its probabilities are reused verbatim.

For each fold k (once its pool clears MIN_POOL_ROWS, same bootstrap rule as
everywhere else in this project):
  1. pool = all already-labeled rows from folds < k.
  2. Split pool chronologically 70/30 into inner-train / inner-val.
  3. Fit a CALIBRATION-ONLY Logistic Regression on inner-train (a throwaway
     model, separate from the real walk-forward LR already cached) purely to
     sweep candidate thresholds on inner-val.
  4. For each candidate threshold, compute inner-val F1; select the
     F1-maximizing threshold (ties broken toward 0.50, the least aggressive
     candidate) -- this is the per-fold SELECTION rule. Final production
     recommendation is a separate judgment made afterward from the full
     result table, not automated from this rule alone.
  5. Freeze that threshold. Apply it to fold k's OWN OOS rows using the
     REAL walk-forward LR's already-cached `lr_probability` (never
     re-predicted here) to get fold k's OOS metrics.
  6. Pool expands to include fold k's rows before calibrating fold k+1.

Run: `python v4d_gate_threshold_calibration.py` (from src/). Expected: <2 min.
Writes to Data/v2_cache/trend_threshold_experiment/gate_calibration/.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "adaptive_fold_threshold_master_oos.csv")
OUT_DIR = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "gate_calibration")

CANDIDATES = [0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.50]
FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]
MIN_POOL_ROWS = 500
SPLIT_FRAC = 0.70
FIXED_BASELINE = 0.50


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def metrics_at(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1),
            "gate_pass_rate": float(y_pred.mean()), "n_signals": int(y_pred.sum())}


def select_threshold(pool: pd.DataFrame):
    n = len(pool)
    split = int(n * SPLIT_FRAC)
    inner_train, inner_val = pool.iloc[:split], pool.iloc[split:]
    if len(inner_train) < 50 or inner_train["label"].nunique() < 2 or len(inner_val) < 20 or inner_val["label"].nunique() < 2:
        return None, None, inner_train, inner_val

    lr = LogisticRegression(max_iter=1000)
    lr.fit(inner_train[FEATURE_COLUMNS].values, inner_train["label"].astype(int).values)
    p_val = lr.predict_proba(inner_val[FEATURE_COLUMNS].values)[:, 1]
    y_val = inner_val["label"].astype(int).values

    sweep = {}
    best_thr, best_f1 = FIXED_BASELINE, -1
    for thr in CANDIDATES:
        m = metrics_at(y_val, p_val, thr)
        sweep[thr] = m
        if m["f1"] > best_f1 + 1e-9:  # strict improvement required; ties keep the earlier (lower) threshold... adjust below
            best_f1, best_thr = m["f1"], thr
    # tie-break toward 0.50 (least aggressive) among thresholds achieving the max F1
    max_f1 = max(v["f1"] for v in sweep.values())
    tied = [t for t, v in sweep.items() if abs(v["f1"] - max_f1) < 1e-9]
    best_thr = max(tied)  # largest (closest to 0.50) among tied best

    return best_thr, sweep, inner_train, inner_val


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"Loading cached adaptive-threshold OOS dataset (no HMM computation): {SRC}")
    df = pd.read_csv(SRC, parse_dates=["timestamp"])
    df["log1p_duration"] = df["log1p_duration"]  # already present, just documenting reuse
    all_folds = sorted(df["fold"].unique())
    log(f"  {len(df):,} rows across {len(all_folds)} folds")

    pool_frames = []
    fold_results = []
    audit_rows = []
    per_fold_master = []

    for fold_id in all_folds:
        fold_rows = df[df["fold"] == fold_id]
        fold_rows_labeled_usable = fold_rows[fold_rows["usable"] == True]  # noqa: E712
        pool = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712

        if len(pool_usable) < MIN_POOL_ROWS or len(fold_rows_labeled_usable) == 0 or fold_rows_labeled_usable["lr_probability"].isna().all():
            pool_frames.append(fold_rows)
            continue

        threshold, sweep, inner_train, inner_val = select_threshold(pool_usable)
        if threshold is None:
            pool_frames.append(fold_rows)
            continue

        # leakage check: calibration data strictly precedes this fold's OOS period
        calib_max_ts = inner_val["timestamp"].max()
        oos_min_ts = fold_rows_labeled_usable["timestamp"].min()
        audit_rows.append({"fold": int(fold_id), "calib_max_ts": str(calib_max_ts), "oos_min_ts": str(oos_min_ts),
                            "ok": bool(pd.Timestamp(calib_max_ts) < pd.Timestamp(oos_min_ts))})

        # apply the FROZEN threshold to this fold's OOS rows using the ALREADY-CACHED real-LR probabilities
        oos = fold_rows_labeled_usable.dropna(subset=["lr_probability"])
        y_true = oos["label"].astype(int).values
        y_prob = oos["lr_probability"].values
        m = metrics_at(y_true, y_prob, threshold)
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None

        fold_results.append({
            "fold": int(fold_id), "selected_threshold": threshold,
            "inner_val_metrics_at_selected": sweep[threshold], "inner_val_sweep": sweep,
            "oos_n": len(oos), "oos_n_actual_positives": int(y_true.sum()),
            "oos_precision": m["precision"], "oos_recall": m["recall"], "oos_f1": m["f1"],
            "oos_gate_pass_rate": m["gate_pass_rate"], "oos_n_signals": m["n_signals"], "oos_auc": auc,
        })
        oos = oos.copy()
        oos["calibrated_gate_pass"] = (y_prob >= threshold).astype(int)
        oos["calibrated_threshold_used"] = threshold
        per_fold_master.append(oos)

        pool_frames.append(fold_rows)

    runtime = time.time() - t0
    log(f"Calibrated {len(fold_results)} folds. Runtime: {runtime:.1f}s")

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("inner_val_metrics_at_selected", "inner_val_sweep")} for r in fold_results])
    results_df.to_csv(os.path.join(OUT_DIR, "fold_calibration_results.csv"), index=False)
    master = pd.concat(per_fold_master, ignore_index=True) if per_fold_master else pd.DataFrame()
    master.to_csv(os.path.join(OUT_DIR, "calibrated_master_oos.csv"), index=False)
    json.dump(fold_results, open(os.path.join(OUT_DIR, "fold_results_full.json"), "w"), indent=2, default=str)

    analyze_and_report(results_df, master, audit_rows, runtime, df)


def analyze_and_report(results_df, master, audit_rows, runtime, full_df):
    print()
    print("=" * 70)
    print("GATE THRESHOLD CALIBRATION -- SUMMARY")
    print("=" * 70)

    print("\n1. Threshold selection frequency:")
    freq = results_df["selected_threshold"].value_counts().sort_index()
    print(freq.to_string())

    print("\n2. Pooled OOS metrics (calibrated, using each fold's own frozen threshold):")
    y_true = master["label"].astype(int).values
    y_prob = master["lr_probability"].values
    y_pred = master["calibrated_gate_pass"].values
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    pooled = {"n": len(master), "n_signals": int(y_pred.sum()), "n_actual_positives": int(y_true.sum()),
              "gate_pass_rate": float(y_pred.mean()), "precision": float(prec), "recall": float(rec),
              "f1": float(f1), "auc": float(auc)}
    print(json.dumps(pooled, indent=2))

    print("\n3. Fold-level stability (OOS F1/precision/recall across calibrated folds):")
    stability = results_df[["oos_precision", "oos_recall", "oos_f1", "oos_gate_pass_rate"]].agg(["mean", "median", "std"])
    print(stability.to_string())

    print("\n4. Comparison vs fixed 0.50 gate (on the SAME OOS rows, using cached lr_probability):")
    y_pred_fixed = (master["lr_probability"] >= 0.50).astype(int).values
    prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(y_true, y_pred_fixed, average="binary", zero_division=0)
    fixed = {"n_signals": int(y_pred_fixed.sum()), "gate_pass_rate": float(y_pred_fixed.mean()),
             "precision": float(prec_f), "recall": float(rec_f), "f1": float(f1_f)}
    print("fixed 0.50:      ", json.dumps(fixed))
    print("calibrated (mix):", json.dumps({k: pooled[k] for k in ("n_signals", "gate_pass_rate", "precision", "recall", "f1")}))

    print("\n7. Calibration/reliability (predicted decile vs realized rate, calibrated approach):")
    if master["lr_probability"].nunique() > 10:
        bins = pd.qcut(master["lr_probability"], 10, duplicates="drop")
        rel = master.assign(bin=bins).groupby("bin").agg(n=("label", "size"), mean_pred=("lr_probability", "mean"), realized=("label", "mean"))
        print(rel.to_string())

    print("\n8. Year-by-year OOS performance:")
    yearly = master.copy()
    yearly["year"] = yearly["timestamp"].dt.year
    year_rows = []
    for yr, grp in yearly.groupby("year"):
        if grp["label"].nunique() < 2:
            continue
        yp = grp["calibrated_gate_pass"].values
        yt = grp["label"].astype(int).values
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        year_rows.append({"year": int(yr), "n": len(grp), "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})
    year_df = pd.DataFrame(year_rows)
    print(year_df.to_string(index=False))

    print("\n9. Leakage audit:")
    violations = [a for a in audit_rows if not a["ok"]]
    audit = {"leakage_found": len(violations) > 0, "n_folds_checked": len(audit_rows), "violations": violations}
    print(json.dumps(audit, indent=2))

    json.dump({"pooled": pooled, "fixed_baseline": fixed, "threshold_frequency": freq.to_dict(),
               "stability": stability.to_dict(), "audit": audit, "runtime_s": runtime},
              open(os.path.join(OUT_DIR, "calibration_summary.json"), "w"), indent=2, default=str)
    year_df.to_csv(os.path.join(OUT_DIR, "yearly_performance.csv"), index=False)
    log(f"All artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
