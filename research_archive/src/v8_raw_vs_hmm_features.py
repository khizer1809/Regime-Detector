"""
v8_raw_vs_hmm_features.py -- Step 5: raw market features vs HMM persistence
features. Pure Stage-B/cache-replay: reads the already-cached adaptive-
threshold OOS dataset (timestamps, fold assignment, usable/label -- i.e.
target, walk-forward, embargo, HMM threshold all untouched) and the
already-existing Data/features_out.csv (the real 25-column production
feature matrix, confirmed against features.py's own docstring -- not
rebuilt, not re-derived). No HMM computation, no raw-data refetch.

Model A: existing 4 HMM persistence features (baseline reproduction).
Model B: 25 raw market features only, causally scaled per fold.
Model C: 25 raw + 4 HMM = 29 features, causally scaled per fold.

Scaling: StandardScaler fit on each fold's training pool ONLY, applied to
that pool and to the fold's OOS rows -- never fit on OOS data. Model A's
4 features are used unscaled, exactly as in every prior experiment (they
were never scaled there either, for consistency/reproducibility).

Run: `python v8_raw_vs_hmm_features.py` (from src/). Expected: a few minutes.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                              precision_recall_fscore_support, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHED_OOS = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "adaptive_fold_threshold_master_oos.csv")
RAW_FEATURES_PATH = os.path.join(_ROOT, "Data", "features_out.csv")
OUT_DIR = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "raw_vs_hmm_features")

HMM_FEATURES = ["confidence", "stay_prob", "log1p_duration", "margin"]
MIN_POOL_ROWS = 500
GATE_THRESHOLD = 0.5


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_data():
    log(f"Loading cached OOS dataset (target/walk-forward/HMM threshold already fixed): {CACHED_OOS}")
    cached = pd.read_csv(CACHED_OOS, parse_dates=["timestamp"])
    log(f"  {len(cached):,} rows across {cached['fold'].nunique()} folds")

    log(f"Loading existing production 25-column raw feature matrix: {RAW_FEATURES_PATH}")
    raw = pd.read_csv(RAW_FEATURES_PATH, parse_dates=["timestamp"])
    raw_cols = [c for c in raw.columns if c != "timestamp"]
    log(f"  exact 25 raw columns loaded: {raw_cols}")
    assert len(raw_cols) == 25, f"expected 25 raw columns, found {len(raw_cols)}"

    merged = cached.merge(raw, on="timestamp", how="left", suffixes=("", "_raw"))
    missing = merged[raw_cols].isna().any(axis=1).sum()
    log(f"  merged on timestamp: {len(merged):,} rows, {missing} rows missing raw features (should be 0)")
    return merged, raw_cols


def run_walkforward(df: pd.DataFrame, features: list, scale: bool):
    all_folds = sorted(df["fold"].unique())
    pool_frames = []
    oos_y, oos_p = [], []
    fold_results = []
    audit_rows = []
    coef_history = []

    for fold_id in all_folds:
        fold_rows = df[df["fold"] == fold_id]
        fold_elig = fold_rows[fold_rows["usable"] == True]  # noqa: E712
        pool = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712

        if len(pool_usable) < MIN_POOL_ROWS or len(fold_elig) == 0:
            pool_frames.append(fold_rows)
            continue

        y_train = pool_usable["label"].astype(int).values
        if len(np.unique(y_train)) < 2:
            pool_frames.append(fold_rows)
            continue

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(pool_usable[features].values)  # fit on TRAIN pool only
            X_oos = scaler.transform(fold_elig[features].values)          # transform only
        else:
            X_train = pool_usable[features].values
            X_oos = fold_elig[features].values

        audit_rows.append({"fold": int(fold_id), "train_max_ts": str(pool_usable["timestamp"].max()),
                            "predict_min_ts": str(fold_elig["timestamp"].min()),
                            "scaler_fit_rows": len(pool_usable) if scale else None})

        pos_rate = float(y_train.mean())
        class_weight = "balanced" if not (0.4 <= pos_rate <= 0.6) else None
        model = LogisticRegression(max_iter=2000, class_weight=class_weight)
        model.fit(X_train, y_train)
        coef_history.append(dict(zip(features, model.coef_[0].tolist())))

        train_pred = model.predict_proba(X_train)[:, 1]
        train_auc = float(roc_auc_score(y_train, train_pred)) if len(set(y_train)) > 1 else None

        y_oos = fold_elig["label"].astype(int).values
        p_oos = model.predict_proba(X_oos)[:, 1]
        oos_y.extend(y_oos)
        oos_p.extend(p_oos)

        fold_auc = float(roc_auc_score(y_oos, p_oos)) if len(set(y_oos)) > 1 else None
        fold_results.append({"fold": int(fold_id), "n_train": len(X_train), "n_oos": len(X_oos),
                             "train_auc": train_auc, "oos_auc": fold_auc,
                             "year": fold_elig["timestamp"].dt.year.mode()[0] if len(fold_elig) else None})
        pool_frames.append(fold_rows)

    oos_y = np.array(oos_y); oos_p = np.array(oos_p)
    violations = [r["fold"] for r in audit_rows if pd.Timestamp(r["train_max_ts"]) >= pd.Timestamp(r["predict_min_ts"])]
    scaler_violations = [r["fold"] for r in audit_rows if scale and r["scaler_fit_rows"] is None]
    return oos_y, oos_p, fold_results, {"violations": violations, "n_checked": len(audit_rows),
                                         "scaler_fit_on_train_only": scale, "scaler_violations": scaler_violations}, coef_history


def summarize(oos_y, oos_p, fold_results, features, model_name):
    y_pred = (oos_p >= GATE_THRESHOLD).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(oos_y, y_pred, average="binary", zero_division=0)
    fold_aucs = [r["oos_auc"] for r in fold_results if r["oos_auc"] is not None]
    train_aucs = [r["train_auc"] for r in fold_results if r["train_auc"] is not None]
    s = {
        "model": model_name, "n_features": len(features), "n": len(oos_y),
        "roc_auc": float(roc_auc_score(oos_y, oos_p)), "pr_auc": float(average_precision_score(oos_y, oos_p)),
        "log_loss": float(log_loss(oos_y, oos_p)), "brier": float(brier_score_loss(oos_y, oos_p)),
        "accuracy": float((y_pred == oos_y).mean()), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "fold_auc_median": float(np.median(fold_aucs)) if fold_aucs else None,
        "fold_auc_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "folds_above_0.5": int(sum(1 for a in fold_aucs if a > 0.5)),
        "folds_below_0.5": int(sum(1 for a in fold_aucs if a < 0.5)),
        "n_folds": len(fold_results),
        "mean_train_auc": float(np.mean(train_aucs)) if train_aucs else None,
        "train_oos_gap": (float(np.mean(train_aucs)) - float(np.mean(fold_aucs))) if train_aucs and fold_aucs else None,
    }
    yr = {}
    for r in fold_results:
        if r["oos_auc"] is not None and r["year"] is not None:
            yr.setdefault(int(r["year"]), []).append(r["oos_auc"])
    s["by_year_auc_mean"] = {y: round(float(np.mean(v)), 4) for y, v in sorted(yr.items())}
    return s


def coef_report(coef_history, features):
    df = pd.DataFrame(coef_history)
    summary = df.agg(["mean", "std"]).T.reset_index().rename(columns={"index": "feature"})
    summary = summary.sort_values("mean", key=abs, ascending=False)
    return summary


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df, raw_cols = load_data()

    results = {}
    coef_reports = {}
    audits = {}

    log("\n=== MODEL A: 4 HMM persistence features (baseline reproduction) ===")
    oos_y, oos_p, fold_results, audit, coefs = run_walkforward(df, HMM_FEATURES, scale=False)
    results["A_hmm4"] = summarize(oos_y, oos_p, fold_results, HMM_FEATURES, "A_hmm4")
    audits["A_hmm4"] = audit
    log(f"  A: OOS_AUC={results['A_hmm4']['roc_auc']:.4f} (expected ~0.5228) fold_mean={results['A_hmm4']['fold_auc_mean']:.4f}")

    log("\n=== MODEL B: 25 raw features only (causally scaled) ===")
    oos_y, oos_p, fold_results, audit, coefs = run_walkforward(df, raw_cols, scale=True)
    results["B_raw25"] = summarize(oos_y, oos_p, fold_results, raw_cols, "B_raw25")
    audits["B_raw25"] = audit
    coef_reports["B_raw25"] = coef_report(coefs, raw_cols)
    log(f"  B: OOS_AUC={results['B_raw25']['roc_auc']:.4f} fold_mean={results['B_raw25']['fold_auc_mean']:.4f} "
        f"train_auc={results['B_raw25']['mean_train_auc']:.4f} gap={results['B_raw25']['train_oos_gap']:.4f}")

    log("\n=== MODEL C: 25 raw + 4 HMM = 29 features (causally scaled) ===")
    all29 = raw_cols + HMM_FEATURES
    oos_y, oos_p, fold_results, audit, coefs = run_walkforward(df, all29, scale=True)
    results["C_raw_plus_hmm"] = summarize(oos_y, oos_p, fold_results, all29, "C_raw_plus_hmm")
    audits["C_raw_plus_hmm"] = audit
    coef_reports["C_raw_plus_hmm"] = coef_report(coefs, all29)
    log(f"  C: OOS_AUC={results['C_raw_plus_hmm']['roc_auc']:.4f} fold_mean={results['C_raw_plus_hmm']['fold_auc_mean']:.4f} "
        f"train_auc={results['C_raw_plus_hmm']['mean_train_auc']:.4f} gap={results['C_raw_plus_hmm']['train_oos_gap']:.4f}")

    runtime = time.time() - t0

    deltas = {
        "delta_B_vs_A": results["B_raw25"]["roc_auc"] - results["A_hmm4"]["roc_auc"],
        "delta_C_vs_A": results["C_raw_plus_hmm"]["roc_auc"] - results["A_hmm4"]["roc_auc"],
        "delta_C_vs_B": results["C_raw_plus_hmm"]["roc_auc"] - results["B_raw25"]["roc_auc"],
    }

    print("\n" + "=" * 70)
    print("STEP 5 RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2, default=str))
    print("\nDELTAS:", json.dumps(deltas, indent=2))
    print("\nModel B coefficient report (standardized, sorted by |mean|):")
    print(coef_reports["B_raw25"].to_string(index=False))
    print("\nModel C coefficient report (standardized, sorted by |mean|):")
    print(coef_reports["C_raw_plus_hmm"].to_string(index=False))
    print("\nLeakage audits:", json.dumps(audits, indent=2, default=str))

    json.dump({"results": results, "deltas": deltas, "raw_columns": raw_cols, "runtime_s": runtime},
              open(os.path.join(OUT_DIR, "step5_results.json"), "w"), indent=2, default=str)
    json.dump(audits, open(os.path.join(OUT_DIR, "step5_leakage_audit.json"), "w"), indent=2, default=str)
    coef_reports["B_raw25"].to_csv(os.path.join(OUT_DIR, "model_B_coefficients.csv"), index=False)
    coef_reports["C_raw_plus_hmm"].to_csv(os.path.join(OUT_DIR, "model_C_coefficients.csv"), index=False)
    log(f"\nRuntime: {runtime:.1f}s. Artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
