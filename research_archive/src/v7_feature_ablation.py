"""
v7_feature_ablation.py -- Step 4: feature ablation (Part A) + targeted
nonlinear test (Part B) for the Persistence LR. Pure Stage-B/cache-replay:
reads only the already-cached adaptive-threshold OOS dataset. No HMM
computation, no target/label/walk-forward/embargo changes anywhere.

Run: `python v7_feature_ablation.py` (from src/). Expected: a few minutes.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                              precision_recall_fscore_support, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "adaptive_fold_threshold_master_oos.csv")
OUT_DIR = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "feature_ablation")

ALL_FEATURES = ["confidence", "stay_prob", "log1p_duration", "margin"]
MIN_POOL_ROWS = 500
GATE_THRESHOLD = 0.5

XGB_PARAMS = dict(max_depth=3, n_estimators=200, learning_rate=0.05, min_child_weight=50,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="auc", n_jobs=4, verbosity=0)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_walkforward(df: pd.DataFrame, features: list, model_kind: str = "lr"):
    """Same expanding-pool, train-strictly-precedes-OOS walk-forward used
    throughout this project, parameterized by feature subset and model type."""
    all_folds = sorted(df["fold"].unique())
    pool_frames = []
    oos_y, oos_p = [], []
    fold_results = []
    audit_rows = []

    for fold_id in all_folds:
        fold_rows = df[df["fold"] == fold_id]
        fold_elig = fold_rows[fold_rows["usable"] == True]  # noqa: E712
        pool = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712

        if len(pool_usable) < MIN_POOL_ROWS or len(fold_elig) == 0:
            pool_frames.append(fold_rows)
            continue

        X_train = pool_usable[features].values
        y_train = pool_usable["label"].astype(int).values
        if len(np.unique(y_train)) < 2:
            pool_frames.append(fold_rows)
            continue

        if model_kind == "lr":
            pos_rate = float(y_train.mean())
            class_weight = "balanced" if not (0.4 <= pos_rate <= 0.6) else None
            model = LogisticRegression(max_iter=1000, class_weight=class_weight)
        else:
            model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)

        audit_rows.append({"fold": int(fold_id), "train_max_ts": str(pool_usable["timestamp"].max()),
                            "predict_min_ts": str(fold_elig["timestamp"].min())})

        train_pred = model.predict_proba(X_train)[:, 1]
        train_auc = float(roc_auc_score(y_train, train_pred)) if len(set(y_train)) > 1 else None

        X_oos = fold_elig[features].values
        y_oos = fold_elig["label"].astype(int).values
        p_oos = model.predict_proba(X_oos)[:, 1]
        oos_y.extend(y_oos)
        oos_p.extend(p_oos)

        fold_auc = float(roc_auc_score(y_oos, p_oos)) if len(set(y_oos)) > 1 else None
        fold_results.append({"fold": int(fold_id), "n_train": len(X_train), "n_oos": len(X_oos),
                             "train_auc": train_auc, "oos_auc": fold_auc})
        pool_frames.append(fold_rows)

    oos_y = np.array(oos_y); oos_p = np.array(oos_p)
    violations = [r["fold"] for r in audit_rows if pd.Timestamp(r["train_max_ts"]) >= pd.Timestamp(r["predict_min_ts"])]
    return oos_y, oos_p, fold_results, {"violations": violations, "n_checked": len(audit_rows)}


def summarize(oos_y, oos_p, fold_results, features, model_kind):
    y_pred = (oos_p >= GATE_THRESHOLD).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(oos_y, y_pred, average="binary", zero_division=0)
    fold_aucs = [r["oos_auc"] for r in fold_results if r["oos_auc"] is not None]
    train_aucs = [r["train_auc"] for r in fold_results if r["train_auc"] is not None]
    return {
        "features": features, "model": model_kind, "n": len(oos_y),
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


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"Loading cached adaptive-threshold OOS dataset (no HMM computation): {SRC}")
    df = pd.read_csv(SRC, parse_dates=["timestamp"])
    log(f"  {len(df):,} rows across {df['fold'].nunique()} folds")

    # ============ PART A: ABLATION ============
    log("\n=== PART A: FEATURE ABLATION ===")
    ablation_specs = {
        "A_full": ALL_FEATURES,
        "B_no_confidence": [f for f in ALL_FEATURES if f != "confidence"],
        "C_no_stay_prob": [f for f in ALL_FEATURES if f != "stay_prob"],
        "D_no_log1p_duration": [f for f in ALL_FEATURES if f != "log1p_duration"],
        "E_no_margin": [f for f in ALL_FEATURES if f != "margin"],
    }
    ablation_results = {}
    audits = {}
    for name, feats in ablation_specs.items():
        oos_y, oos_p, fold_results, audit = run_walkforward(df, feats, "lr")
        summary = summarize(oos_y, oos_p, fold_results, feats, "lr")
        ablation_results[name] = summary
        audits[name] = audit
        log(f"  {name} ({feats}): OOS_AUC={summary['roc_auc']:.4f} fold_mean={summary['fold_auc_mean']:.4f} "
            f"fold_std={summary['fold_auc_std']:.4f} folds>0.5={summary['folds_above_0.5']}/{summary['n_folds']}")

    full = ablation_results["A_full"]
    delta_table = []
    for name in ["B_no_confidence", "C_no_stay_prob", "D_no_log1p_duration", "E_no_margin"]:
        r = ablation_results[name]
        removed = (set(ALL_FEATURES) - set(r["features"])).pop()
        delta_auc = full["roc_auc"] - r["roc_auc"]
        delta_fold_mean = full["fold_auc_mean"] - r["fold_auc_mean"]
        delta_folds_above = full["folds_above_0.5"] - r["folds_above_0.5"]
        delta_table.append({"removed_feature": removed, "delta_pooled_auc": delta_auc,
                            "delta_fold_mean_auc": delta_fold_mean, "delta_folds_above_0.5": delta_folds_above,
                            "ablated_pooled_auc": r["roc_auc"], "ablated_fold_mean_auc": r["fold_auc_mean"]})
    delta_df = pd.DataFrame(delta_table)
    print("\n--- DELTA TABLE (Full - Ablated) ---")
    print(delta_df.to_string(index=False))

    pd.DataFrame(ablation_results.values()).to_csv(os.path.join(OUT_DIR, "part_a_ablation_results.csv"), index=False)
    delta_df.to_csv(os.path.join(OUT_DIR, "part_a_delta_table.csv"), index=False)
    json.dump({"results": ablation_results, "audits": audits}, open(os.path.join(OUT_DIR, "part_a_full.json"), "w"), indent=2, default=str)

    runtime_a = time.time() - t0
    log(f"\nPart A runtime: {runtime_a:.1f}s")

    return df, ablation_results, delta_df


if __name__ == "__main__":
    df, ablation_results, delta_df = main()
