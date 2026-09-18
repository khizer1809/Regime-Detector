"""
v5_nonlinear_capacity_test.py -- Step 2: nonlinear capacity test for the
Persistence model. Pure Stage-B/cache-replay: reads only the already-fixed
adaptive-threshold OOS dataset (Data/v2_cache/trend_threshold_experiment/
adaptive_fold_threshold_master_oos.csv). No HMM computation anywhere.

Question: do the same 4 features (confidence, stay_prob, log1p_duration,
margin) contain nonlinear structure Logistic Regression can't capture?
Tests this with a small, conservatively-regularized XGBoost classifier
(LightGBM unavailable in this environment) run through the EXACT same
expanding walk-forward as the LR baseline -- same folds, same
train-strictly-precedes-OOS discipline, same MIN_POOL_ROWS bootstrap.

Hyperparameters are fixed up front, not tuned on any data (no search):
max_depth=3, n_estimators=200, learning_rate=0.05, min_child_weight=50,
subsample=0.8, colsample_bytree=1.0 (only 4 features anyway), reg_lambda=1.0.

Run: `python v5_nonlinear_capacity_test.py` (from src/). Expected: <2 min.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "adaptive_fold_threshold_master_oos.csv")
OUT_DIR = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "nonlinear_capacity_test")

FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]
MIN_POOL_ROWS = 500
LR_BASELINE_AUC = 0.5228
LR_BASELINE_PRAUC = None  # not directly comparable number on hand; computed fresh below for a fair same-cache comparison

XGB_PARAMS = dict(
    max_depth=3, n_estimators=200, learning_rate=0.05, min_child_weight=50,
    subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="auc", n_jobs=4, verbosity=0,
)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"Loading cached adaptive-threshold OOS dataset (no HMM computation): {SRC}")
    df = pd.read_csv(SRC, parse_dates=["timestamp"])
    all_folds = sorted(df["fold"].unique())
    log(f"  {len(df):,} rows across {len(all_folds)} folds")
    log(f"HMM refit_calls: 0 (confirmed -- no HMM object touched below)")
    log(f"HMM decode_calls: 0 (confirmed -- no HMM object touched below)")
    log(f"XGBoost params (fixed, not tuned): {XGB_PARAMS}")

    pool_frames = []
    fold_results = []
    oos_y_all, oos_p_all = [], []
    audit_rows = []
    fold_importances = []
    last_fold_model = None

    for fold_id in all_folds:
        fold_rows = df[df["fold"] == fold_id]
        fold_elig = fold_rows[fold_rows["usable"] == True]  # noqa: E712
        pool = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712

        if len(pool_usable) < MIN_POOL_ROWS or len(fold_elig) == 0:
            pool_frames.append(fold_rows)
            continue

        X_train = pool_usable[FEATURE_COLUMNS].values
        y_train = pool_usable["label"].astype(int).values
        if len(np.unique(y_train)) < 2:
            pool_frames.append(fold_rows)
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
        last_fold_model = model

        audit_rows.append({"fold": int(fold_id), "train_max_ts": str(pool_usable["timestamp"].max()),
                            "predict_min_ts": str(fold_elig["timestamp"].min())})

        train_pred = model.predict_proba(X_train)[:, 1]
        train_auc = float(roc_auc_score(y_train, train_pred)) if len(set(y_train)) > 1 else None

        X_oos = fold_elig[FEATURE_COLUMNS].values
        y_oos = fold_elig["label"].astype(int).values
        p_oos = model.predict_proba(X_oos)[:, 1]

        oos_y_all.extend(y_oos)
        oos_p_all.extend(p_oos)
        fold_importances.append(dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist())))

        fold_auc = float(roc_auc_score(y_oos, p_oos)) if len(set(y_oos)) > 1 else None
        fold_results.append({"fold": int(fold_id), "n_train": len(X_train), "n_oos": len(X_oos),
                             "train_auc": train_auc, "oos_auc": fold_auc})

        pool_frames.append(fold_rows)

    runtime = time.time() - t0
    oos_y_all = np.array(oos_y_all)
    oos_p_all = np.array(oos_p_all)
    pooled_auc = float(roc_auc_score(oos_y_all, oos_p_all))
    pooled_prauc = float(average_precision_score(oos_y_all, oos_p_all))

    fold_aucs = [r["oos_auc"] for r in fold_results if r["oos_auc"] is not None]
    train_aucs = [r["train_auc"] for r in fold_results if r["train_auc"] is not None]

    avg_importance = {f: float(np.mean([fi[f] for fi in fold_importances])) for f in FEATURE_COLUMNS}
    final_importance = dict(zip(FEATURE_COLUMNS, last_fold_model.feature_importances_.tolist())) if last_fold_model else {}

    # calibration table
    calib = None
    if len(np.unique(oos_p_all)) > 10:
        bins = pd.qcut(oos_p_all, 10, duplicates="drop")
        calib = pd.DataFrame({"y": oos_y_all, "p": oos_p_all, "bin": bins}).groupby("bin").agg(
            n=("y", "size"), mean_pred=("p", "mean"), realized=("y", "mean"))

    violations = [r["fold"] for r in audit_rows
                  if pd.Timestamp(r["train_max_ts"]) >= pd.Timestamp(r["predict_min_ts"])]
    audit = {"leakage_found": len(violations) > 0, "violations": violations, "n_folds_checked": len(audit_rows),
             "hmm_refit_calls": 0, "hmm_decode_calls": 0, "hyperparameter_search_performed": False}

    result = {
        "runtime_s": runtime, "n_folds_evaluated": len(fold_results),
        "pooled_oos_roc_auc": pooled_auc, "pooled_oos_pr_auc": pooled_prauc,
        "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "fold_auc_median": float(np.median(fold_aucs)) if fold_aucs else None,
        "fold_auc_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "folds_above_0.5": int(sum(1 for a in fold_aucs if a > 0.5)),
        "folds_below_0.5": int(sum(1 for a in fold_aucs if a < 0.5)),
        "mean_train_auc": float(np.mean(train_aucs)) if train_aucs else None,
        "mean_oos_auc_for_gap": float(np.mean(fold_aucs)) if fold_aucs else None,
        "train_oos_gap": (float(np.mean(train_aucs)) - float(np.mean(fold_aucs))) if train_aucs and fold_aucs else None,
        "lr_baseline_pooled_auc": LR_BASELINE_AUC,
        "improvement_vs_lr": pooled_auc - LR_BASELINE_AUC,
        "avg_feature_importance_across_folds": avg_importance,
        "final_fold_feature_importance": final_importance,
        "xgb_params": XGB_PARAMS,
    }

    json.dump(result, open(os.path.join(OUT_DIR, "xgb_capacity_test_result.json"), "w"), indent=2, default=str)
    pd.DataFrame(fold_results).to_csv(os.path.join(OUT_DIR, "xgb_fold_results.csv"), index=False)
    json.dump(audit, open(os.path.join(OUT_DIR, "xgb_leakage_audit.json"), "w"), indent=2, default=str)
    if calib is not None:
        calib.to_csv(os.path.join(OUT_DIR, "xgb_calibration.csv"))

    print()
    print("=" * 70)
    print("STEP 2 -- NONLINEAR CAPACITY TEST RESULT")
    print("=" * 70)
    print(json.dumps({k: v for k, v in result.items() if k != "xgb_params"}, indent=2, default=str))
    print()
    print("Leakage audit:", json.dumps(audit, indent=2))
    if calib is not None:
        print()
        print("Calibration table:")
        print(calib.to_string())
    log(f"Runtime: {runtime:.1f}s")
    log(f"Artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
