"""
v2_stage_b_lr.py -- V2 Stage B (the cheap, re-runnable part).

Reads ONLY the per-fold causal_output.csv files v2_stage_a_hmm.py already
wrote to Data/v2_cache/fold_XXX/ -- never touches the HMM, never re-decodes
anything. Builds an expanding Logistic Regression training pool exactly
the way a live system would have experienced it: fold k's LR trains only
on rows that were THEMSELVES already-finalized (labeled, epsilon-filtered)
OOS test rows from folds 0..k-1. Nothing is ever retroactively relabeled
once a fold's own frozen epsilon has been applied to it.

Per fold:
  1. epsilon_k = 10th percentile of |forward_ret_24| over the CURRENT POOL
     (folds < k, already-trending, already-forward-valid rows) -- i.e.
     "training data only", frozen before touching fold k's test rows.
  2. Apply epsilon_k to fold k's own is_trending & forward_valid rows to
     decide usable/label for fold k -- this is a per-fold, one-time
     decision, never revisited.
  3. Train LogisticRegression on the pool's usable rows (features:
     confidence, stay_prob, log1p_duration, margin).
  4. Predict P(trend persists) for every is_trending row in fold k
     (whether or not it's "usable" for evaluation -- a live system would
     still want a prediction). Trend Gate = 1 if probability >= 0.5 (fixed,
     not tuned on any OOS data).
  5. Append fold k's now-finalized rows to the pool for fold k+1.

Bootstrap: folds are skipped for LR prediction until the pool holds at
least MIN_POOL_ROWS usable rows (there's no such thing as training data
for fold 0). This is documented, not hidden -- see the final report.

Run: `python v2_stage_b_lr.py` (from src/). Safe to re-run after any
change to this file's LR/threshold/feature logic without re-running Stage A.
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                              precision_recall_fscore_support, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persistence_common

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_ROOT, "Data", "v2_cache")
LR_DIR = os.path.join(CACHE_DIR, "lr_folds")
LOG_PATH = os.path.join(CACHE_DIR, "stage_b_run_log.jsonl")
MASTER_OOS_PATH = os.path.join(CACHE_DIR, "master_oos.csv")
REPORT_PATH = os.path.join(CACHE_DIR, "v2_final_report.md")
AUDIT_PATH = os.path.join(CACHE_DIR, "leakage_audit.json")

FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]
EPSILON_PERCENTILE = 10
MIN_POOL_ROWS = 500  # bootstrap threshold -- documented cold-start decision
GATE_THRESHOLD = 0.5  # fixed, never tuned on OOS data


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def discover_completed_folds() -> list:
    fold_dirs = sorted(d for d in os.listdir(CACHE_DIR) if d.startswith("fold_"))
    completed = []
    for d in fold_dirs:
        fold_path = os.path.join(CACHE_DIR, d)
        if os.path.exists(os.path.join(fold_path, "DONE")):
            fold_id = int(d.split("_")[1])
            completed.append(fold_id)
    return sorted(completed)


def load_fold_output(fold_id: int) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"fold_{fold_id:03d}", "causal_output.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


def finalize_fold_rows(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """One-time, frozen labeling for this fold's own rows -- never revisited later."""
    df = df.copy()
    trending = df["is_trending"] == True  # noqa: E712
    fwd_valid = df["forward_valid"] == True  # noqa: E712
    usable = trending & fwd_valid & (df["forward_ret_24"].abs() > epsilon)
    label = np.where(np.sign(df["forward_ret_24"]) == df["trend_sign"], 1, 0)
    df["epsilon"] = epsilon
    df["usable"] = usable
    df["persistence_label"] = np.where(usable, label, np.nan)
    df["log1p_duration"] = persistence_common.duration_feature(df["duration"].values)
    return df


def main():
    os.makedirs(LR_DIR, exist_ok=True)
    log("=== V2 Stage B starting ===")
    completed = discover_completed_folds()
    log(f"found {len(completed)} completed Stage A folds: {completed[:3]}...{completed[-3:] if len(completed) > 3 else ''}")

    pool_frames = []
    master_rows = []
    fold_metrics = []
    bootstrap_skipped = []
    audit_rows = {"train_before_test": [], "epsilon_from_pool_only": [], "lr_train_before_predict": []}

    for fold_id in completed:
        lr_fold_dir = os.path.join(LR_DIR, f"fold_{fold_id:03d}")
        done_marker = os.path.join(lr_fold_dir, "DONE")
        raw = load_fold_output(fold_id)
        if raw.empty:
            continue

        pool_df = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        pool_usable = pool_df[pool_df.get("usable", pd.Series(dtype=bool)) == True] if not pool_df.empty else pd.DataFrame()  # noqa: E712

        if len(pool_usable) < MIN_POOL_ROWS:
            # bootstrap phase: not enough training-only data yet to fit an LR.
            # Still need a frozen epsilon for THIS fold's own rows so they can
            # later join the pool -- use this fold's own trending&forward-valid
            # rows for epsilon (there is no earlier data at all for fold 0;
            # this is the one unavoidable exception to "epsilon from training
            # data only", and it's isolated to the bootstrap folds, documented
            # explicitly rather than silently treated as the general rule).
            trending_fwd = raw[(raw["is_trending"] == True) & (raw["forward_valid"] == True)]  # noqa: E712
            eps = float(np.percentile(trending_fwd["forward_ret_24"].abs(), EPSILON_PERCENTILE)) if len(trending_fwd) > 10 else np.nan
            finalized = finalize_fold_rows(raw, eps) if not np.isnan(eps) else raw.assign(epsilon=np.nan, usable=False, persistence_label=np.nan, log1p_duration=np.nan)
            finalized["lr_probability"] = np.nan
            finalized["lr_prediction"] = np.nan
            finalized["trend_gate"] = np.nan
            pool_frames.append(finalized)
            master_rows.append(finalized)
            bootstrap_skipped.append(fold_id)
            log(f"fold {fold_id:03d}: bootstrap (pool={len(pool_usable)} < {MIN_POOL_ROWS}) -- "
                f"no LR prediction, epsilon frozen from THIS fold's own rows only (documented exception)")
            continue

        if os.path.exists(done_marker):
            with open(os.path.join(lr_fold_dir, "finalized.csv")) as f:
                pass
            finalized = pd.read_csv(os.path.join(lr_fold_dir, "finalized.csv"), parse_dates=["timestamp"])
            with open(os.path.join(lr_fold_dir, "metrics.json")) as f:
                fold_metrics.append(json.load(f))
            log(f"fold {fold_id:03d}: LR stage already done, reusing cached result")
        else:
            os.makedirs(lr_fold_dir, exist_ok=True)
            epsilon = float(np.percentile(pool_usable["forward_ret_24"].abs(), EPSILON_PERCENTILE))
            audit_rows["epsilon_from_pool_only"].append({"fold": fold_id, "pool_max_ts": str(pool_df["timestamp"].max()),
                                                          "fold_test_min_ts": str(raw["timestamp"].min())})

            X_train = pool_usable[FEATURE_COLUMNS].values
            y_train = pool_usable["persistence_label"].astype(int).values
            pos_rate = float(y_train.mean())
            class_weight = "balanced" if not (0.4 <= pos_rate <= 0.6) else None
            lr = LogisticRegression(max_iter=1000, class_weight=class_weight)
            lr.fit(X_train, y_train)
            with open(os.path.join(lr_fold_dir, "lr_model.pkl"), "wb") as f:
                pickle.dump(lr, f)

            audit_rows["lr_train_before_predict"].append({
                "fold": fold_id, "train_max_ts": str(pool_usable["timestamp"].max()) if len(pool_usable) else None,
                "predict_min_ts": str(raw["timestamp"].min()),
            })

            finalized = finalize_fold_rows(raw, epsilon)
            trending_mask = finalized["is_trending"] == True  # noqa: E712
            X_pred = finalized.loc[trending_mask, FEATURE_COLUMNS].values
            probs = lr.predict_proba(X_pred)[:, 1] if len(X_pred) else np.array([])
            finalized.loc[trending_mask, "lr_probability"] = probs
            finalized.loc[trending_mask, "lr_prediction"] = (probs >= GATE_THRESHOLD).astype(int)
            finalized.loc[trending_mask, "trend_gate"] = (probs >= GATE_THRESHOLD).astype(int)
            finalized.loc[~trending_mask, ["lr_probability", "lr_prediction", "trend_gate"]] = np.nan

            eligible = finalized[finalized["usable"] == True]  # noqa: E712
            fold_metric = {"fold": fold_id, "pool_train_rows": len(X_train), "pool_pos_rate": pos_rate,
                           "epsilon": epsilon, "n_test_eligible": len(eligible)}
            if len(eligible) > 5 and eligible["persistence_label"].nunique() > 1:
                y_true = eligible["persistence_label"].astype(int).values
                y_prob = eligible["lr_probability"].values
                fold_metric["fold_auc"] = float(roc_auc_score(y_true, y_prob))
            fold_metrics.append(fold_metric)
            with open(os.path.join(lr_fold_dir, "metrics.json"), "w") as f:
                json.dump(fold_metric, f, indent=2, default=str)
            finalized.to_csv(os.path.join(lr_fold_dir, "finalized.csv"), index=False)
            open(done_marker, "w").close()
            log(f"fold {fold_id:03d}: LR trained on {len(X_train):,} pool rows, "
                f"predicted {len(eligible):,} eligible OOS rows, epsilon={epsilon:.6f}"
                + (f", fold_auc={fold_metric.get('fold_auc'):.3f}" if "fold_auc" in fold_metric else ""))

        pool_frames.append(finalized)
        master_rows.append(finalized)

    master = pd.concat(master_rows, ignore_index=True) if master_rows else pd.DataFrame()
    master.to_csv(MASTER_OOS_PATH, index=False)
    log(f"wrote master OOS dataset: {len(master):,} rows -> {MASTER_OOS_PATH}")

    build_final_report(master, fold_metrics, bootstrap_skipped, completed, audit_rows)
    log("=== V2 Stage B complete ===")


def compute_calibration(y_true, y_prob, n_bins=10):
    bins = pd.qcut(y_prob, n_bins, duplicates="drop")
    t = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bin": bins}).groupby("bin").agg(
        n=("y_true", "size"), mean_pred=("y_prob", "mean"), realized_rate=("y_true", "mean"))
    return t


def build_final_report(master, fold_metrics, bootstrap_skipped, completed, audit_rows):
    eligible = master[master["usable"] == True].copy()  # noqa: E712
    eligible = eligible.dropna(subset=["lr_probability", "persistence_label"])

    audit = {"leakage_found": False, "issues": [], "checks": {}}

    # Check 1: HMM train_end < test_start - embargo, for every fold (from Stage A metadata)
    train_before_test_violations = []
    for fold_id in completed:
        meta_path = os.path.join(CACHE_DIR, f"fold_{fold_id:03d}", "metadata.json")
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path))
        if "train_end" in meta and "test_start" in meta:
            if pd.Timestamp(meta["train_end"]) >= pd.Timestamp(meta["test_start"]):
                train_before_test_violations.append(fold_id)
    audit["checks"]["hmm_train_before_test"] = {"violations": train_before_test_violations, "n_checked": len(completed)}
    if train_before_test_violations:
        audit["leakage_found"] = True
        audit["issues"].append(f"HMM train_end >= test_start for folds {train_before_test_violations}")

    # Check 2: epsilon computed from pool only (pool's max timestamp < fold's test min timestamp)
    epsilon_violations = [r["fold"] for r in audit_rows["epsilon_from_pool_only"]
                          if r["pool_max_ts"] != "NaT" and pd.Timestamp(r["pool_max_ts"]) >= pd.Timestamp(r["fold_test_min_ts"])]
    audit["checks"]["epsilon_from_training_only"] = {"violations": epsilon_violations, "n_checked": len(audit_rows["epsilon_from_pool_only"])}
    if epsilon_violations:
        audit["leakage_found"] = True
        audit["issues"].append(f"epsilon pool overlapped fold test window for folds {epsilon_violations}")

    # Check 3: LR trained only on rows with timestamp before the fold's own test start
    lr_violations = [r["fold"] for r in audit_rows["lr_train_before_predict"]
                     if r["train_max_ts"] and pd.Timestamp(r["train_max_ts"]) >= pd.Timestamp(r["predict_min_ts"])]
    audit["checks"]["lr_train_before_predict"] = {"violations": lr_violations, "n_checked": len(audit_rows["lr_train_before_predict"])}
    if lr_violations:
        audit["leakage_found"] = True
        audit["issues"].append(f"LR training pool overlapped prediction fold for folds {lr_violations}")

    # Check 4: every usable row's forward_ret_24 came from within its own gap-aware segment (already
    # enforced structurally by forward_valid in Stage A; spot-check here that no usable row has forward_valid=False)
    bad_usable = master[(master["usable"] == True) & (master["forward_valid"] == False)]  # noqa: E712
    audit["checks"]["usable_implies_forward_valid"] = {"violations": int(len(bad_usable))}
    if len(bad_usable):
        audit["leakage_found"] = True
        audit["issues"].append(f"{len(bad_usable)} rows marked usable without a valid forward window")

    json.dump(audit, open(AUDIT_PATH, "w"), indent=2, default=str)

    lines = []
    lines.append("# V2 Leakage-Safe Walk-Forward -- Final Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"## 1-3. Run status\nCompleted Stage A folds: {len(completed)}. "
                 f"Bootstrap-skipped (no LR, pool < {MIN_POOL_ROWS}): {len(bootstrap_skipped)} -> {bootstrap_skipped}\n")
    lines.append(f"## 6. Leakage audit\nLeakage found: **{audit['leakage_found']}**\n")
    for check, result in audit["checks"].items():
        lines.append(f"- `{check}`: {result}\n")
    if audit["issues"]:
        lines.append("### Issues\n" + "\n".join(f"- {i}" for i in audit["issues"]) + "\n")

    lines.append(f"## 8-9. Persistence dataset / LR performance\n")
    lines.append(f"Total master OOS rows: {len(master):,}. Eligible (usable, labeled) OOS rows: {len(eligible):,}.\n")
    if len(eligible) > 10 and eligible["persistence_label"].nunique() > 1:
        y_true = eligible["persistence_label"].astype(int).values
        y_prob = eligible["lr_probability"].values
        y_pred = (y_prob >= GATE_THRESHOLD).astype(int)
        auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ll = log_loss(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        acc = float((y_pred == y_true).mean())
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        lines.append(f"```\nn={len(eligible):,}  positives={int(y_true.sum()):,}  negatives={int((1-y_true).sum()):,}\n"
                     f"ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  LogLoss={ll:.4f}  Brier={brier:.4f}\n"
                     f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}\n```\n")
        calib = compute_calibration(y_true, y_prob)
        lines.append("### Calibration\n```\n" + calib.to_string() + "\n```\n")

        # trend-sign-alone baseline: label==1 whenever trend_sign matched the eventual outcome
        # is definitionally the target itself, so the fair "trend alone" baseline is the trivial
        # always-predict-majority-class rate, reported for comparison.
        majority_rate = max(y_true.mean(), 1 - y_true.mean())
        lines.append(f"### Baseline comparison\nTrend-sign-alone (always predict majority class): accuracy={majority_rate:.4f}\n"
                     f"HMM + persistence LR probability: accuracy={acc:.4f}, AUC={auc:.4f}\n"
                     f"HMM + Trend Gate (threshold={GATE_THRESHOLD}): same predictions as above (gate = LR>={GATE_THRESHOLD})\n")
    else:
        lines.append("Not enough eligible labeled OOS rows yet to compute aggregate metrics "
                     "(Stage A likely still running -- rerun this script once more folds complete).\n")

    lines.append("## 12-13. Saved artifacts\n")
    lines.append(f"- Per-fold HMM+scaler+causal output: `Data/v2_cache/fold_XXX/`\n")
    lines.append(f"- Per-fold LR artifacts: `Data/v2_cache/lr_folds/fold_XXX/`\n")
    lines.append(f"- Master OOS dataset: `{MASTER_OOS_PATH}`\n")
    lines.append(f"- Leakage audit: `{AUDIT_PATH}`\n")
    lines.append("\n**Future downstream experiments can reuse the saved causal HMM outputs "
                 "without refitting the HMM, provided the HMM layer itself is unchanged.**\n")

    with open(REPORT_PATH, "w") as f:
        f.writelines(lines)
    log(f"wrote final report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
