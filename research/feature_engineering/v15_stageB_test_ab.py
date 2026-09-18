"""
v15_stageB_test_ab.py -- A/B/C experiment, Stage B (cheap, reads Stage A cache).

Reads ONLY the per-fold causal_output.csv + hmm.pkl/metadata.json files
v15_stageA_expanded_hmm.py wrote to
Data/hmm_lr_abc_experiment/expanded_hmm_cache/fold_XXX/. Never re-fits or
re-decodes the HMM.

Applies the SAME frozen trend-classification RULE validated earlier this
session (+/-0.30 default, +/-0.15 fallback for any fold with zero eligible
trending states) -- recomputed here because the rule must be re-applied to
the NEW (33-feature) HMM's own state scores; the OLD HMM's classification
numbers are not transferable to different states. This is the same rule,
not a new one -- see v4c_adaptive_fold_threshold.py, which this mirrors.

Then runs the SAME expanding-pool / training-only-epsilon / fixed-0.5-gate
walk-forward Logistic Regression methodology as v4_threshold_sensitivity.py,
generalized to a parameterized feature-column list:

    Test A: FEATURE_COLUMNS = 4 baseline persistence features only.
    Test B: FEATURE_COLUMNS = 4 baseline + 15 market-structure features
            (joined in from features_expanded_33.csv by timestamp -- the
            SAME structure feature values already used as HMM input, now
            ALSO given directly to the LR).

Run: `python v15_stageB_test_ab.py` (from research_archive/src/).
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
from sklearn.preprocessing import StandardScaler

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import persistence_common  # noqa: E402

CACHE_DIR = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "expanded_hmm_cache")
EXPANDED_FEATURES_PATH = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "features_expanded_33.csv")
OUT_DIR = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment")

PRIMARY_THRESHOLD = 0.30
FALLBACK_THRESHOLD = 0.15
BASE_FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]
STRUCTURE_FEATURE_COLUMNS = [
    "hh_strength", "hl_strength", "lh_strength", "ll_strength",
    "structure_direction", "structure_consistency", "bos_up", "bos_down",
    "choch_up", "choch_down", "pullback_ratio", "impulse_strength",
    "distance_from_last_swing_high", "distance_from_last_swing_low", "trend_structure_age",
]
MIN_POOL_ROWS = 500
GATE_THRESHOLD = 0.5
EPSILON_PERCENTILE = 10


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_fold_state_scores(fold_dirs):
    rows = []
    for d in fold_dirs:
        fold_path = os.path.join(CACHE_DIR, d)
        hmm_path = os.path.join(fold_path, "hmm.pkl")
        meta_path = os.path.join(fold_path, "metadata.json")
        if not (os.path.exists(hmm_path) and os.path.exists(meta_path)):
            continue
        meta = json.load(open(meta_path))
        with open(hmm_path, "rb") as f:
            model = pickle.load(f)
        means_df = pd.DataFrame(model.means_, columns=meta["feature_columns"])
        scores = persistence_common.state_trend_scores(means_df)
        row = {"fold": meta["fold_id"]}
        for s in range(len(scores)):
            row[f"state_{s}_score"] = float(scores[s])
        for i in range(len(scores), 4):
            row[f"state_{i}_score"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def load_all_causal_outputs(fold_dirs):
    frames = []
    for d in fold_dirs:
        path = os.path.join(CACHE_DIR, d, "causal_output.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, parse_dates=["timestamp"]))
    df = pd.concat(frames, ignore_index=True).sort_values(["fold", "timestamp"]).reset_index(drop=True)
    df["log1p_duration"] = persistence_common.duration_feature(df["duration"].values)
    return df


def classify_at_threshold(state_scores_df, threshold):
    out = state_scores_df.copy()
    n_up = np.zeros(len(out), dtype=int)
    n_down = np.zeros(len(out), dtype=int)
    for s in range(4):
        col = f"state_{s}_score"
        valid = out[col].notna()
        n_up += (valid & (out[col] > threshold)).astype(int)
        n_down += (valid & (out[col] < -threshold)).astype(int)
    out["zero_eligible_fold"] = (n_up + n_down) == 0
    return out


def run_walkforward(causal, trend_sign_row, is_trending_row, feature_columns):
    """Mirrors v4_threshold_sensitivity.run_v2_style_walkforward exactly
    (expanding pool, training-only epsilon frozen per fold, fixed 0.5 gate),
    generalized to a parameterized feature-column list, plus a training-pool
    StandardScaler (consistent with v14's harness, for apples-to-apples
    comparability against the Test C number produced by that same harness)."""
    df = causal.copy()
    df["trend_sign_t"] = trend_sign_row
    df["is_trending_t"] = is_trending_row

    folds = sorted(df["fold"].unique())
    pool_frames, master_rows, fold_metrics, audit_rows = [], [], [], []

    for fold_id in folds:
        fold_rows = df[df["fold"] == fold_id].copy()
        pool = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712

        if len(pool_usable) < MIN_POOL_ROWS:
            trending_fwd = fold_rows[(fold_rows["is_trending_t"]) & (fold_rows["forward_valid"])]
            eps = float(np.percentile(trending_fwd["forward_ret_24"].abs(), EPSILON_PERCENTILE)) if len(trending_fwd) > 10 else np.nan
            usable = (fold_rows["is_trending_t"] & fold_rows["forward_valid"] &
                      (fold_rows["forward_ret_24"].abs() > eps)) if not np.isnan(eps) else pd.Series(False, index=fold_rows.index)
            label = np.where(np.sign(fold_rows["forward_ret_24"]) == fold_rows["trend_sign_t"], 1, 0)
            fold_rows["epsilon"] = eps
            fold_rows["usable"] = usable
            fold_rows["label"] = np.where(usable, label, np.nan)
            fold_rows["lr_probability"] = np.nan
            pool_frames.append(fold_rows)
            master_rows.append(fold_rows)
            fold_metrics.append({"fold": int(fold_id), "n_eligible": int(usable.sum()), "pool_train_rows": 0})
            continue

        epsilon = float(np.percentile(pool_usable["forward_ret_24"].abs(), EPSILON_PERCENTILE))
        usable_fold = (fold_rows["is_trending_t"] & fold_rows["forward_valid"] &
                       (fold_rows["forward_ret_24"].abs() > epsilon))
        label_fold = np.where(np.sign(fold_rows["forward_ret_24"]) == fold_rows["trend_sign_t"], 1, 0)
        fold_rows["epsilon"] = epsilon
        fold_rows["usable"] = usable_fold
        fold_rows["label"] = np.where(usable_fold, label_fold, np.nan)

        pool_fit = pool_usable.dropna(subset=feature_columns + ["label"])
        X_train = pool_fit[feature_columns].values
        y_train = pool_fit["label"].astype(int).values
        pos_rate = float(y_train.mean()) if len(y_train) else 0.5
        class_weight = "balanced" if not (0.4 <= pos_rate <= 0.6) else None

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        lr = LogisticRegression(max_iter=1000, class_weight=class_weight)
        lr.fit(X_train_scaled, y_train)

        audit_rows.append({"fold": int(fold_id), "train_max_ts": str(pool_fit["timestamp"].max()) if len(pool_fit) else None,
                            "predict_min_ts": str(fold_rows["timestamp"].min()) if len(fold_rows) else None})

        trending_mask = fold_rows["is_trending_t"] & fold_rows[feature_columns].notna().all(axis=1)
        X_pred = fold_rows.loc[trending_mask, feature_columns].values
        fold_rows["lr_probability"] = np.nan
        if len(X_pred):
            X_pred_scaled = scaler.transform(X_pred)
            probs = lr.predict_proba(X_pred_scaled)[:, 1]
            fold_rows.loc[trending_mask, "lr_probability"] = probs

        eligible = fold_rows[fold_rows["usable"] & fold_rows["lr_probability"].notna()]
        fm = {"fold": int(fold_id), "n_eligible": len(eligible), "pool_train_rows": len(X_train), "epsilon": epsilon}
        if len(eligible) > 5 and eligible["label"].nunique() > 1:
            fm["fold_auc"] = float(roc_auc_score(eligible["label"].astype(int), eligible["lr_probability"]))
        fold_metrics.append(fm)

        pool_frames.append(fold_rows)
        master_rows.append(fold_rows)

    master = pd.concat(master_rows, ignore_index=True)
    return master, fold_metrics, audit_rows


def compute_metrics_block(eligible):
    y_true = eligible["label"].astype(int).values
    y_prob = eligible["lr_probability"].values
    y_pred = (y_prob >= GATE_THRESHOLD).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "n": len(eligible), "positives": int(y_true.sum()), "negatives": int((1 - y_true).sum()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()), "precision": float(prec), "recall": float(rec), "f1": float(f1),
    }


def train_auc_check(master, feature_columns):
    usable = master[master["usable"] == True].dropna(subset=feature_columns + ["label"])  # noqa: E712
    if len(usable) < 50 or usable["label"].nunique() < 2:
        return None
    X = usable[feature_columns].values
    y = usable["label"].astype(int).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cw = "balanced" if not (0.4 <= y.mean() <= 0.6) else None
    lr = LogisticRegression(max_iter=1000, class_weight=cw)
    lr.fit(Xs, y)
    p = lr.predict_proba(Xs)[:, 1]
    return float(roc_auc_score(y, p))


def summarize_run(tag, master, fold_metrics, train_metrics):
    eligible = master[master["usable"] & master["lr_probability"].notna()]
    metrics = compute_metrics_block(eligible) if len(eligible) > 10 and eligible["label"].nunique() > 1 else {}
    fold_aucs = [fm["fold_auc"] for fm in fold_metrics if "fold_auc" in fm]
    return {
        "tag": tag, "oos_roc_auc": metrics.get("roc_auc"), "oos_pr_auc": metrics.get("pr_auc"),
        "n_eligible": len(eligible), "positives": metrics.get("positives"), "negatives": metrics.get("negatives"),
        "fold_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "fold_median": float(np.median(fold_aucs)) if fold_aucs else None,
        "fold_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "folds_above_050": int(sum(1 for a in fold_aucs if a > 0.50)),
        "folds_above_052": int(sum(1 for a in fold_aucs if a > 0.52)),
        "n_computable_folds": len(fold_aucs), "train_metrics": train_metrics,
        "full_metrics": metrics, "fold_metrics": fold_metrics,
    }


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    fold_dirs = sorted(d for d in os.listdir(CACHE_DIR) if d.startswith("fold_")
                        and os.path.exists(os.path.join(CACHE_DIR, d, "DONE")))
    log(f"Found {len(fold_dirs)} completed Stage A folds")

    state_scores = load_fold_state_scores(fold_dirs)
    causal = load_all_causal_outputs(fold_dirs)
    log(f"Loaded {len(causal):,} causal-output rows across {causal['fold'].nunique()} folds")

    classified_primary = classify_at_threshold(state_scores, PRIMARY_THRESHOLD)
    fallback_folds = set(classified_primary[classified_primary["zero_eligible_fold"]]["fold"].tolist())
    log(f"Folds on primary +/-{PRIMARY_THRESHOLD}: {len(classified_primary)-len(fallback_folds)}; "
        f"folds on fallback +/-{FALLBACK_THRESHOLD}: {len(fallback_folds)}")

    fold_threshold = causal["fold"].map(lambda f: FALLBACK_THRESHOLD if f in fallback_folds else PRIMARY_THRESHOLD)
    is_trending_row = causal["trend_score"].abs() > fold_threshold
    trend_sign_row = np.where(is_trending_row, np.sign(causal["trend_score"]), 0.0)

    log("=== TEST A: expanded HMM, baseline 4 LR features ===")
    master_a, fm_a, audit_a = run_walkforward(causal, trend_sign_row, is_trending_row, BASE_FEATURE_COLUMNS)
    train_auc_a = train_auc_check(master_a, BASE_FEATURE_COLUMNS)
    summary_a = summarize_run("A_expanded_hmm_baseline_lr", master_a, fm_a, train_auc_a)
    log(f"  Test A: OOS AUC={summary_a['oos_roc_auc']}")

    log("=== TEST B: expanded HMM, expanded 19 LR features ===")
    expanded_feats = pd.read_csv(EXPANDED_FEATURES_PATH, usecols=["timestamp"] + STRUCTURE_FEATURE_COLUMNS,
                                  parse_dates=["timestamp"])
    causal_b = causal.merge(expanded_feats, on="timestamp", how="left")
    assert len(causal_b) == len(causal), "row count changed on structure-feature merge for Test B"
    feature_cols_b = BASE_FEATURE_COLUMNS + STRUCTURE_FEATURE_COLUMNS
    master_b, fm_b, audit_b = run_walkforward(causal_b, trend_sign_row, is_trending_row, feature_cols_b)
    train_auc_b = train_auc_check(master_b, feature_cols_b)
    summary_b = summarize_run("B_expanded_hmm_expanded_lr", master_b, fm_b, train_auc_b)
    log(f"  Test B: OOS AUC={summary_b['oos_roc_auc']}")

    runtime = time.time() - t0
    log(f"Stage B runtime: {runtime:.1f}s")

    json.dump({
        "test_a": {k: v for k, v in summary_a.items() if k != "fold_metrics"},
        "test_b": {k: v for k, v in summary_b.items() if k != "fold_metrics"},
        "fold_metrics_a": fm_a, "fold_metrics_b": fm_b,
        "fallback_folds": sorted(fallback_folds), "n_fallback_folds": len(fallback_folds),
        "audit_a": audit_a, "audit_b": audit_b, "runtime_s": runtime,
    }, open(os.path.join(OUT_DIR, "stageB_ab_results.json"), "w"), indent=2, default=str)

    master_a.to_csv(os.path.join(OUT_DIR, "master_oos_test_a.csv"), index=False)
    master_b.to_csv(os.path.join(OUT_DIR, "master_oos_test_b.csv"), index=False)
    log(f"Saved Stage B results to {OUT_DIR}")


if __name__ == "__main__":
    main()
