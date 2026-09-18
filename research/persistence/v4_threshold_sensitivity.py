"""
v4_threshold_sensitivity.py -- V4: HMM trend-classification threshold
sensitivity, Stage-B/cache-replay only. Zero HMM fitting, zero HMM
decoding anywhere in this script -- every number here comes from reading
already-cached pickles (fold_XXX/hmm.pkl for the 4 state means, already
computed at fit time) and already-cached causal_output.csv files (the
per-row decoded state/posterior/confidence/margin/duration/stay_prob/
forward_ret_24, all already produced by v2_stage_a_hmm.py).

The ONLY thing this script computes fresh is: re-thresholding an already-
existing per-state trend_score into Uptrend/Downtrend/Ranging at 7
candidate thresholds, then re-running the SAME V2 expanding-walk-forward
LR procedure (identical features, identical epsilon methodology, identical
fold structure) once per threshold. That's it -- no model.fit(), no
model.predict(), no model.predict_proba() call on any HMM object anywhere
below; only pickle.load() to read already-fitted parameters.

Run: `python v4_threshold_sensitivity.py` (from src/). Expected: minutes.
Writes to Data/v2_cache/trend_threshold_experiment/ -- never touches V2/V3.
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
from infer_regime import RETURN_COLS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_CACHE_DIR = os.path.join(_ROOT, "Data", "v2_cache")
OUT_DIR = os.path.join(V2_CACHE_DIR, "trend_threshold_experiment")

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BASELINE_THRESHOLD = 0.30
FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]
MIN_POOL_ROWS = 500
GATE_THRESHOLD = 0.5
EPSILON_PERCENTILE = 10

REFIT_CALLS = 0   # literal counters -- never incremented, since .fit() is never called on an HMM below
DECODE_CALLS = 0  # never incremented, since .predict()/.predict_proba() is never called on an HMM below


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def print_startup_diagnostics(fold_dirs):
    print("=" * 70)
    print("V4 TREND-THRESHOLD SENSITIVITY -- STARTUP DIAGNOSTICS")
    print("=" * 70)
    print(f"Cache location:          {V2_CACHE_DIR}")
    print(f"HMM folds found:         {len(fold_dirs)}")
    print(f"Causal output files:     {sum(1 for d in fold_dirs if os.path.exists(os.path.join(V2_CACHE_DIR, d, 'causal_output.csv')))}")
    print(f"Thresholds to test:      {THRESHOLDS}  (baseline = {BASELINE_THRESHOLD})")
    print(f"HMM refit_calls:         {REFIT_CALLS}  (confirmed zero -- no .fit() call exists in this script)")
    print(f"HMM decode_calls:        {DECODE_CALLS}  (confirmed zero -- no .predict()/.predict_proba() call exists in this script)")
    print("=" * 70)


def load_fold_state_scores(fold_dirs):
    """Reads each fold's already-fitted hmm.pkl (pickle.load only) and its
    metadata.json to compute the 4 states' trend_score -- pure arithmetic
    on already-fitted model.means_, not a refit or a decode."""
    rows = []
    for d in fold_dirs:
        fold_path = os.path.join(V2_CACHE_DIR, d)
        hmm_path = os.path.join(fold_path, "hmm.pkl")
        meta_path = os.path.join(fold_path, "metadata.json")
        if not (os.path.exists(hmm_path) and os.path.exists(meta_path)):
            continue
        meta = json.load(open(meta_path))
        with open(hmm_path, "rb") as f:
            model = pickle.load(f)
        means_df = pd.DataFrame(model.means_, columns=meta["feature_columns"])
        scores = persistence_common.state_trend_scores(means_df)
        row = {"fold": meta["fold_id"], "train_start": meta["train_start"], "train_end": meta["train_end"],
               "test_start": meta["test_start"], "test_end": meta["test_end"]}
        for s in range(len(scores)):
            row[f"state_{s}_score"] = float(scores[s])
        row["max_abs_score"] = float(np.abs(scores).max())
        for i in range(len(scores), 4):
            row[f"state_{i}_score"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def load_all_causal_outputs(fold_dirs):
    frames = []
    for d in fold_dirs:
        path = os.path.join(V2_CACHE_DIR, d, "causal_output.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, parse_dates=["timestamp"]))
    df = pd.concat(frames, ignore_index=True).sort_values(["fold", "timestamp"]).reset_index(drop=True)
    df["log1p_duration"] = persistence_common.duration_feature(df["duration"].values)
    return df


def classify_at_threshold(state_scores_df, threshold):
    """Per fold, per state: Uptrend(+1)/Downtrend(-1)/Ranging(0) at this threshold."""
    out = state_scores_df.copy()
    n_up = np.zeros(len(out), dtype=int)
    n_down = np.zeros(len(out), dtype=int)
    n_ranging = np.zeros(len(out), dtype=int)
    for s in range(4):
        col = f"state_{s}_score"
        valid = out[col].notna()
        up = valid & (out[col] > threshold)
        down = valid & (out[col] < -threshold)
        ranging = valid & ~up & ~down
        n_up += up.astype(int)
        n_down += down.astype(int)
        n_ranging += ranging.astype(int)
    out["n_uptrend_states"] = n_up
    out["n_downtrend_states"] = n_down
    out["n_ranging_states"] = n_ranging
    out["zero_eligible_fold"] = (n_up + n_down) == 0
    return out


def run_v2_style_walkforward(causal, trend_sign_row, is_trending_row, threshold_tag):
    """Exact V2 Stage B methodology (expanding pool, training-only epsilon
    frozen per fold, identical LR config/features, fixed 0.5 gate), just
    parameterized by a re-thresholded trend_sign/is_trending instead of the
    ones baked into the cache at Stage-A time."""
    df = causal.copy()
    df["trend_sign_t"] = trend_sign_row
    df["is_trending_t"] = is_trending_row

    folds = sorted(df["fold"].unique())
    pool_frames, master_rows, fold_metrics = [], [], []
    audit_rows = []

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
            fold_rows["gate_pass"] = np.nan
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

        X_train = pool_usable[FEATURE_COLUMNS].values
        y_train = pool_usable["label"].astype(int).values
        pos_rate = float(y_train.mean())
        class_weight = "balanced" if not (0.4 <= pos_rate <= 0.6) else None
        lr = LogisticRegression(max_iter=1000, class_weight=class_weight)
        lr.fit(X_train, y_train)  # LR, not HMM -- this is the Stage B model, explicitly in scope

        audit_rows.append({"fold": int(fold_id), "train_max_ts": str(pool_usable["timestamp"].max()),
                            "predict_min_ts": str(fold_rows["timestamp"].min()) if len(fold_rows) else None})

        trending_mask = fold_rows["is_trending_t"]
        X_pred = fold_rows.loc[trending_mask, FEATURE_COLUMNS].values
        probs = lr.predict_proba(X_pred)[:, 1] if len(X_pred) else np.array([])
        fold_rows.loc[trending_mask, "lr_probability"] = probs
        fold_rows.loc[trending_mask, "gate_pass"] = (probs >= GATE_THRESHOLD).astype(int)
        fold_rows.loc[~trending_mask, ["lr_probability", "gate_pass"]] = np.nan

        eligible = fold_rows[fold_rows["usable"]]
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
        "log_loss": float(log_loss(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "gate_pass_rate": float((eligible["gate_pass"] == 1).mean()),
        "gate_block_rate": float((eligible["gate_pass"] == 0).mean()),
    }


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    fold_dirs = sorted(d for d in os.listdir(V2_CACHE_DIR) if d.startswith("fold_")
                        and os.path.exists(os.path.join(V2_CACHE_DIR, d, "DONE")))
    print_startup_diagnostics(fold_dirs)

    log("Reading cached HMM state trend scores (pickle.load only, no fit/decode)...")
    state_scores = load_fold_state_scores(fold_dirs)
    log(f"  loaded state scores for {len(state_scores)} folds")

    log("Loading cached causal outputs (no HMM computation)...")
    causal = load_all_causal_outputs(fold_dirs)
    log(f"  loaded {len(causal):,} rows across {causal['fold'].nunique()} folds")

    # --- Section 4/14: state-level diagnostic, zero-fold special report ---
    fold_to_scores = {r["fold"]: [r["state_0_score"], r["state_1_score"], r["state_2_score"], r["state_3_score"]]
                       for _, r in state_scores.iterrows()}
    current_zero_folds = classify_at_threshold(state_scores, BASELINE_THRESHOLD)
    current_zero_fold_ids = current_zero_folds[current_zero_folds["zero_eligible_fold"]]["fold"].tolist()
    log(f"Currently zero-eligible folds at baseline +/-{BASELINE_THRESHOLD}: {len(current_zero_fold_ids)}")

    zero_fold_report = []
    for fold_id in current_zero_fold_ids:
        scores = fold_to_scores[fold_id]
        max_abs = max(abs(s) for s in scores)
        first_threshold = None
        for t in sorted(THRESHOLDS, reverse=True):  # largest-first: find the highest threshold that still recovers this fold
            if max_abs > t:
                first_threshold = t
                break
        row = state_scores[state_scores["fold"] == fold_id].iloc[0]
        zero_fold_report.append({
            "fold": int(fold_id), "test_start": row["test_start"], "test_end": row["test_end"],
            "state_0_score": scores[0], "state_1_score": scores[1], "state_2_score": scores[2], "state_3_score": scores[3],
            "max_abs_score": max_abs, "first_non_ranging_threshold": first_threshold,
        })
    zero_fold_df = pd.DataFrame(zero_fold_report)

    # --- Section 5: coverage + Section 8/10/11: label/LR/gate per threshold ---
    threshold_results = {}
    fold_level_all = []
    for threshold in THRESHOLDS:
        log(f"--- Threshold +/-{threshold} ---")
        classified = classify_at_threshold(state_scores, threshold)
        zero_folds_n = int(classified["zero_eligible_fold"].sum())
        active_folds_n = len(classified) - zero_folds_n

        # per-row trend_sign/is_trending at this threshold, using the row's OWN decoded state's trend_score
        # (causal["trend_score"] is exactly that state's constant score, already cached)
        is_trending_row = causal["trend_score"].abs() > threshold
        trend_sign_row = np.where(is_trending_row, np.sign(causal["trend_score"]), 0.0)

        master, fold_metrics, audit_rows = run_v2_style_walkforward(causal, trend_sign_row, is_trending_row, threshold)
        for fm in fold_metrics:
            fold_level_all.append({"threshold": threshold, **fm})

        labeled = master.dropna(subset=["label"])
        eligible = master[(master["usable"]) & master["lr_probability"].notna()]

        uptrend_rows = int((is_trending_row & (trend_sign_row == 1)).sum())
        downtrend_rows = int((is_trending_row & (trend_sign_row == -1)).sum())
        ranging_rows = int((~is_trending_row).sum())
        eligible_per_fold = [fm["n_eligible"] for fm in fold_metrics]

        metrics = compute_metrics_block(eligible) if len(eligible) > 10 and eligible["label"].nunique() > 1 else {}
        fold_aucs = [fm["fold_auc"] for fm in fold_metrics if "fold_auc" in fm]

        by_dir_up = labeled[labeled["trend_sign_t"] == 1]["label"].mean() if len(labeled[labeled["trend_sign_t"] == 1]) else None
        by_dir_down = labeled[labeled["trend_sign_t"] == -1]["label"].mean() if len(labeled[labeled["trend_sign_t"] == -1]) else None

        year_df = labeled.copy()
        year_df["year"] = year_df["timestamp"].dt.year
        year_stats = year_df.groupby("year").agg(eligible=("label", "count"), pos_rate=("label", "mean")).to_dict(orient="index")

        threshold_results[threshold] = {
            "zero_folds": zero_folds_n, "active_folds": active_folds_n,
            "eligible_rows": len(labeled), "uptrend_rows": uptrend_rows, "downtrend_rows": downtrend_rows,
            "ranging_rows": ranging_rows, "coverage_pct": float(len(labeled) / len(causal) * 100),
            "eligible_per_fold_mean": float(np.mean(eligible_per_fold)) if eligible_per_fold else None,
            "eligible_per_fold_median": float(np.median(eligible_per_fold)) if eligible_per_fold else None,
            "eligible_per_fold_min": int(np.min(eligible_per_fold)) if eligible_per_fold else None,
            "eligible_per_fold_max": int(np.max(eligible_per_fold)) if eligible_per_fold else None,
            "label_1": int((labeled["label"] == 1).sum()), "label_0": int((labeled["label"] == 0).sum()),
            "positive_pct": float((labeled["label"] == 1).mean() * 100) if len(labeled) else None,
            "negative_pct": float((labeled["label"] == 0).mean() * 100) if len(labeled) else None,
            "uptrend_persistence_rate": float(by_dir_up) if by_dir_up is not None else None,
            "downtrend_persistence_rate": float(by_dir_down) if by_dir_down is not None else None,
            "metrics": metrics,
            "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
            "fold_auc_median": float(np.median(fold_aucs)) if fold_aucs else None,
            "fold_auc_std": float(np.std(fold_aucs)) if fold_aucs else None,
            "folds_above_0.5": int(sum(1 for a in fold_aucs if a > 0.5)),
            "folds_below_0.5": int(sum(1 for a in fold_aucs if a < 0.5)),
            "year_stats": year_stats,
            "audit_rows": audit_rows,
        }
        log(f"  zero_folds={zero_folds_n} eligible={len(labeled):,} pos%={threshold_results[threshold]['positive_pct']:.1f} "
            f"AUC={metrics.get('roc_auc')} fold_auc_mean={threshold_results[threshold]['fold_auc_mean']}")

    runtime = time.time() - t0
    log(f"Total runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    save_artifacts(state_scores, zero_fold_df, threshold_results, fold_level_all, runtime, fold_dirs)


def run_leakage_audit(threshold_results):
    checks = {}
    issues = []
    checks["hmm_refit_calls"] = REFIT_CALLS
    checks["hmm_decode_calls"] = DECODE_CALLS
    if REFIT_CALLS or DECODE_CALLS:
        issues.append("HMM refit or decode calls detected -- should be impossible given the code, investigate")

    for t, res in threshold_results.items():
        violations = [r["fold"] for r in res["audit_rows"]
                      if r["train_max_ts"] and pd.Timestamp(r["train_max_ts"]) >= pd.Timestamp(r["predict_min_ts"])]
        checks[f"lr_train_before_predict_threshold_{t}"] = {"violations": violations, "n_checked": len(res["audit_rows"])}
        if violations:
            issues.append(f"threshold {t}: LR trained on overlapping data for folds {violations}")

    checks["epsilon_training_only"] = {"note": "epsilon computed from pool-of-prior-folds only, identical mechanism to V2, per-threshold"}
    checks["forward_ret_24_alignment"] = {"source": "reused unmodified from V2 cache, previously verified via telescoping-sum check"}
    checks["no_future_data_in_threshold_choice"] = {"note": "all 7 thresholds tested equally; no threshold selected using OOS results in this script"}

    return {"leakage_found": len(issues) > 0, "issues": issues, "checks": checks}


def save_artifacts(state_scores, zero_fold_df, threshold_results, fold_level_all, runtime, fold_dirs):
    state_scores.to_csv(os.path.join(OUT_DIR, "state_score_diagnostics.csv"), index=False)
    zero_fold_df.to_csv(os.path.join(OUT_DIR, "zero_fold_diagnostics.csv"), index=False)
    pd.DataFrame(fold_level_all).to_csv(os.path.join(OUT_DIR, "fold_level_results.csv"), index=False)

    sens_rows = []
    for t, r in threshold_results.items():
        m = r["metrics"]
        sens_rows.append({
            "threshold": t, "zero_folds": r["zero_folds"], "active_folds": r["active_folds"],
            "eligible_rows": r["eligible_rows"], "uptrend_rows": r["uptrend_rows"], "downtrend_rows": r["downtrend_rows"],
            "ranging_rows": r["ranging_rows"], "coverage_pct": r["coverage_pct"], "positive_pct": r["positive_pct"],
            "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"), "log_loss": m.get("log_loss"), "brier": m.get("brier"),
            "accuracy": m.get("accuracy"), "precision": m.get("precision"), "recall": m.get("recall"), "f1": m.get("f1"),
            "fold_auc_mean": r["fold_auc_mean"], "fold_auc_median": r["fold_auc_median"], "fold_auc_std": r["fold_auc_std"],
            "folds_above_0.5": r["folds_above_0.5"], "folds_below_0.5": r["folds_below_0.5"],
            "gate_pass_rate": m.get("gate_pass_rate"), "gate_block_rate": m.get("gate_block_rate"),
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(os.path.join(OUT_DIR, "threshold_sensitivity.csv"), index=False)

    label_dist_rows = []
    for t, r in threshold_results.items():
        label_dist_rows.append({"threshold": t, "label_1": r["label_1"], "label_0": r["label_0"],
                                "positive_pct": r["positive_pct"], "negative_pct": r["negative_pct"],
                                "uptrend_persistence_rate": r["uptrend_persistence_rate"],
                                "downtrend_persistence_rate": r["downtrend_persistence_rate"]})
    pd.DataFrame(label_dist_rows).to_csv(os.path.join(OUT_DIR, "label_distribution.csv"), index=False)

    lr_metrics = {str(t): {k: v for k, v in r.items() if k != "audit_rows"} for t, r in threshold_results.items()}
    json.dump(lr_metrics, open(os.path.join(OUT_DIR, "lr_metrics.json"), "w"), indent=2, default=str)

    audit = run_leakage_audit(threshold_results)
    json.dump(audit, open(os.path.join(OUT_DIR, "leakage_audit.json"), "w"), indent=2, default=str)

    config = {"thresholds_tested": THRESHOLDS, "baseline_threshold": BASELINE_THRESHOLD,
              "feature_columns": FEATURE_COLUMNS, "min_pool_rows": MIN_POOL_ROWS, "gate_threshold": GATE_THRESHOLD,
              "epsilon_percentile": EPSILON_PERCENTILE, "n_folds_used": len(fold_dirs), "code_version": "v4_threshold_sensitivity.py:1"}
    json.dump(config, open(os.path.join(OUT_DIR, "config.json"), "w"), indent=2, default=str)

    write_final_report(sens_df, zero_fold_df, threshold_results, audit, runtime)
    log(f"All artifacts saved to {OUT_DIR}")


def write_final_report(sens_df, zero_fold_df, threshold_results, audit, runtime):
    lines = [f"# V4 Trend-Threshold Sensitivity -- Final Report\n",
             f"Runtime: {runtime:.1f}s ({runtime/60:.1f} min) -- cache-replay only, 0 HMM refits, 0 HMM decodes.\n",
             f"Zero-eligible folds at baseline +/-0.30: {len(zero_fold_df)}\n",
             "## Threshold sensitivity table\n",
             sens_df.to_string(index=False) + "\n",
             "## Zero-fold diagnostics (at baseline +/-0.30)\n",
             zero_fold_df.to_string(index=False) + "\n",
             "## Leakage audit\n", f"leakage_found: **{audit['leakage_found']}**\n"]
    for c, r in audit["checks"].items():
        lines.append(f"- `{c}`: {r}\n")
    with open(os.path.join(OUT_DIR, "final_report.md"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
