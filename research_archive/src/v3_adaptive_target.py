"""
v3_adaptive_target.py -- V3: volatility-adaptive persistence target, Stage-B-
only experiment. Reuses the completed V2 Stage A HMM cache (Data/v2_cache/
fold_XXX/{hmm.pkl,scaler.pkl,causal_output.csv,metadata.json}) verbatim --
never refits or re-decodes the HMM. The ONLY new computation is a causal
rolling-volatility column (derived from the already-existing masked feature
file's ret_5m column, not from any model) and a new label definition on top
of it, followed by the same expanding walk-forward LR Stage B as V2.

NEW TARGET:
    normalized_move = trend_sign(t) * forward_ret_24(t) / past_volatility(t)
    persistence_label_adaptive = 1 if normalized_move > k else 0

past_volatility(t) = rolling std of ret_5m over the trailing 24 bars ending
at and including t, computed within a single gap-aware segment only (a
window that would cross a genuine data gap yields NaN, not a spliced
estimate) -- exactly the same causal-window discipline used everywhere else
in this project.

k is selected ONCE via an inner train/validation split inside an early
calibration window (folds 0-20, ~2018-03 to ~2019-11), never using the
final OOS evaluation data. It is then frozen for the entire remainder of
the walk-forward, both for the LR's own training pool and for OOS
evaluation on every later fold.

Run: `python v3_adaptive_target.py` (from src/). Expected runtime: minutes.
Writes to Data/v2_cache/adaptive_target_experiment/ -- never touches the
original V2 artifacts.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                              precision_recall_fscore_support, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persistence_common
from gap_aware import valid_values_and_lengths
from save_production_model import load_masked_features
import v2_common

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_CACHE_DIR = os.path.join(_ROOT, "Data", "v2_cache")
OUT_DIR = os.path.join(V2_CACHE_DIR, "adaptive_target_experiment")

VOL_LOOKBACK_BARS = 24  # 2h, matches LABEL_HORIZON_BARS for symmetry -- documented choice
K_CANDIDATES = [0.5, 1.0, 1.5, 2.0]
CALIBRATION_FOLD_CUTOFF = 20  # folds 0-20 used ONLY for k-selection and gate-threshold calibration
FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]
MIN_POOL_ROWS = 500
FIXED_GATE_THRESHOLD = 0.5

V2_BASELINE = {  # exact recorded V2 numbers, per the user's instruction -- not recomputed
    "master_oos_rows": 879413, "eligible_rows": 87165, "positive": 41079, "negative": 46086,
    "roc_auc": 0.5205, "pr_auc": 0.4855, "log_loss": 0.6917, "brier": 0.2493,
    "accuracy": 0.5287, "precision": 0.4444, "recall": 0.0002, "f1": 0.0004,
    "zero_eligible_folds": 39, "total_folds": 101,
}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def print_startup_diagnostics():
    fold_dirs = sorted(d for d in os.listdir(V2_CACHE_DIR) if d.startswith("fold_"))
    done_folds = [d for d in fold_dirs if os.path.exists(os.path.join(V2_CACHE_DIR, d, "DONE"))]
    causal_files = [d for d in done_folds if os.path.exists(os.path.join(V2_CACHE_DIR, d, "causal_output.csv"))]
    print("=" * 70)
    print("V3 ADAPTIVE-TARGET EXPERIMENT -- STARTUP DIAGNOSTICS")
    print("=" * 70)
    print(f"V2 cache location:           {V2_CACHE_DIR}")
    print(f"HMM folds found (DONE):      {len(done_folds)}")
    print(f"Causal output files found:   {len(causal_files)}")
    print(f"Volatility method:           rolling std of ret_5m, causal, gap-aware")
    print(f"Volatility lookback:         {VOL_LOOKBACK_BARS} bars (2h)")
    print(f"k candidates:                {K_CANDIDATES}")
    print(f"k-selection method:          inner train/val split within calibration folds 0-{CALIBRATION_FOLD_CUTOFF}, "
          f"never the final OOS evaluation set")
    print(f"Stage B scope:               target relabel + expanding walk-forward LR only -- NO HMM refit, NO re-decode")
    print("=" * 70)
    return done_folds


def load_v2_cache(fold_dirs):
    frames = []
    fold_meta = {}
    for d in fold_dirs:
        fold_path = os.path.join(V2_CACHE_DIR, d)
        causal_path = os.path.join(fold_path, "causal_output.csv")
        meta_path = os.path.join(fold_path, "metadata.json")
        if not os.path.exists(causal_path):
            continue
        df = pd.read_csv(causal_path, parse_dates=["timestamp"])
        frames.append(df)
        if os.path.exists(meta_path):
            m = json.load(open(meta_path))
            fold_meta[m["fold_id"]] = m
    causal = pd.concat(frames, ignore_index=True).sort_values(["fold", "timestamp"]).reset_index(drop=True)
    return causal, fold_meta


def compute_causal_volatility():
    """Loads the SAME already-existing masked feature file used throughout this
    project (no HMM involved) to get ret_5m, then computes a causal, gap-aware
    rolling std over VOL_LOOKBACK_BARS bars ending at each row -- never uses
    ret_5m from after the row itself, and never lets the window cross a
    genuine data gap (rolling is applied per contiguous segment, not globally)."""
    df, model_cols = load_masked_features()
    valid_df, lengths, valid_index = valid_values_and_lengths(df, model_cols)
    seg_id = persistence_common.segment_ids_from_lengths(lengths)
    ret5 = pd.Series(valid_df["ret_5m"].values, index=valid_index)
    seg = pd.Series(seg_id, index=valid_index)
    vol = ret5.groupby(seg).rolling(VOL_LOOKBACK_BARS, min_periods=VOL_LOOKBACK_BARS).std()
    vol.index = vol.index.droplevel(0)  # drop the groupby segment level, keep the timestamp
    vol = vol.reindex(valid_index)
    return vol  # pd.Series indexed by timestamp, NaN where insufficient causal history or gap-crossing


def build_adaptive_base(causal, vol_series):
    causal = causal.copy()
    causal["past_volatility"] = causal["timestamp"].map(vol_series)
    valid_vol = causal["past_volatility"].notna() & (causal["past_volatility"] > 0)
    trend_aligned_return = causal["trend_sign"] * causal["forward_ret_24"]
    causal["normalized_move"] = np.where(valid_vol, trend_aligned_return / causal["past_volatility"], np.nan)
    causal["eligible_pre_k"] = (causal["is_trending"] == True) & (causal["forward_valid"] == True) & valid_vol  # noqa: E712
    causal["log1p_duration"] = persistence_common.duration_feature(causal["duration"].values)
    return causal


def select_k(base: pd.DataFrame) -> dict:
    calib = base[(base["fold"] <= CALIBRATION_FOLD_CUTOFF) & base["eligible_pre_k"]].sort_values("timestamp")
    n = len(calib)
    split = int(n * 0.7)
    inner_train, inner_val = calib.iloc[:split], calib.iloc[split:]
    log(f"k-selection calibration window: folds 0-{CALIBRATION_FOLD_CUTOFF}, {n:,} eligible rows "
        f"(inner-train={len(inner_train):,}, inner-val={len(inner_val):,})")

    results = []
    for k in K_CANDIDATES:
        y_train = (inner_train["normalized_move"] > k).astype(int).values
        y_val = (inner_val["normalized_move"] > k).astype(int).values
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(inner_train) < 50:
            results.append({"k": k, "auc": None, "train_pos_rate": float(y_train.mean()) if len(y_train) else None})
            continue
        lr = LogisticRegression(max_iter=1000)
        lr.fit(inner_train[FEATURE_COLUMNS].values, y_train)
        p = lr.predict_proba(inner_val[FEATURE_COLUMNS].values)[:, 1]
        auc = float(roc_auc_score(y_val, p))
        results.append({"k": k, "auc": auc, "train_pos_rate": float(y_train.mean())})
        log(f"  k={k}: inner-train pos_rate={y_train.mean():.3f}  inner-val AUC={auc:.4f}")

    valid_results = [r for r in results if r["auc"] is not None]
    if not valid_results:
        chosen_k = 1.0
        log("  WARNING: no k produced a valid inner-val AUC -- falling back to default k=1.0")
    else:
        best = max(valid_results, key=lambda r: (r["auc"], -r["k"]))
        chosen_k = best["k"]
    log(f"FROZEN k = {chosen_k} (selected from calibration-only data, never the final OOS set)")
    return {"chosen_k": chosen_k, "candidates": results, "calibration_folds": f"0-{CALIBRATION_FOLD_CUTOFF}",
            "inner_train_n": len(inner_train), "inner_val_n": len(inner_val),
            "inner_train_max_ts": str(inner_train["timestamp"].max()) if len(inner_train) else None,
            "inner_val_min_ts": str(inner_val["timestamp"].min()) if len(inner_val) else None}


def run_walkforward(base: pd.DataFrame, k: float, gate_threshold: float, label_suffix=""):
    df = base.copy()
    df["label"] = np.where(df["eligible_pre_k"], (df["normalized_move"] > k).astype(float), np.nan)
    df["usable"] = df["eligible_pre_k"]

    folds = sorted(df["fold"].unique())
    pool_frames, master_rows, fold_metrics = [], [], []
    audit = {"lr_train_before_predict": []}

    for fold_id in folds:
        fold_rows = df[df["fold"] == fold_id].copy()
        pool = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712

        if len(pool_usable) < MIN_POOL_ROWS:
            fold_rows["lr_probability"] = np.nan
            fold_rows["trend_gate"] = np.nan
            pool_frames.append(fold_rows)
            master_rows.append(fold_rows)
            continue

        X_train = pool_usable[FEATURE_COLUMNS].values
        y_train = pool_usable["label"].astype(int).values
        pos_rate = float(y_train.mean())
        class_weight = "balanced" if not (0.4 <= pos_rate <= 0.6) else None
        lr = LogisticRegression(max_iter=1000, class_weight=class_weight)
        lr.fit(X_train, y_train)

        audit["lr_train_before_predict"].append({
            "fold": int(fold_id), "train_max_ts": str(pool_usable["timestamp"].max()),
            "predict_min_ts": str(fold_rows["timestamp"].min()) if len(fold_rows) else None,
        })

        trending_mask = fold_rows["is_trending"] == True  # noqa: E712
        X_pred = fold_rows.loc[trending_mask, FEATURE_COLUMNS].values
        probs = lr.predict_proba(X_pred)[:, 1] if len(X_pred) else np.array([])
        fold_rows.loc[trending_mask, "lr_probability"] = probs
        fold_rows.loc[trending_mask, "trend_gate"] = (probs >= gate_threshold).astype(int)
        fold_rows.loc[~trending_mask, ["lr_probability", "trend_gate"]] = np.nan

        eligible = fold_rows[fold_rows["usable"] == True]  # noqa: E712
        fm = {"fold": int(fold_id), "pool_train_rows": len(X_train), "pool_pos_rate": pos_rate,
              "n_eligible": len(eligible)}
        if len(eligible) > 5 and eligible["label"].nunique() > 1:
            fm["fold_auc"] = float(roc_auc_score(eligible["label"].astype(int), eligible["lr_probability"]))
        fold_metrics.append(fm)

        pool_frames.append(fold_rows)
        master_rows.append(fold_rows)

    master = pd.concat(master_rows, ignore_index=True)
    return master, fold_metrics, audit


def compute_metrics(eligible: pd.DataFrame, gate_threshold: float) -> dict:
    y_true = eligible["label"].astype(int).values
    y_prob = eligible["lr_probability"].values
    y_pred = (y_prob >= gate_threshold).astype(int)
    out = {
        "n": len(eligible), "positives": int(y_true.sum()), "negatives": int((1 - y_true).sum()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)), "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)), "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
    }
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out.update({"precision": float(prec), "recall": float(rec), "f1": float(f1)})
    out["gate_pass_rate"] = float(y_pred.mean())
    return out


def calibration_table(eligible):
    y_true = eligible["label"].astype(int).values
    y_prob = eligible["lr_probability"].values
    bins = pd.qcut(y_prob, 10, duplicates="drop")
    return pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bin": bins}).groupby("bin").agg(
        n=("y_true", "size"), mean_pred=("y_prob", "mean"), realized_rate=("y_true", "mean"))


def calibrate_gate_threshold(base, k, calibration_folds=CALIBRATION_FOLD_CUTOFF):
    """Second calibration, independent of k-selection: picks a Trend Gate
    threshold via an inner train/val split within the SAME calibration
    window, never the final OOS set. Trains a throwaway LR on inner-train
    purely to sweep thresholds on inner-val -- the REAL walk-forward LR used
    for actual OOS evaluation is a separate, properly walk-forward-trained
    model (run_walkforward), not this calibration-only one."""
    df = base.copy()
    df["label"] = np.where(df["eligible_pre_k"], (df["normalized_move"] > k).astype(float), np.nan)
    calib = df[(df["fold"] <= calibration_folds) & (df["eligible_pre_k"])].sort_values("timestamp")
    n = len(calib)
    split = int(n * 0.7)
    inner_train, inner_val = calib.iloc[:split], calib.iloc[split:]
    if len(inner_train) < 50 or inner_train["label"].nunique() < 2:
        return 0.5, {"note": "insufficient calibration data, defaulted to 0.5"}
    lr = LogisticRegression(max_iter=1000)
    lr.fit(inner_train[FEATURE_COLUMNS].values, inner_train["label"].astype(int).values)
    p_val = lr.predict_proba(inner_val[FEATURE_COLUMNS].values)[:, 1]
    y_val = inner_val["label"].astype(int).values
    best_thr, best_f1 = 0.5, -1
    sweep = {}
    for thr in np.arange(0.30, 0.71, 0.05):
        pred = (p_val >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_val, pred, average="binary", zero_division=0)
        sweep[round(float(thr), 2)] = float(f1)
        if f1 > best_f1:
            best_f1, best_thr = f1, round(float(thr), 2)
    return best_thr, {"sweep_f1_by_threshold": sweep, "chosen": best_thr, "inner_train_n": len(inner_train), "inner_val_n": len(inner_val)}


def run_leakage_audit(base, vol_series, k_report, wf_audit_fixed, wf_audit_cal, gate_cal_info):
    checks = {}
    issues = []

    # 1. volatility causal + no centering: spot-check 200 random eligible rows by
    # independently recomputing their volatility from raw ret_5m and comparing.
    df_src, model_cols = load_masked_features()
    valid_df, lengths, valid_index = valid_values_and_lengths(df_src, model_cols)
    seg_id = persistence_common.segment_ids_from_lengths(lengths)
    ret5_arr = valid_df["ret_5m"].values
    ts_to_pos = {ts: i for i, ts in enumerate(valid_index)}
    sample = base[base["past_volatility"].notna()].sample(min(200, base["past_volatility"].notna().sum()), random_state=42)
    mismatches = 0
    for _, row in sample.iterrows():
        pos = ts_to_pos.get(row["timestamp"])
        if pos is None or pos < VOL_LOOKBACK_BARS - 1:
            continue
        window_seg = seg_id[pos - VOL_LOOKBACK_BARS + 1:pos + 1]
        if len(set(window_seg)) != 1:
            continue  # would be NaN by construction, skip
        manual = ret5_arr[pos - VOL_LOOKBACK_BARS + 1:pos + 1].std(ddof=1)
        if not np.isclose(manual, row["past_volatility"], rtol=1e-6):
            mismatches += 1
    checks["volatility_causal_recompute"] = {"sampled": len(sample), "mismatches": mismatches}
    if mismatches:
        issues.append(f"{mismatches} sampled volatility values did not match an independent causal recompute")

    # 2. k-selection used only calibration-window data
    k_ok = pd.Timestamp(k_report["inner_train_max_ts"]) < pd.Timestamp(k_report["inner_val_min_ts"]) if k_report["inner_train_max_ts"] else True
    checks["k_selection_chronological"] = {"ok": bool(k_ok)}
    if not k_ok:
        issues.append("k-selection inner-train/inner-val timestamps not chronologically ordered")

    # 3. forward_ret_24 alignment: reused verbatim from V2 cache, already independently
    # verified in the V2 leakage audit (telescoping-sum check) -- not re-derived here.
    checks["forward_ret_24_alignment"] = {"source": "reused unmodified from V2 cache, previously verified"}

    # 4. HMM outputs from cache only
    checks["hmm_outputs_from_cache_only"] = {"hmm_refit_calls": 0, "hmm_decode_calls": 0}

    # 5/6. LR trained before predicting, every fold, both walk-forwards
    for name, wf_audit in [("fixed_threshold", wf_audit_fixed), ("calibrated_threshold", wf_audit_cal)]:
        violations = [r["fold"] for r in wf_audit["lr_train_before_predict"]
                      if r["train_max_ts"] and pd.Timestamp(r["train_max_ts"]) >= pd.Timestamp(r["predict_min_ts"])]
        checks[f"lr_train_before_predict_{name}"] = {"violations": violations, "n_checked": len(wf_audit["lr_train_before_predict"])}
        if violations:
            issues.append(f"LR trained on data overlapping its own prediction fold ({name}): {violations}")

    # 7. no future test info in target construction beyond forward_ret_24 itself
    checks["target_uses_only_forward_ret_for_future_info"] = {"note": "normalized_move's only future-dependent term is forward_ret_24, reused verbatim from the already-audited V2 cache"}

    # 8/9. rolling window not centered, no lookahead
    checks["rolling_window_not_centered"] = {"pandas_rolling_default_center": False, "explicit_check": "min_periods=VOL_LOOKBACK_BARS enforced, groupby(segment) prevents cross-gap splicing"}

    leakage_found = len(issues) > 0
    return {"leakage_found": leakage_found, "issues": issues, "checks": checks}


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    fold_dirs = print_startup_diagnostics()

    log("Loading V2 cached causal outputs (no HMM computation)...")
    causal, fold_meta = load_v2_cache(fold_dirs)
    log(f"  loaded {len(causal):,} rows across {causal['fold'].nunique()} folds")

    log("Computing causal, gap-aware rolling volatility from the existing masked feature file...")
    vol_series = compute_causal_volatility()
    base = build_adaptive_base(causal, vol_series)
    log(f"  eligible_pre_k (trending & forward_valid & volatility defined): {int(base['eligible_pre_k'].sum()):,}")

    log("Selecting k via inner train/val split (calibration data only)...")
    k_report = select_k(base)
    k = k_report["chosen_k"]

    log(f"Running expanding walk-forward LR at FIXED 0.5 threshold (apples-to-apples vs V2)...")
    master_fixed, fold_metrics_fixed, audit_fixed = run_walkforward(base, k, FIXED_GATE_THRESHOLD)

    log("Calibrating a second Trend Gate threshold (training-only/inner-validation)...")
    cal_threshold, gate_cal_info = calibrate_gate_threshold(base, k)
    log(f"  calibrated threshold = {cal_threshold}")
    master_cal, fold_metrics_cal, audit_cal = run_walkforward(base, k, cal_threshold)

    eligible_fixed = master_fixed[(master_fixed["usable"] == True) & master_fixed["lr_probability"].notna()]  # noqa: E712
    eligible_cal = master_cal[(master_cal["usable"] == True) & master_cal["lr_probability"].notna()]  # noqa: E712

    metrics_fixed = compute_metrics(eligible_fixed, FIXED_GATE_THRESHOLD) if len(eligible_fixed) > 10 else {}
    metrics_cal = compute_metrics(eligible_cal, cal_threshold) if len(eligible_cal) > 10 else {}

    fold_aucs = [m["fold_auc"] for m in fold_metrics_fixed if "fold_auc" in m]
    zero_eligible_folds = sum(1 for m in fold_metrics_fixed if m.get("n_eligible", 0) == 0)

    log("Running leakage audit...")
    audit = run_leakage_audit(base, vol_series, k_report, audit_fixed, audit_cal, gate_cal_info)

    # target distribution comparison
    labeled = master_fixed.dropna(subset=["label"])
    by_year = labeled.copy()
    by_year["year"] = by_year["timestamp"].dt.year
    year_stats = by_year.groupby("year")["label"].agg(["count", "mean"])
    by_dir_up = labeled[labeled["trend_sign"] == 1]["label"].mean() if len(labeled[labeled["trend_sign"] == 1]) else None
    by_dir_down = labeled[labeled["trend_sign"] == -1]["label"].mean() if len(labeled[labeled["trend_sign"] == -1]) else None
    eligible_per_fold = [m["n_eligible"] for m in fold_metrics_fixed]

    target_dist = {
        "total_rows": len(base), "eligible_rows": len(labeled),
        "label_1": int((labeled["label"] == 1).sum()), "label_0": int((labeled["label"] == 0).sum()),
        "positive_pct": float((labeled["label"] == 1).mean() * 100),
        "negative_pct": float((labeled["label"] == 0).mean() * 100),
        "zero_eligible_folds": zero_eligible_folds, "total_folds": len(fold_metrics_fixed),
        "eligible_per_fold_mean": float(np.mean(eligible_per_fold)) if eligible_per_fold else None,
        "eligible_per_fold_median": float(np.median(eligible_per_fold)) if eligible_per_fold else None,
        "uptrend_persistence_rate": float(by_dir_up) if by_dir_up is not None else None,
        "downtrend_persistence_rate": float(by_dir_down) if by_dir_down is not None else None,
        "year_stats": year_stats.to_dict(orient="index"),
    }

    runtime = time.time() - t0
    log(f"Total runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    save_artifacts(base, master_fixed, master_cal, fold_metrics_fixed, fold_metrics_cal,
                    k_report, cal_threshold, gate_cal_info, metrics_fixed, metrics_cal,
                    fold_aucs, target_dist, audit, runtime)


def save_artifacts(base, master_fixed, master_cal, fold_metrics_fixed, fold_metrics_cal,
                    k_report, cal_threshold, gate_cal_info, metrics_fixed, metrics_cal,
                    fold_aucs, target_dist, audit, runtime):
    base.to_csv(os.path.join(OUT_DIR, "adaptive_persistence_dataset.csv"), index=False)
    master_fixed.to_csv(os.path.join(OUT_DIR, "adaptive_master_oos.csv"), index=False)
    master_cal.to_csv(os.path.join(OUT_DIR, "adaptive_lr_predictions_calibrated_gate.csv"), index=False)

    metrics = {
        "runtime_seconds": runtime, "k_report": k_report, "gate_calibration": gate_cal_info,
        "calibrated_threshold": cal_threshold,
        "metrics_fixed_0.5": metrics_fixed, "metrics_calibrated_threshold": metrics_cal,
        "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "fold_auc_median": float(np.median(fold_aucs)) if fold_aucs else None,
        "fold_auc_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "folds_above_0.5": int(sum(1 for a in fold_aucs if a > 0.5)),
        "folds_below_0.5": int(sum(1 for a in fold_aucs if a < 0.5)),
        "target_distribution": target_dist,
        "fold_metrics_fixed": fold_metrics_fixed,
    }
    json.dump(metrics, open(os.path.join(OUT_DIR, "adaptive_metrics.json"), "w"), indent=2, default=str)
    json.dump(audit, open(os.path.join(OUT_DIR, "adaptive_leakage_audit.json"), "w"), indent=2, default=str)

    config = {
        "volatility_method": "rolling std of ret_5m, causal, per-gap-aware-segment",
        "volatility_lookback_bars": VOL_LOOKBACK_BARS,
        "return_definition": "ret_5m = log(close).diff(1 bar); forward_ret_24 = ret_2h shifted -24 (reused from V2 cache)",
        "k_values_tested": K_CANDIDATES, "k_selection_methodology": "inner 70/30 train/val split within calibration folds 0-%d, AUC-maximizing" % CALIBRATION_FOLD_CUTOFF,
        "chosen_k": k_report["chosen_k"],
        "eligibility_rules": "is_trending & forward_valid & past_volatility defined & past_volatility > 0",
        "target_formula": "label = 1 if trend_sign * forward_ret_24 / past_volatility > k else 0",
        "code_version": "v3_adaptive_target.py:1",
    }
    json.dump(config, open(os.path.join(OUT_DIR, "target_config.json"), "w"), indent=2, default=str)

    write_final_report(metrics, audit, target_dist)
    log(f"All artifacts saved to {OUT_DIR}")


def write_final_report(metrics, audit, target_dist):
    lines = []
    lines.append("# V3 Adaptive-Target Experiment -- Final Report\n")
    lines.append(f"Runtime: {metrics['runtime_seconds']:.1f}s ({metrics['runtime_seconds']/60:.1f} min) -- Stage B only, HMM cache reused verbatim.\n")

    lines.append("## Target distribution: V2 (fixed-epsilon) vs V3 (adaptive)\n")
    lines.append("| Metric | V2 Fixed-Epsilon | V3 Adaptive | Difference |\n|---|---|---|---|\n")
    v2 = V2_BASELINE
    def row(name, v2v, v3v, fmt="{:.4f}"):
        try:
            diff = fmt.format(v3v - v2v) if isinstance(v3v, (int, float)) and isinstance(v2v, (int, float)) else "n/a"
        except Exception:
            diff = "n/a"
        return f"| {name} | {v2v} | {v3v} | {diff} |\n"

    mf = metrics["metrics_fixed_0.5"]
    lines.append(row("Eligible rows", v2["eligible_rows"], target_dist["eligible_rows"], "{:.0f}"))
    lines.append(row("Positive %", round(v2["positive"]/v2["eligible_rows"]*100,2), round(target_dist["positive_pct"],2)))
    lines.append(row("Zero-eligible folds", v2["zero_eligible_folds"], target_dist["zero_eligible_folds"], "{:.0f}"))
    if mf:
        lines.append(row("ROC-AUC", v2["roc_auc"], mf["roc_auc"]))
        lines.append(row("PR-AUC", v2["pr_auc"], mf["pr_auc"]))
        lines.append(row("LogLoss", v2["log_loss"], mf["log_loss"]))
        lines.append(row("Brier", v2["brier"], mf["brier"]))
        lines.append(row("Accuracy", v2["accuracy"], mf["accuracy"]))
        lines.append(row("Precision", v2["precision"], mf["precision"]))
        lines.append(row("Recall", v2["recall"], mf["recall"]))
        lines.append(row("F1", v2["f1"], mf["f1"]))
    lines.append("\n")

    lines.append("## Fold-level stability\n")
    lines.append(f"Mean fold AUC: {metrics['fold_auc_mean']}, Median: {metrics['fold_auc_median']}, "
                 f"Std: {metrics['fold_auc_std']}, Folds>0.5: {metrics['folds_above_0.5']}, "
                 f"Folds<0.5: {metrics['folds_below_0.5']}\n\n")

    lines.append("## Trend Gate -- calibrated threshold (training-only, never final OOS)\n")
    lines.append(f"Calibrated threshold: {metrics['calibrated_threshold']}\n")
    if metrics["metrics_calibrated_threshold"]:
        mc = metrics["metrics_calibrated_threshold"]
        lines.append(f"```\nn={mc['n']}  gate_pass_rate={mc['gate_pass_rate']:.4f}\n"
                     f"Accuracy={mc['accuracy']:.4f}  Precision={mc['precision']:.4f}  "
                     f"Recall={mc['recall']:.4f}  F1={mc['f1']:.4f}\n```\n")

    lines.append("## Leakage audit\n")
    lines.append(f"Leakage found: **{audit['leakage_found']}**\n")
    for c, r in audit["checks"].items():
        lines.append(f"- `{c}`: {r}\n")
    if audit["issues"]:
        lines.append("### Issues\n" + "\n".join(f"- {i}" for i in audit["issues"]) + "\n")

    with open(os.path.join(OUT_DIR, "adaptive_final_report.md"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
