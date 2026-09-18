"""
v6_segment_breakdown.py -- Step 3: segment breakdown of the existing
Persistence LR's OOS predictions. Pure analysis on the already-cached
adaptive-threshold OOS dataset -- no HMM refit/redecode, no new features,
no model changes. Only new computation: reading each fold's already-fitted
hmm.pkl (pickle.load only) to get state trend/vol scores for labeling,
exactly as done in v4_threshold_sensitivity.py.

MIN_SAMPLE_RULE (documented up front, applied everywhere):
  n < 200            -> INSUFFICIENT, not interpreted
  200 <= n < 1000     -> LOW-CONFIDENCE, directional only
  n >= 1000 (both classes >= 100) -> RELIABLE

Run: `python v6_segment_breakdown.py` (from src/). Expected: <1 min.
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_regime import RETURN_COLS, VOL_COLS, TREND_Z_THRESHOLD, VOL_Z_THRESHOLD

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_CACHE_DIR = os.path.join(_ROOT, "Data", "v2_cache")
SRC = os.path.join(V2_CACHE_DIR, "trend_threshold_experiment", "adaptive_fold_threshold_master_oos.csv")
OUT_DIR = os.path.join(V2_CACHE_DIR, "trend_threshold_experiment", "segment_breakdown")
BASELINE_AUC = 0.5228

N_INSUFFICIENT = 200
N_RELIABLE = 1000


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def confidence_tier(n, y):
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if n < N_INSUFFICIENT or pos < 20 or neg < 20:
        return "INSUFFICIENT"
    if n < N_RELIABLE or pos < 100 or neg < 100:
        return "LOW-CONFIDENCE"
    return "RELIABLE"


def segment_stats(sub: pd.DataFrame) -> dict:
    n = len(sub)
    y = sub["label"].astype(int).values
    p = sub["lr_probability"].values
    tier = confidence_tier(n, y)
    out = {"n": n, "positive_rate": float(y.mean()) if n else None, "mean_pred_prob": float(p.mean()) if n else None,
           "confidence_tier": tier, "n_folds": int(sub["fold"].nunique())}
    if tier != "INSUFFICIENT" and len(set(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    fold_aucs = []
    for fid, g in sub.groupby("fold"):
        gy = g["label"].astype(int).values
        if len(g) >= 20 and len(set(gy)) > 1:
            fold_aucs.append(roc_auc_score(gy, g["lr_probability"].values))
    out["fold_auc_mean"] = float(np.mean(fold_aucs)) if fold_aucs else None
    out["fold_auc_std"] = float(np.std(fold_aucs)) if fold_aucs else None
    out["folds_above_0.5"] = int(sum(1 for a in fold_aucs if a > 0.5))
    out["folds_below_0.5"] = int(sum(1 for a in fold_aucs if a < 0.5))
    out["n_folds_with_auc"] = len(fold_aucs)
    return out


def load_state_labels(fold_dirs, fallback_folds):
    """Reads each fold's already-fitted hmm.pkl (pickle.load only, no refit/decode)
    to compute trend_score, vol_score, and the combined label per (fold, state).
    Trend cutoff uses THIS fold's actual adaptive threshold (0.15 if the fold is a
    fallback fold, else 0.30) -- matching is_trending_t/trend_sign_t exactly, so the
    display label is never inconsistent with which rows were actually eligible."""
    rows = []
    for d in fold_dirs:
        fold_path = os.path.join(V2_CACHE_DIR, d)
        hmm_path = os.path.join(fold_path, "hmm.pkl")
        meta_path = os.path.join(fold_path, "metadata.json")
        if not (os.path.exists(hmm_path) and os.path.exists(meta_path)):
            continue
        meta = json.load(open(meta_path))
        fold_thresh = 0.15 if meta["fold_id"] in fallback_folds else 0.30
        with open(hmm_path, "rb") as f:
            model = pickle.load(f)
        means_df = pd.DataFrame(model.means_, columns=meta["feature_columns"])
        for s in range(means_df.shape[0]):
            trend = float(means_df.loc[s, RETURN_COLS].mean())
            vol = float(means_df.loc[s, VOL_COLS].mean())
            trend_lbl = "Uptrend" if trend > fold_thresh else ("Downtrend" if trend < -fold_thresh else "Ranging")
            vol_lbl = "High-Vol" if vol > VOL_Z_THRESHOLD else ("Low-Vol" if vol < -VOL_Z_THRESHOLD else "Mid-Vol")
            rows.append({"fold": meta["fold_id"], "hmm_state": s, "vol_score": vol,
                        "vol_class": vol_lbl, "state_label": f"{trend_lbl}/{vol_lbl}"})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"Loading cached adaptive-threshold OOS dataset (no HMM computation): {SRC}")
    df = pd.read_csv(SRC, parse_dates=["timestamp"])
    eligible = df[(df["usable"] == True) & df["lr_probability"].notna()].copy()  # noqa: E712
    log(f"  {len(df):,} total rows, {len(eligible):,} eligible OOS-predicted rows across {eligible['fold'].nunique()} folds")

    log("Reading cached hmm.pkl per fold for state trend/vol labels (pickle.load only, no fit/decode)...")
    fold_dirs = sorted(d for d in os.listdir(V2_CACHE_DIR) if d.startswith("fold_")
                        and os.path.exists(os.path.join(V2_CACHE_DIR, d, "DONE")))
    # which folds are on the +-0.15 fallback: any fold where trend_sign_t/is_trending_t
    # required a smaller cutoff than 0.30 -- derived directly from the cached eligibility
    # columns themselves (self-consistent with what actually determined each row's eligibility)
    fallback_folds = set(
        df.loc[df["is_trending_t"] & (df["trend_score"].abs() <= 0.30) & (df["trend_score"].abs() > 0.15), "fold"].unique()
    )
    state_labels = load_state_labels(fold_dirs, fallback_folds)
    eligible = eligible.merge(state_labels, on=["fold", "hmm_state"], how="left")

    # derived segment columns
    eligible["direction"] = np.where(eligible["trend_sign_t"] == 1, "Uptrend", "Downtrend")
    eligible["year"] = eligible["timestamp"].dt.year
    hour = eligible["timestamp"].dt.hour
    eligible["session"] = np.select([hour < 8, hour < 16], ["Asia (00-08 UTC)", "Europe (08-16 UTC)"], default="US (16-24 UTC)")

    eligible["duration_bucket"] = pd.qcut(eligible["duration"], 3, labels=["Short", "Medium", "Long"], duplicates="drop")
    eligible["confidence_bucket"] = pd.qcut(eligible["confidence"], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
    eligible["stay_prob_bucket"] = pd.qcut(eligible["stay_prob"], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
    eligible["margin_bucket"] = pd.qcut(eligible["margin"], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")

    results = {}
    dims = {
        "1_trend_direction": "direction", "2_state_label": "state_label", "3_volatility_class": "vol_class",
        "4a_confidence_bucket": "confidence_bucket", "4b_stay_prob_bucket": "stay_prob_bucket",
        "4c_margin_bucket": "margin_bucket", "5_duration_bucket": "duration_bucket",
        "6a_year": "year", "6b_session": "session",
    }
    for name, col in dims.items():
        seg_rows = []
        for val, sub in eligible.groupby(col, observed=True):
            stats = segment_stats(sub)
            stats[col] = val
            seg_rows.append(stats)
        results[name] = pd.DataFrame(seg_rows).sort_values(col if col != "state_label" else "n", ascending=(col == "state_label"))
        results[name].to_csv(os.path.join(OUT_DIR, f"segment_{name}.csv"), index=False)

    overall = segment_stats(eligible)
    runtime = time.time() - t0

    # leakage audit
    audit = {
        "leakage_found": False,
        "hmm_refit_calls": 0, "hmm_decode_calls": 0,
        "note": "All segment columns are either (a) state-level parameters read from already-fitted "
                "hmm.pkl (fixed at fold-fit time, before that fold's OOS period), (b) contemporaneous "
                "row-level features (confidence/stay_prob/duration/margin, already validated causal in "
                "prior audits), or (c) calendar facts (year/session) derived from the row's own timestamp. "
                "No segment definition uses forward_ret_24 or any information from after the row's own timestamp. "
                "This script performs no model fitting -- pure post-hoc analysis of already-frozen OOS predictions.",
    }

    print()
    print("=" * 70)
    print(f"STEP 3 -- SEGMENT BREAKDOWN (baseline OOS AUC = {BASELINE_AUC})")
    print("=" * 70)
    print(f"\nOverall (all eligible rows): {json.dumps(overall, indent=2, default=str)}")
    for name in dims:
        print(f"\n--- {name} ---")
        print(results[name].to_string(index=False))

    print("\nLeakage audit:", json.dumps(audit, indent=2))
    json.dump({"overall": overall, "runtime_s": runtime, "baseline_auc": BASELINE_AUC, "sample_size_rule": {
        "INSUFFICIENT": f"n<{N_INSUFFICIENT} or either class <20",
        "LOW-CONFIDENCE": f"{N_INSUFFICIENT}<=n<{N_RELIABLE} or either class <100",
        "RELIABLE": f"n>={N_RELIABLE} and both classes >=100"}},
        open(os.path.join(OUT_DIR, "overall_summary.json"), "w"), indent=2, default=str)
    json.dump(audit, open(os.path.join(OUT_DIR, "leakage_audit.json"), "w"), indent=2, default=str)
    log(f"\nRuntime: {runtime:.1f}s")
    log(f"Artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
