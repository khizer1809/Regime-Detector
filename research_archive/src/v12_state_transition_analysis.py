"""
v12_state_transition_analysis.py -- Step 8: HMM state + state-transition
conditional return analysis. Pure Stage-B/cache-replay + analysis: reuses
the already-cached adaptive-threshold OOS dataset and v11's already-
validated forward-return builder. No HMM refit/redecode.

METHODOLOGICAL NOTE (flagged up front, not silently worked around): raw
HMM state INDICES (0,1,2,3) are NOT comparable across folds -- each fold's
HMM is independently refit from scratch (established repeatedly throughout
this project). Aggregating "state 0" across 93 folds would silently average
together unrelated regimes. This script aggregates by STATE LABEL
(Uptrend/Mid-Vol, Downtrend/Mid-Vol, Ranging/Low-Vol, Ranging/High-Vol --
each fold's adaptive threshold, exactly matching is_trending_t/trend_sign_t)
instead of raw index, which is the only fold-comparable dimension available.
This is the same substitution already made (and explained) in Steps 3 and 6.

Transitions are computed as consecutive-row label pairs WITHIN each fold's
own decoded sequence (never across a fold boundary, since that would compare
two unrelated models) and then aggregated by label-pair across folds, since
labels (not indices) are the comparable unit.

Run: `python v12_state_transition_analysis.py` (from src/). Expected: 1-2 min.
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_regime import RETURN_COLS, VOL_COLS, TREND_Z_THRESHOLD, VOL_Z_THRESHOLD
from v11_horizon_target_analysis import build_forward_returns, HORIZONS, HORIZON_NAMES

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_CACHE_DIR = os.path.join(_ROOT, "Data", "v2_cache")
CACHED_OOS = os.path.join(V2_CACHE_DIR, "trend_threshold_experiment", "adaptive_fold_threshold_master_oos.csv")
OUT_DIR = os.path.join(V2_CACHE_DIR, "trend_threshold_experiment", "state_transition_analysis")
MIN_N = 500  # minimum sample size to report a cell as interpretable


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_labels_per_fold_state(fold_dirs):
    """Same construction as v6_segment_breakdown.py: label per (fold, hmm_state)
    using EACH fold's own adaptive threshold (0.15 fallback or 0.30 default),
    read from already-fitted hmm.pkl (pickle.load only, no refit/decode)."""
    cached = pd.read_csv(CACHED_OOS, usecols=["fold", "is_trending_t", "trend_score"])
    fallback_folds = set(cached.loc[cached["is_trending_t"] & (cached["trend_score"].abs() <= 0.30) &
                                     (cached["trend_score"].abs() > 0.15), "fold"].unique())
    rows = []
    for d in fold_dirs:
        fold_path = os.path.join(V2_CACHE_DIR, d)
        meta_path = os.path.join(fold_path, "metadata.json")
        hmm_path = os.path.join(fold_path, "hmm.pkl")
        if not (os.path.exists(meta_path) and os.path.exists(hmm_path)):
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
            rows.append({"fold": meta["fold_id"], "hmm_state": s, "state_label": f"{trend_lbl}/{vol_lbl}",
                        "trend_sign": 1 if trend_lbl == "Uptrend" else (-1 if trend_lbl == "Downtrend" else 0)})
    return pd.DataFrame(rows)


def summarize_returns(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    if n == 0:
        return {"n": 0}
    return {"n": n, "mean": float(x.mean()), "median": float(x.median()), "std": float(x.std()),
            "se": float(x.std() / np.sqrt(n)) if n > 1 else None,
            "p5": float(x.quantile(0.05)), "p25": float(x.quantile(0.25)),
            "p75": float(x.quantile(0.75)), "p95": float(x.quantile(0.95)),
            "pct_positive": float((x > 0).mean() * 100), "pct_negative": float((x < 0).mean() * 100),
            "mean_abs": float(x.abs().mean())}


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log("Loading cached OOS data + fold labels (no HMM refit/decode)...")
    cached = pd.read_csv(CACHED_OOS, parse_dates=["timestamp"])
    fold_dirs = sorted(d for d in os.listdir(V2_CACHE_DIR) if d.startswith("fold_")
                        and os.path.exists(os.path.join(V2_CACHE_DIR, d, "DONE")))
    labels = load_labels_per_fold_state(fold_dirs)
    df = cached.merge(labels[["fold", "hmm_state", "state_label"]], on=["fold", "hmm_state"], how="left")

    log("Building forward returns at 5 horizons (reusing v11's validated builder)...")
    fwd = build_forward_returns()
    df = df.merge(fwd, on="timestamp", how="left")
    for h in HORIZONS:
        df[f"aligned_ret_{h}"] = df["trend_sign_t"] * df[f"fwd_ret_{h}"]
    log(f"  {len(df):,} total rows, {df['state_label'].nunique()} distinct labels: {sorted(df['state_label'].dropna().unique())}")

    # ============ PART 1: state (label) characterization ============
    log("\n=== PART 1: state/label characterization ===")
    part1 = df.groupby("state_label").agg(n=("state_label", "size")).reset_index()
    part1["pct_of_eligible"] = part1["n"] / len(df) * 100
    print(part1.to_string(index=False))

    # ============ PART 2 + 3: conditional & trend-aligned returns per label x horizon ============
    log("\n=== PART 2/3: conditional forward returns + trend-aligned returns per label x horizon ===")
    part2, part3 = {}, {}
    for label, g in df.groupby("state_label"):
        part2[label] = {}
        part3[label] = {}
        for h in HORIZONS:
            valid = g[g[f"fwd_valid_{h}"] == True]  # noqa: E712
            part2[label][HORIZON_NAMES[h]] = summarize_returns(valid[f"fwd_ret_{h}"])
            if (valid["trend_sign_t"] != 0).any():
                part3[label][HORIZON_NAMES[h]] = summarize_returns(valid.loc[valid["trend_sign_t"] != 0, f"aligned_ret_{h}"])
    for label in part2:
        row2h = part2[label].get("2h", {})
        row3h = part3[label].get("2h", {})
        log(f"  {label:>20}: raw_2h_mean={row2h.get('mean')}  aligned_2h_mean={row3h.get('mean')}  "
            f"aligned_2h_hitrate={row3h.get('pct_positive')}")

    # ============ PART 4/5: transitions (within-fold consecutive label pairs) ============
    log("\n=== PART 4: state transitions (label pairs, within-fold consecutive rows) ===")
    df_sorted = df.sort_values(["fold", "timestamp"]).reset_index(drop=True)
    same_fold_next = df_sorted["fold"] == df_sorted["fold"].shift(-1)
    from_label = df_sorted["state_label"]
    to_label = df_sorted["state_label"].shift(-1)
    trans = pd.DataFrame({"from": from_label, "to": to_label, "same_fold": same_fold_next}).iloc[:-1]
    valid_trans_mask = trans["same_fold"].values
    trans = trans[valid_trans_mask].reset_index(drop=True)
    # attach forward returns evaluated AT THE ARRIVAL bar (t+1, i.e. row index+1 in df_sorted)
    arrival_idx = np.where(valid_trans_mask)[0] + 1
    for h in HORIZONS:
        trans[f"fwd_ret_{h}"] = df_sorted.iloc[arrival_idx][f"fwd_ret_{h}"].values
        trans[f"fwd_valid_{h}"] = df_sorted.iloc[arrival_idx][f"fwd_valid_{h}"].values
    trans["to_trend_sign"] = df_sorted.iloc[arrival_idx]["trend_sign_t"].values

    total_trans = len(trans)
    trans_summary = []
    for (a, b), g in trans.groupby(["from", "to"]):
        row = {"from": a, "to": b, "count": len(g), "pct_of_total": len(g) / total_trans * 100}
        for h in [24]:  # headline horizon for the summary table; full detail saved to CSV
            valid = g[g[f"fwd_valid_{h}"] == True]  # noqa: E712
            stats_h = summarize_returns(valid[f"fwd_ret_{h}"])
            row[f"mean_ret_2h"] = stats_h.get("mean")
            row[f"pct_pos_2h"] = stats_h.get("pct_positive")
        trans_summary.append(row)
    trans_df = pd.DataFrame(trans_summary).sort_values("count", ascending=False)
    print(f"\nTotal within-fold consecutive-row transitions: {total_trans:,}")
    print(trans_df.to_string(index=False))

    # full multi-horizon detail per transition (for reliable-sample transitions only)
    trans_detail = []
    for (a, b), g in trans.groupby(["from", "to"]):
        if len(g) < MIN_N:
            continue
        row = {"from": a, "to": b, "count": len(g)}
        for h in HORIZONS:
            valid = g[g[f"fwd_valid_{h}"] == True]  # noqa: E712
            s = summarize_returns(valid[f"fwd_ret_{h}"])
            row[f"mean_ret_{HORIZON_NAMES[h]}"] = s.get("mean")
            row[f"pct_pos_{HORIZON_NAMES[h]}"] = s.get("pct_positive")
        trans_detail.append(row)
    trans_detail_df = pd.DataFrame(trans_detail)

    # ============ PART 5: transition type categories ============
    log("\n=== PART 5: transition type categories ===")
    def trend_of(lbl):
        if pd.isna(lbl):
            return None
        return "Uptrend" if lbl.startswith("Uptrend") else ("Downtrend" if lbl.startswith("Downtrend") else "Ranging")
    trans["from_trend"] = trans["from"].map(trend_of)
    trans["to_trend"] = trans["to"].map(trend_of)

    def categorize(row):
        if row["from"] == row["to"]:
            return "SAME_STATE"
        if row["from_trend"] == row["to_trend"] and row["from_trend"] in ("Uptrend", "Downtrend"):
            return "TREND_CONTINUATION"
        if row["from_trend"] in ("Uptrend", "Downtrend") and row["to_trend"] == "Ranging":
            return "TREND_TO_RANGE"
        if row["from_trend"] == "Ranging" and row["to_trend"] in ("Uptrend", "Downtrend"):
            return "RANGE_TO_TREND"
        if {row["from_trend"], row["to_trend"]} == {"Uptrend", "Downtrend"}:
            return "TREND_REVERSAL"
        return "OTHER"
    trans["category"] = trans.apply(categorize, axis=1)

    part5 = {}
    for cat, g in trans.groupby("category"):
        row = {"n": len(g), "pct_of_total": len(g) / total_trans * 100}
        for h in HORIZONS:
            valid = g[g[f"fwd_valid_{h}"] == True]  # noqa: E712
            s = summarize_returns(valid[f"fwd_ret_{h}"])
            row[f"mean_ret_{HORIZON_NAMES[h]}"] = s.get("mean")
            row[f"pct_pos_{HORIZON_NAMES[h]}"] = s.get("pct_positive")
        part5[cat] = row
        log(f"  {cat:>20}: n={row['n']:>7,} ({row['pct_of_total']:.1f}%)  mean_2h={row.get('mean_ret_2h'):.6f}  pct_pos_2h={row.get('pct_pos_2h'):.2f}%")

    runtime = time.time() - t0

    json.dump({"part1": part1.to_dict(orient="records"), "part2": part2, "part3": part3,
               "part5_categories": part5, "runtime_s": runtime},
              open(os.path.join(OUT_DIR, "step8_parts1_3_5.json"), "w"), indent=2, default=str)
    trans_df.to_csv(os.path.join(OUT_DIR, "part4_transition_summary.csv"), index=False)
    trans_detail_df.to_csv(os.path.join(OUT_DIR, "part4_transition_detail_multihorizon.csv"), index=False)
    trans.to_csv(os.path.join(OUT_DIR, "transitions_raw.csv"), index=False)
    df.to_csv(os.path.join(OUT_DIR, "labeled_data_with_forward_returns.csv"), index=False)

    log(f"\nRuntime so far: {runtime:.1f}s. Saved to {OUT_DIR}")
    return df, trans, part5, trans_detail_df


if __name__ == "__main__":
    main()
