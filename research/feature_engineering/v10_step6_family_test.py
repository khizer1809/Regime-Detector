"""
v10_step6_family_test.py -- Step 6 Part 1: test each new Trend feature
family independently against the baseline. Pure Stage-B/cache-replay:
merges v9's already-built trend_features.csv onto the already-cached
adaptive-threshold OOS dataset. No HMM computation.

Run: `python v10_step6_family_test.py` (from src/).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v8_raw_vs_hmm_features import run_walkforward, summarize, HMM_FEATURES
from v9_trend_features import FEATURE_FAMILIES, ALL_NEW_FEATURES

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHED_OOS = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "adaptive_fold_threshold_master_oos.csv")
TREND_FEATS = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "trend_features", "trend_features.csv")
OUT_DIR = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "trend_feature_test")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log("Loading cached OOS dataset + new trend features (no HMM computation)...")
    cached = pd.read_csv(CACHED_OOS, parse_dates=["timestamp"])
    trend = pd.read_csv(TREND_FEATS, parse_dates=["timestamp"])
    df = cached.merge(trend, on="timestamp", how="left")
    missing = df[ALL_NEW_FEATURES].isna().any(axis=1).sum()
    log(f"  merged: {len(df):,} rows, {missing:,} rows with any NaN in new features (warm-up rows, will be dropped by usable&notna)")

    # rows must also have complete new-feature data to be usable for models B-G
    df["usable_orig"] = df["usable"]
    for fam_cols in FEATURE_FAMILIES.values():
        pass  # per-model usability handled inside run_walkforward via NaN rows failing LR fit; we filter explicitly below

    results = {}
    audits = {}

    log("\n=== MODEL A: baseline (4 HMM features) ===")
    oos_y, oos_p, fold_results, audit, _ = run_walkforward(df, HMM_FEATURES, scale=False)
    results["A_baseline"] = summarize(oos_y, oos_p, fold_results, HMM_FEATURES, "A_baseline")
    audits["A_baseline"] = audit
    log(f"  A: OOS_AUC={results['A_baseline']['roc_auc']:.4f} fold_mean={results['A_baseline']['fold_auc_mean']:.4f}")

    model_letter = {"family1_structure": "B", "family2_efficiency": "C", "family3_acceleration": "D",
                    "family4_mtf_alignment": "E", "family5_trend_age": "F", "family6_deterioration": "G"}

    for fam, cols in FEATURE_FAMILIES.items():
        letter = model_letter[fam]
        feats = HMM_FEATURES + cols
        # drop rows with NaN in the new family's columns for THIS model only (warm-up rows)
        sub = df.dropna(subset=cols).copy()
        log(f"\n=== MODEL {letter}: baseline + {fam} ({cols}) -- {len(sub):,} rows after dropping warm-up NaNs ===")
        oos_y, oos_p, fold_results, audit, coefs = run_walkforward(sub, feats, scale=True)
        name = f"{letter}_{fam}"
        results[name] = summarize(oos_y, oos_p, fold_results, feats, name)
        audits[name] = audit
        r = results[name]
        log(f"  {letter}: OOS_AUC={r['roc_auc']:.4f} fold_mean={r['fold_auc_mean']:.4f} fold_std={r['fold_auc_std']:.4f} "
            f"train_auc={r['mean_train_auc']:.4f} gap={r['train_oos_gap']:.4f} folds>0.5={r['folds_above_0.5']}/{r['n_folds']}")

    runtime = time.time() - t0

    baseline = results["A_baseline"]
    delta_rows = []
    for fam, cols in FEATURE_FAMILIES.items():
        letter = model_letter[fam]
        name = f"{letter}_{fam}"
        r = results[name]
        delta_rows.append({
            "model": name, "family": fam, "delta_pooled_auc": r["roc_auc"] - baseline["roc_auc"],
            "delta_fold_mean_auc": r["fold_auc_mean"] - baseline["fold_auc_mean"],
            "delta_folds_above_0.5": r["folds_above_0.5"] - baseline["folds_above_0.5"],
            "pooled_auc": r["roc_auc"], "fold_mean_auc": r["fold_auc_mean"], "fold_std": r["fold_auc_std"],
            "train_oos_gap": r["train_oos_gap"],
        })
    delta_df = pd.DataFrame(delta_rows)

    print("\n" + "=" * 70)
    print("STEP 6 PART 1 -- FAMILY-LEVEL RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2, default=str))
    print("\n--- DELTA TABLE (family model - baseline) ---")
    print(delta_df.to_string(index=False))
    print("\nLeakage audits:")
    for k, a in audits.items():
        print(f"  {k}: violations={a['violations']}")

    json.dump({"results": results, "runtime_s": runtime}, open(os.path.join(OUT_DIR, "part1_family_results.json"), "w"), indent=2, default=str)
    delta_df.to_csv(os.path.join(OUT_DIR, "part1_delta_table.csv"), index=False)
    json.dump(audits, open(os.path.join(OUT_DIR, "part1_leakage_audit.json"), "w"), indent=2, default=str)
    log(f"\nRuntime: {runtime:.1f}s. Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
