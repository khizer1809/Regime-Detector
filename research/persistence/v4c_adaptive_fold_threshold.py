"""
v4c_adaptive_fold_threshold.py -- period-adaptive trend threshold:
    +/-0.30 for a fold, UNLESS that fold has zero eligible trending states
    at +/-0.30, in which case fall back to +/-0.15 for that fold only.

This is NOT a blanket threshold swap (V4 tested that and it failed
validation: +/-0.30 outperforms +/-0.15 by ~1.1pp AUC when both have data).
Here, the tighter, better-performing +/-0.30 threshold is kept everywhere
it already works; +/-0.15 only substitutes in for the folds that would
otherwise contribute nothing at all.

Zero new HMM computation -- reuses v4_threshold_sensitivity.py's cached
state scores and causal outputs, and its exact walk-forward/epsilon/LR
methodology, just with a per-fold (not global) threshold assignment.

Run: `python v4c_adaptive_fold_threshold.py` (from src/).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_threshold_sensitivity as v4

OUT_DIR = v4.OUT_DIR
PRIMARY_THRESHOLD = 0.30
FALLBACK_THRESHOLD = 0.15


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("Loading cached state scores + causal outputs (no HMM computation)...")
    fold_dirs = sorted(d for d in os.listdir(v4.V2_CACHE_DIR) if d.startswith("fold_")
                        and os.path.exists(os.path.join(v4.V2_CACHE_DIR, d, "DONE")))
    state_scores = v4.load_fold_state_scores(fold_dirs)
    causal = v4.load_all_causal_outputs(fold_dirs)

    classified_primary = v4.classify_at_threshold(state_scores, PRIMARY_THRESHOLD)
    fallback_folds = set(classified_primary[classified_primary["zero_eligible_fold"]]["fold"].tolist())
    log(f"Folds using primary +/-{PRIMARY_THRESHOLD}: {len(classified_primary) - len(fallback_folds)}")
    log(f"Folds falling back to +/-{FALLBACK_THRESHOLD} (zero-eligible at primary): {len(fallback_folds)} -> {sorted(fallback_folds)}")

    # per-row threshold = fallback threshold if that row's fold is in fallback_folds, else primary
    fold_threshold = causal["fold"].map(lambda f: FALLBACK_THRESHOLD if f in fallback_folds else PRIMARY_THRESHOLD)
    is_trending_row = causal["trend_score"].abs() > fold_threshold
    trend_sign_row = np.where(is_trending_row, np.sign(causal["trend_score"]), 0.0)

    classified_fallback_check = v4.classify_at_threshold(
        state_scores[state_scores["fold"].isin(fallback_folds)], FALLBACK_THRESHOLD)
    still_zero_after_fallback = classified_fallback_check[classified_fallback_check["zero_eligible_fold"]]["fold"].tolist()
    log(f"Of those {len(fallback_folds)} fallback folds, still zero-eligible even at +/-{FALLBACK_THRESHOLD}: "
        f"{len(still_zero_after_fallback)} -> {still_zero_after_fallback}")

    log("Running V2-style expanding walk-forward LR with the adaptive per-fold threshold...")
    master, fold_metrics, audit_rows = v4.run_v2_style_walkforward(causal, trend_sign_row, is_trending_row, "adaptive")

    labeled = master.dropna(subset=["label"])
    eligible = master[(master["usable"]) & master["lr_probability"].notna()]
    metrics = v4.compute_metrics_block(eligible) if len(eligible) > 10 and eligible["label"].nunique() > 1 else {}
    fold_aucs = [fm["fold_auc"] for fm in fold_metrics if "fold_auc" in fm]
    zero_folds_final = sum(1 for fm in fold_metrics if fm.get("n_eligible", 0) == 0)

    uptrend_rows = int((is_trending_row & (trend_sign_row == 1)).sum())
    downtrend_rows = int((is_trending_row & (trend_sign_row == -1)).sum())
    ranging_rows = int((~is_trending_row).sum())

    runtime = time.time() - t0
    log(f"Total runtime: {runtime:.1f}s")

    result = {
        "primary_threshold": PRIMARY_THRESHOLD, "fallback_threshold": FALLBACK_THRESHOLD,
        "folds_on_primary": len(classified_primary) - len(fallback_folds),
        "folds_on_fallback": len(fallback_folds),
        "still_zero_after_fallback": still_zero_after_fallback,
        "zero_eligible_folds_final": zero_folds_final,
        "eligible_rows": len(labeled), "uptrend_rows": uptrend_rows, "downtrend_rows": downtrend_rows,
        "ranging_rows": ranging_rows, "coverage_pct": float(len(labeled) / len(causal) * 100),
        "positive_pct": float((labeled["label"] == 1).mean() * 100) if len(labeled) else None,
        "metrics": metrics,
        "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "fold_auc_median": float(np.median(fold_aucs)) if fold_aucs else None,
        "fold_auc_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "folds_above_0.5": int(sum(1 for a in fold_aucs if a > 0.5)),
        "folds_below_0.5": int(sum(1 for a in fold_aucs if a < 0.5)),
        "runtime_s": runtime,
    }
    json.dump(result, open(os.path.join(OUT_DIR, "adaptive_fold_threshold_result.json"), "w"), indent=2, default=str)
    master.to_csv(os.path.join(OUT_DIR, "adaptive_fold_threshold_master_oos.csv"), index=False)

    print()
    print("=" * 70)
    print("ADAPTIVE FOLD-THRESHOLD RESULT")
    print("=" * 70)
    print(json.dumps({k: v for k, v in result.items() if k not in ("metrics",)}, indent=2, default=str))
    print("metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
