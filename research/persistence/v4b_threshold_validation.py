"""
v4b_threshold_validation.py -- validates the ±0.15 trend-threshold candidate
(surfaced in V4 by comparing thresholds on the final OOS table, which is
NOT a leakage-safe way to pick a production value) using a proper inner
train/validation split inside an early calibration window only -- same
discipline as V3's k-selection. The final V4 OOS numbers are never touched
here; this is an independent confirmation check.

Calibration window: folds 0-20 (2018-03 to 2019-11), same window V3 used.
Split: chronological 70/30 inner-train/inner-val within that window.
Candidates: 0.30 (current baseline), 0.15 (V4's candidate), plus 0.10/0.20
for context. Epsilon frozen from inner-train only, exactly like every other
epsilon in this project.

Reuses v4_threshold_sensitivity.py's classify_at_threshold() and the
already-cached HMM outputs -- zero HMM computation here either.

Run: `python v4b_threshold_validation.py` (from src/).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_threshold_sensitivity as v4

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = v4.OUT_DIR
CALIBRATION_FOLD_CUTOFF = 10  # folds 0-10 only: the ORIGINAL folds 0-20 window was tried first and
# rejected -- it overlaps almost entirely with the exact 2019-02+ dead zone this experiment is
# testing, so the +-0.30 baseline produced ZERO inner-val rows there (informative on its own, but
# unusable for a fair head-to-head AUC comparison). Folds 0-10 (2018-03 to 2019-01) predate that
# zone entirely, so both thresholds have real data to compare on equal footing.
CANDIDATES = [0.10, 0.15, 0.20, 0.30]
SPLIT_FRAC = 0.70


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    log("Loading cached state scores + causal outputs (no HMM computation)...")
    fold_dirs = sorted(d for d in os.listdir(v4.V2_CACHE_DIR) if d.startswith("fold_")
                        and os.path.exists(os.path.join(v4.V2_CACHE_DIR, d, "DONE")))
    state_scores = v4.load_fold_state_scores(fold_dirs)
    causal = v4.load_all_causal_outputs(fold_dirs)

    calib = causal[causal["fold"] <= CALIBRATION_FOLD_CUTOFF].sort_values("timestamp").reset_index(drop=True)
    n = len(calib)
    split = int(n * SPLIT_FRAC)
    train_end_ts = calib.iloc[split - 1]["timestamp"]
    log(f"Calibration window: folds 0-{CALIBRATION_FOLD_CUTOFF} ({calib['timestamp'].min()} -> {calib['timestamp'].max()}), "
        f"{n:,} total rows. Inner-train/inner-val split at {train_end_ts} ({SPLIT_FRAC:.0%}/{1-SPLIT_FRAC:.0%}).")

    results = []
    for threshold in CANDIDATES:
        is_trending = calib["trend_score"].abs() > threshold
        trend_sign = np.where(is_trending, np.sign(calib["trend_score"]), 0.0)
        df = calib.copy()
        df["is_trending_t"] = is_trending
        df["trend_sign_t"] = trend_sign

        inner_train = df.iloc[:split]
        inner_val = df.iloc[split:]

        tr_trending_fwd = inner_train[(inner_train["is_trending_t"]) & (inner_train["forward_valid"])]
        if len(tr_trending_fwd) < 20:
            results.append({"threshold": threshold, "status": "insufficient inner-train data", "inner_train_eligible": len(tr_trending_fwd)})
            continue
        epsilon = float(np.percentile(tr_trending_fwd["forward_ret_24"].abs(), v4.EPSILON_PERCENTILE))

        def label_usable(part):
            usable = part["is_trending_t"] & part["forward_valid"] & (part["forward_ret_24"].abs() > epsilon)
            label = np.where(np.sign(part["forward_ret_24"]) == part["trend_sign_t"], 1, 0)
            return usable, np.where(usable, label, np.nan)

        tr_usable, tr_label = label_usable(inner_train)
        va_usable, va_label = label_usable(inner_val)

        tr_elig = inner_train[tr_usable].copy(); tr_elig["label"] = tr_label[tr_usable]
        va_elig = inner_val[va_usable].copy(); va_elig["label"] = va_label[va_usable]

        n_zero_fold_in_window = int(df.groupby("fold")["is_trending_t"].sum().eq(0).sum())

        if len(tr_elig) < 50 or tr_elig["label"].nunique() < 2 or len(va_elig) < 20 or va_elig["label"].nunique() < 2:
            results.append({"threshold": threshold, "status": "insufficient labeled data for a fair AUC comparison",
                            "inner_train_eligible": len(tr_elig), "inner_val_eligible": len(va_elig),
                            "zero_eligible_folds_in_window": n_zero_fold_in_window})
            continue

        lr = LogisticRegression(max_iter=1000)
        lr.fit(tr_elig[v4.FEATURE_COLUMNS].values, tr_elig["label"].astype(int).values)
        p_val = lr.predict_proba(va_elig[v4.FEATURE_COLUMNS].values)[:, 1]
        auc = float(roc_auc_score(va_elig["label"].astype(int).values, p_val))

        results.append({
            "threshold": threshold, "status": "ok", "epsilon": epsilon,
            "inner_train_eligible": len(tr_elig), "inner_val_eligible": len(va_elig),
            "zero_eligible_folds_in_window": n_zero_fold_in_window,
            "inner_val_auc": auc, "inner_val_pos_rate": float(va_elig["label"].mean()),
        })
        log(f"  threshold=+/-{threshold}: inner_train={len(tr_elig):,} inner_val={len(va_elig):,} "
            f"zero_folds_in_window={n_zero_fold_in_window}  inner_val_AUC={auc:.4f}")

    res_df = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, "threshold_015_validation.csv")
    res_df.to_csv(out_path, index=False)
    print()
    print("=" * 70)
    print("VALIDATION RESULT (inner train/val, calibration window only -- never the final OOS set)")
    print("=" * 70)
    print(res_df.to_string(index=False))
    print()
    print(f"Saved to: {out_path}")

    baseline_row = res_df[res_df["threshold"] == 0.30]
    candidate_row = res_df[res_df["threshold"] == 0.15]
    if len(baseline_row) and len(candidate_row) and baseline_row.iloc[0]["status"] == "ok" and candidate_row.iloc[0]["status"] == "ok":
        b_auc, c_auc = baseline_row.iloc[0]["inner_val_auc"], candidate_row.iloc[0]["inner_val_auc"]
        b_zero, c_zero = baseline_row.iloc[0]["zero_eligible_folds_in_window"], candidate_row.iloc[0]["zero_eligible_folds_in_window"]
        print(f"\nVERDICT: baseline(+/-0.30) inner-val AUC={b_auc:.4f}, zero-folds-in-window={b_zero}")
        print(f"         candidate(+/-0.15) inner-val AUC={c_auc:.4f}, zero-folds-in-window={c_zero}")
        if c_auc >= b_auc - 0.01 and c_zero <= b_zero:
            print("         -> +/-0.15 VALIDATES: no meaningful AUC cost, recovers coverage. Safe to promote.")
        elif c_auc < b_auc - 0.01:
            print("         -> +/-0.15 DOES NOT CLEANLY VALIDATE: inner-val AUC dropped more than a trivial amount. Do not promote without further review.")
        else:
            print("         -> Mixed result, review before promoting.")


if __name__ == "__main__":
    main()
