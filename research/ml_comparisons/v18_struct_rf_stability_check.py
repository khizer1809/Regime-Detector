"""
v18_struct_rf_stability_check.py -- EXPERIMENTAL, NOT PRODUCTION.

Stability/overfitting verification for struct_rf (Random Forest, 19
features -- 4 baseline HMM-derived + 15 market-structure), the current
pooled-AUC leader from v17's corrected (symmetric class_weight) run. Mirrors
the same scrutiny already applied to the A/B/C expanded-HMM experiment
before trusting its headline number: year-by-year stability + train-vs-OOS
overfitting gap.

Reuses v17's already-computed predictions (predictions_all.csv) for the
year-by-year check -- no retraining needed. Re-runs ONLY struct_rf's
walk-forward (identical config to v17's fixed run) to additionally capture
per-fold TRAIN AUC, which v17 didn't save.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRED_PATH = os.path.join(_RESEARCH_ROOT, "Data", "rf_vs_lr_experiment", "structure_features", "predictions_all.csv")
OUT_DIR = os.path.join(_RESEARCH_ROOT, "Data", "rf_vs_lr_experiment", "structure_features")

BASE_FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]
STRUCTURE_FEATURE_COLUMNS = [
    "hh_strength", "hl_strength", "lh_strength", "ll_strength",
    "structure_direction", "structure_consistency", "bos_up", "bos_down",
    "choch_up", "choch_down", "pullback_ratio", "impulse_strength",
    "distance_from_last_swing_high", "distance_from_last_swing_low", "trend_structure_age",
]
FULL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + STRUCTURE_FEATURE_COLUMNS
MIN_POOL_ROWS = 500
RF_BASE_PARAMS = dict(n_estimators=300, max_depth=8, min_samples_leaf=50, min_samples_split=100,
                       max_features="sqrt", random_state=42, n_jobs=-1)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def year_stability(master):
    m = master.copy()
    m["year"] = pd.to_datetime(m["timestamp"]).dt.year
    eligible = m[m["usable"] == True]  # noqa: E712
    rows = []
    for year, sub in eligible.groupby("year"):
        row = {"year": int(year), "n": len(sub)}
        for tag in ["base_lr", "base_rf", "struct_lr", "struct_rf"]:
            col = f"{tag}_probability"
            valid = sub[sub[col].notna()]
            if len(valid) > 20 and valid["label"].nunique() > 1:
                row[tag] = float(roc_auc_score(valid["label"].astype(int), valid[col]))
            else:
                row[tag] = None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")


def train_oos_gap_struct_rf(master):
    folds = sorted(master["fold"].unique())
    pool_frames = []
    fold_results = []
    master_by_fold_idx = {f: master.index[master["fold"] == f] for f in folds}

    for fold_id in tqdm(folds, desc="struct_rf train-vs-OOS check"):
        fold_idx = master_by_fold_idx[fold_id]
        fold_rows = master.loc[fold_idx]
        pool = pd.concat(pool_frames, ignore_index=False) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712
        pool_fit = pool_usable.dropna(subset=FULL_FEATURE_COLUMNS) if len(pool_usable) else pool_usable

        if len(pool_fit) < MIN_POOL_ROWS:
            pool_frames.append(fold_rows)
            continue

        X_train = pool_fit[FULL_FEATURE_COLUMNS].values
        y_train = pool_fit["label"].astype(int).values
        pos_rate = float(y_train.mean())
        cw = "balanced" if not (0.4 <= pos_rate <= 0.6) else None

        rf = RandomForestClassifier(**{**RF_BASE_PARAMS, "class_weight": cw})
        rf.fit(X_train, y_train)
        train_probs = rf.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_probs) if len(set(y_train)) > 1 else None

        trending_mask = (fold_rows["is_trending_t"] == True) & fold_rows[FULL_FEATURE_COLUMNS].notna().all(axis=1)  # noqa: E712
        eligible = fold_rows[(fold_rows["usable"] == True) & trending_mask]  # noqa: E712
        oos_auc = None
        if len(eligible) > 5 and eligible["label"].nunique() > 1:
            X_pred = eligible[FULL_FEATURE_COLUMNS].values
            probs = rf.predict_proba(X_pred)[:, 1]
            oos_auc = float(roc_auc_score(eligible["label"].astype(int), probs))

        fold_results.append({"fold": int(fold_id), "n_train": len(X_train), "n_oos_eligible": len(eligible),
                              "train_auc": train_auc, "oos_auc": oos_auc})
        pool_frames.append(fold_rows)

    return pd.DataFrame(fold_results)


def main():
    log("Loading v17's saved predictions for year-by-year stability check...")
    master = pd.read_csv(PRED_PATH, parse_dates=["timestamp"])

    log("=== Year-by-year AUC, all 4 configs ===")
    stability_df = year_stability(master)
    log("\n" + stability_df.to_string(index=False))
    stability_df.to_csv(os.path.join(OUT_DIR, "year_stability_all_configs.csv"), index=False)

    log("=== struct_rf train-vs-OOS overfitting check (per fold) ===")
    gap_df = train_oos_gap_struct_rf(master)
    mean_train = gap_df["train_auc"].dropna().mean()
    mean_oos = gap_df["oos_auc"].dropna().mean()
    log(f"\nMean per-fold train AUC: {mean_train:.4f}")
    log(f"Mean per-fold OOS AUC:   {mean_oos:.4f}")
    log(f"Train-OOS gap:           {mean_train - mean_oos:+.4f}")
    gap_df.to_csv(os.path.join(OUT_DIR, "struct_rf_train_oos_gap.csv"), index=False)

    log(f"\nAll artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
