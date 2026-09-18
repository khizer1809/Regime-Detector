"""
v17_rf_lr_structure_features.py -- EXPERIMENTAL, NOT PRODUCTION.

Extends v16_rf_vs_lr_experiment.py's RF-vs-LR comparison with the 15 causal
market-structure features (k=3 -- the SAME fixed k already selected/
validated in the earlier market-structure experiment, not reselected here),
on top of the SAME frozen, unmodified HMM-4 cache.

REUSED, UNMODIFIED:
  - Data/v2_cache/trend_threshold_experiment/adaptive_fold_threshold_master_oos.csv
    (frozen HMM-4 baseline, same as v16).
  - research_archive/Data/hmm_lr_abc_experiment/features_expanded_33.csv's
    15 structure-feature columns (k=3, already computed and validated in
    v14_market_structure_test.py / v15's expanded-feature build -- NOT
    recomputed, NOT re-selected here).

NEW: runs BOTH LR and RF, in the SAME script/walk-forward loop (so both see
identical pools/folds/features -- unlike v16, which reused an
already-cached LR column computed by a different script with its own
scaling convention; here LR is re-run fresh alongside RF specifically to
keep the 19-feature comparison apples-to-apples within one code path).

FEATURE SETS compared:
    baseline (4):  confidence, stay_prob, log1p_duration, margin
    structure (19): baseline (4) + 15 market-structure features

LR uses StandardScaler (fit on pool only, standard practice for LR); RF
does not (tree-invariant to monotonic per-feature transforms).

Target/walk-forward/embargo: unchanged, identical to v16.

Run: `python v17_rf_lr_structure_features.py` (from research_archive/src/).
Writes to research_archive/Data/rf_vs_lr_experiment/structure_features/.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

BASELINE_MASTER_PATH = os.path.join(_RESEARCH_ROOT, "Data", "v2_cache", "trend_threshold_experiment",
                                     "adaptive_fold_threshold_master_oos.csv")
STRUCTURE_FEATURES_PATH = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "features_expanded_33.csv")
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
GATE_THRESHOLD = 0.5
SEED = 42
RF_PARAMS = dict(n_estimators=300, max_depth=8, min_samples_leaf=50, min_samples_split=100,
                  max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_walkforward(master, feature_columns, tag):
    folds = sorted(master["fold"].unique())
    pool_frames = []
    rf_probs = np.full(len(master), np.nan)
    lr_probs = np.full(len(master), np.nan)
    fold_metrics = []
    master_by_fold_idx = {f: master.index[master["fold"] == f] for f in folds}

    pbar = tqdm(folds, desc=f"{tag} ({len(feature_columns)} feats)")
    for fold_id in pbar:
        fold_idx = master_by_fold_idx[fold_id]
        fold_rows = master.loc[fold_idx]
        pool = pd.concat(pool_frames, ignore_index=False) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712
        pool_fit = pool_usable.dropna(subset=feature_columns) if len(pool_usable) else pool_usable

        if len(pool_fit) < MIN_POOL_ROWS:
            fold_metrics.append({"fold": int(fold_id), "pool_train_rows": 0})
            pool_frames.append(fold_rows)
            pbar.set_postfix_str(f"fold {fold_id} bootstrap-skip")
            continue

        X_train = pool_fit[feature_columns].values
        y_train = pool_fit["label"].astype(int).values

        # class_weight rule must be IDENTICAL for RF and LR -- only balance when
        # the pool is actually skewed. Previously RF was hard-coded "balanced"
        # always, while LR only balanced when skewed -- a real asymmetry (for RF,
        # class_weight changes the impurity criterion used at every split, not
        # just a final rescaling, so this was not a cosmetic difference). Fixed
        # here: both models now share one class-weight decision per fold.
        pos_rate = float(y_train.mean())
        cw = "balanced" if not (0.4 <= pos_rate <= 0.6) else None

        rf_params = {**RF_PARAMS, "class_weight": cw}
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        lr = LogisticRegression(max_iter=1000, class_weight=cw)
        lr.fit(X_train_scaled, y_train)

        trending_mask = (fold_rows["is_trending_t"] == True) & fold_rows[feature_columns].notna().all(axis=1)  # noqa: E712
        X_pred = fold_rows.loc[trending_mask, feature_columns].values
        if len(X_pred):
            idx_pos = master.index.get_indexer(fold_rows.index[trending_mask])
            rf_probs[idx_pos] = rf.predict_proba(X_pred)[:, 1]
            lr_probs[idx_pos] = lr.predict_proba(scaler.transform(X_pred))[:, 1]

        eligible = fold_rows[(fold_rows["usable"] == True) & trending_mask]  # noqa: E712
        fm = {"fold": int(fold_id), "pool_train_rows": len(X_train), "n_eligible": len(eligible)}
        if len(eligible) > 5 and eligible["label"].nunique() > 1:
            idx_e = master.index.get_indexer(eligible.index)
            fm["rf_fold_auc"] = float(roc_auc_score(eligible["label"].astype(int), rf_probs[idx_e]))
            fm["lr_fold_auc"] = float(roc_auc_score(eligible["label"].astype(int), lr_probs[idx_e]))
        fold_metrics.append(fm)
        pbar.set_postfix_str(f"fold {fold_id} n={len(X_train):,} rf_auc={fm.get('rf_fold_auc', float('nan')):.3f} "
                              f"lr_auc={fm.get('lr_fold_auc', float('nan')):.3f}")

        pool_frames.append(fold_rows)

    master[f"{tag}_rf_probability"] = rf_probs
    master[f"{tag}_lr_probability"] = lr_probs
    return master, fold_metrics


def metrics_block(y_true, y_prob):
    y_pred = (y_prob >= GATE_THRESHOLD).astype(int)
    return {
        "n": len(y_true), "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])) if len(set(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
    }


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    log("Loading frozen HMM-4 baseline + 15 structure features (k=3, reused verbatim)...")
    master = pd.read_csv(BASELINE_MASTER_PATH, parse_dates=["timestamp"])
    struct = pd.read_csv(STRUCTURE_FEATURES_PATH, usecols=["timestamp"] + STRUCTURE_FEATURE_COLUMNS,
                          parse_dates=["timestamp"])
    master = master.merge(struct, on="timestamp", how="left")
    log(f"  {len(master):,} rows after merge")

    log("=== BASELINE (4 features) -- RF and LR, same harness ===")
    master, fm_base = run_walkforward(master.copy(), BASE_FEATURE_COLUMNS, "base")

    log("=== STRUCTURE (19 features: base 4 + 15 market-structure, k=3) -- RF and LR, same harness ===")
    master, fm_struct = run_walkforward(master, FULL_FEATURE_COLUMNS, "struct")

    runtime = time.time() - t0
    log(f"Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    results = {}
    for tag, feature_columns in [("base", BASE_FEATURE_COLUMNS), ("struct", FULL_FEATURE_COLUMNS)]:
        eligible = master[(master["usable"] == True) & master[f"{tag}_rf_probability"].notna() &
                           master[f"{tag}_lr_probability"].notna()]  # noqa: E712
        y = eligible["label"].astype(int).values
        results[f"{tag}_rf"] = metrics_block(y, eligible[f"{tag}_rf_probability"].values)
        results[f"{tag}_lr"] = metrics_block(y, eligible[f"{tag}_lr_probability"].values)

    comp_df = pd.DataFrame(results).T
    comp_df.to_csv(os.path.join(OUT_DIR, "base_vs_structure_rf_lr_comparison.csv"))
    log("\n" + comp_df.to_string())

    master.to_csv(os.path.join(OUT_DIR, "predictions_all.csv"), index=False)
    pd.DataFrame(fm_base).to_csv(os.path.join(OUT_DIR, "fold_results_base.csv"), index=False)
    pd.DataFrame(fm_struct).to_csv(os.path.join(OUT_DIR, "fold_results_struct.csv"), index=False)
    json.dump({"results": results, "runtime_s": runtime, "feature_columns_struct": FULL_FEATURE_COLUMNS},
              open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2, default=str)
    log(f"All artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
