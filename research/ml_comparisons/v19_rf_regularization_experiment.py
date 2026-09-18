"""
v19_rf_regularization_experiment.py -- EXPERIMENTAL, NOT PRODUCTION.

Controlled RF regularization grid: can a more-regularized Random Forest
close the severe train->OOS overfitting gap found in the current RF
(struct_rf: 19 features -- 4 HMM-derived + 15 market-structure, depth=8,
min_samples_leaf=50, min_samples_split=100, pooled OOS AUC=0.5232, train
AUC=0.6882, gap=+0.178) while maintaining or improving OOS performance?

REUSED, UNMODIFIED (identical to the RF-vs-LR experiment this extends):
  - Data/v2_cache/trend_threshold_experiment/adaptive_fold_threshold_master_oos.csv
    (frozen HMM-4 baseline -- target/epsilon/usable/label/fold all reused verbatim).
  - research_archive/Data/hmm_lr_abc_experiment/features_expanded_33.csv's 15
    structure-feature columns (k=3, unchanged).
  - v2_common.generate_folds() fold schedule (baked into `fold` column).
  - The class_weight rule fixed after the earlier flaw was found: "balanced"
    only if the fold's own training pool is skewed outside [0.4, 0.6],
    identical for every RF config here (never hard-coded "balanced").

REUSED RESULTS (not recomputed, to avoid a redundant 8th full walk-forward):
  - base_lr / struct_lr / struct_rf(current, unregularized) pooled+fold-level
    results, loaded directly from the already-completed and-verified
    research_archive/Data/rf_vs_lr_experiment/structure_features/ outputs.

NEW: 7 regularization configurations (RF-1..RF-7, all n_estimators=300,
random_state=42, matching the existing experiment's fixed values), each run
through the IDENTICAL walk-forward mechanics (same folds, same features,
same target, same class_weight rule) -- only the RF hyperparameters differ.

Run: `python v19_rf_regularization_experiment.py` (from research_archive/src/).
Writes to research_archive/Data/rf_regularization_experiment/ (does not
overwrite the earlier rf_vs_lr_experiment/ outputs).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                              log_loss, precision_score, recall_score, roc_auc_score)
from tqdm import tqdm

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

BASELINE_MASTER_PATH = os.path.join(_RESEARCH_ROOT, "Data", "v2_cache", "trend_threshold_experiment",
                                     "adaptive_fold_threshold_master_oos.csv")
STRUCTURE_FEATURES_PATH = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "features_expanded_33.csv")
PRIOR_EXP_DIR = os.path.join(_RESEARCH_ROOT, "Data", "rf_vs_lr_experiment", "structure_features")
OUT_DIR = os.path.join(_RESEARCH_ROOT, "Data", "rf_regularization_experiment")

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
N_ESTIMATORS = 300  # matches the existing experiment's fixed value

RF_CONFIGS = {
    "RF-1": dict(max_depth=3, min_samples_leaf=100, min_samples_split=200, max_features="sqrt"),
    "RF-2": dict(max_depth=5, min_samples_leaf=100, min_samples_split=200, max_features="sqrt"),
    "RF-3": dict(max_depth=5, min_samples_leaf=250, min_samples_split=500, max_features="sqrt"),
    "RF-4": dict(max_depth=8, min_samples_leaf=100, min_samples_split=200, max_features="sqrt"),
    "RF-5": dict(max_depth=8, min_samples_leaf=250, min_samples_split=500, max_features="sqrt"),
    "RF-6": dict(max_depth=5, min_samples_leaf=100, min_samples_split=200, max_features=0.5),
    "RF-7": dict(max_depth=8, min_samples_leaf=250, min_samples_split=500, max_features=0.5),
}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def print_inspection_summary():
    log("=" * 78)
    log("INSPECTION (reusing prior RF-vs-LR / A-B-C infrastructure verbatim)")
    log("=" * 78)
    log("HMM cache: Data/v2_cache/fold_XXX/ (frozen HMM-4, untouched)")
    log("Baseline master: Data/v2_cache/trend_threshold_experiment/adaptive_fold_threshold_master_oos.csv")
    log("Structure features: research_archive/Data/hmm_lr_abc_experiment/features_expanded_33.csv (k=3, unchanged)")
    log("Existing RF/LR experiment: research_archive/src/v17_rf_lr_structure_features.py "
        "(current struct_rf: depth=8, leaf=50, split=100 -- this is the model being regularized here)")
    log("Fold schedule: v2_common.generate_folds() -- 101 folds, expanding, 48-bar embargo (unchanged)")
    log("Target: sign(forward_ret_24)==trend_sign, gated by usable (unchanged, reused verbatim)")
    log("class_weight rule: 'balanced' only if fold pool skewed outside [0.4,0.6] -- same rule now used "
        "consistently for RF and LR since the earlier asymmetry fix")
    log("=" * 78)


def class_balance_check(master):
    log("=== Class balance check (Section 6) ===")
    usable = master[master["usable"] == True]  # noqa: E712
    overall = usable["label"].value_counts(normalize=True).to_dict()
    log(f"Overall usable-row label balance: {overall}")
    for fold_id in [10, 50, 90]:
        fold_rows = master[master["fold"] == fold_id]
        train_pool = master[(master["fold"] < fold_id) & (master["usable"] == True)]  # noqa: E712
        test_eligible = fold_rows[fold_rows["usable"] == True]  # noqa: E712
        train_pos = train_pool["label"].mean() if len(train_pool) else None
        test_pos = test_eligible["label"].mean() if len(test_eligible) else None
        log(f"  Fold {fold_id}: TRAIN pos%={train_pos}, TRAIN neg%={1-train_pos if train_pos is not None else None}, "
            f"TEST pos%={test_pos}, TEST neg%={1-test_pos if test_pos is not None else None}")
    log("Classes are close to balanced overall (~53/47) -- class_weight='balanced' has limited expected value "
        "except in the minority of skewed folds, which is exactly what the existing conditional rule already targets.")


def fit_predict_fold(pool_fit, fold_rows, feature_columns, rf_params):
    X_train = pool_fit[feature_columns].values
    y_train = pool_fit["label"].astype(int).values
    pos_rate = float(y_train.mean())
    cw = "balanced" if not (0.4 <= pos_rate <= 0.6) else None

    rf = RandomForestClassifier(**rf_params, class_weight=cw, n_estimators=N_ESTIMATORS,
                                 random_state=SEED, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    fit_time = time.time() - t0

    train_probs = rf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs) if len(set(y_train)) > 1 else None

    trending_mask = (fold_rows["is_trending_t"] == True) & fold_rows[feature_columns].notna().all(axis=1)  # noqa: E712
    probs = np.full(len(fold_rows), np.nan)
    if trending_mask.any():
        X_pred = fold_rows.loc[trending_mask, feature_columns].values
        probs[trending_mask.values] = rf.predict_proba(X_pred)[:, 1]

    return rf, probs, fit_time, train_auc, len(X_train)


def run_one_config(config_name, rf_params, master):
    folds = sorted(master["fold"].unique())
    pool_frames = []
    all_probs = np.full(len(master), np.nan)
    fold_metrics = []
    master_by_fold_idx = {f: master.index[master["fold"] == f] for f in folds}
    last_rf = None

    pbar = tqdm(folds, desc=config_name, leave=False)
    for fold_id in pbar:
        fold_idx = master_by_fold_idx[fold_id]
        fold_rows = master.loc[fold_idx]
        pool = pd.concat(pool_frames, ignore_index=False) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712
        pool_fit = pool_usable.dropna(subset=FULL_FEATURE_COLUMNS) if len(pool_usable) else pool_usable

        if len(pool_fit) < MIN_POOL_ROWS:
            fold_metrics.append({"fold": int(fold_id), "config": config_name, "pool_train_rows": 0})
            pool_frames.append(fold_rows)
            continue

        rf, probs, fit_time, train_auc, n_train = fit_predict_fold(pool_fit, fold_rows, FULL_FEATURE_COLUMNS, rf_params)
        idx_pos = master.index.get_indexer(fold_idx)
        all_probs[idx_pos] = probs
        last_rf = rf

        eligible = fold_rows[(fold_rows["usable"] == True) & ~np.isnan(probs)]  # noqa: E712
        fm = {"fold": int(fold_id), "config": config_name, "pool_train_rows": n_train,
              "n_eligible": len(eligible), "fit_time_s": fit_time, "train_auc": train_auc}
        if len(eligible) > 5 and eligible["label"].nunique() > 1:
            idx_e = master.index.get_indexer(eligible.index)
            fm["oos_fold_auc"] = float(roc_auc_score(eligible["label"].astype(int), all_probs[idx_e]))
        fold_metrics.append(fm)
        pbar.set_postfix_str(f"n={n_train:,} train_auc={train_auc:.3f} oos={fm.get('oos_fold_auc', float('nan')):.3f}")

        pool_frames.append(fold_rows)

        # checkpoint after each fold
        pd.DataFrame(fold_metrics).to_csv(os.path.join(OUT_DIR, f"rf_regularization_fold_results_{config_name}.csv"),
                                           index=False)

    return all_probs, fold_metrics, last_rf


def metrics_block(y_true, y_prob):
    y_pred = (y_prob >= GATE_THRESHOLD).astype(int)
    return {
        "n": len(y_true), "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])) if len(set(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def year_breakdown(master, prob_col):
    m = master.copy()
    m["year"] = pd.to_datetime(m["timestamp"]).dt.year
    eligible = m[(m["usable"] == True) & m[prob_col].notna()]  # noqa: E712
    rows = []
    for year, sub in eligible.groupby("year"):
        if len(sub) > 20 and sub["label"].nunique() > 1:
            y = sub["label"].astype(int).values
            p = sub[prob_col].values
            rows.append({"year": int(year), "n": len(sub), "auc": float(roc_auc_score(y, p)),
                         "pr_auc": float(average_precision_score(y, p)),
                         "log_loss": float(log_loss(y, p, labels=[0, 1]))})
    return pd.DataFrame(rows)


def fold_delta_analysis(fold_metrics_df, lr_fold_df):
    """ΔAUC = RF fold AUC - LR fold AUC, per fold, using struct_lr's already-computed fold results."""
    merged = fold_metrics_df.merge(lr_fold_df[["fold", "lr_fold_auc"]], on="fold", how="inner")
    merged = merged.dropna(subset=["oos_fold_auc", "lr_fold_auc"])
    merged["delta_auc"] = merged["oos_fold_auc"] - merged["lr_fold_auc"]
    return {
        "n_folds_compared": len(merged),
        "rf_beats_lr": int((merged["delta_auc"] > 0).sum()),
        "lr_beats_rf": int((merged["delta_auc"] < 0).sum()),
        "mean_delta_auc": float(merged["delta_auc"].mean()) if len(merged) else None,
        "median_delta_auc": float(merged["delta_auc"].median()) if len(merged) else None,
        "std_delta_auc": float(merged["delta_auc"].std()) if len(merged) else None,
        "min_delta_auc": float(merged["delta_auc"].min()) if len(merged) else None,
        "max_delta_auc": float(merged["delta_auc"].max()) if len(merged) else None,
        "pct_positive": float((merged["delta_auc"] > 0).mean()) if len(merged) else None,
    }


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print_inspection_summary()

    log("Loading frozen HMM-4 baseline + 15 structure features (k=3)...")
    master = pd.read_csv(BASELINE_MASTER_PATH, parse_dates=["timestamp"])
    struct = pd.read_csv(STRUCTURE_FEATURES_PATH, usecols=["timestamp"] + STRUCTURE_FEATURE_COLUMNS,
                          parse_dates=["timestamp"])
    master = master.merge(struct, on="timestamp", how="left")
    log(f"  {len(master):,} rows after merge")

    class_balance_check(master)

    log("Loading prior (already-verified) LR baseline + current-RF results for comparison, not recomputed...")
    prior_preds = pd.read_csv(os.path.join(PRIOR_EXP_DIR, "predictions_all.csv"), parse_dates=["timestamp"])
    prior_struct_fold = pd.read_csv(os.path.join(PRIOR_EXP_DIR, "fold_results_struct.csv"))
    prior_base_fold = pd.read_csv(os.path.join(PRIOR_EXP_DIR, "fold_results_base.csv"))
    prior_gap = pd.read_csv(os.path.join(PRIOR_EXP_DIR, "struct_rf_train_oos_gap.csv"))

    # --- Benchmark, then proceed automatically (Section 14) ---
    last_fold = sorted(master["fold"].unique())[-1]
    pool = master[master["fold"] < last_fold]
    pool_usable = pool[pool["usable"] == True].dropna(subset=FULL_FEATURE_COLUMNS)  # noqa: E712
    fold_rows_bench = master[master["fold"] == last_fold]
    log(f"Benchmarking RF-4 (representative config) on largest pool ({len(pool_usable):,} rows)...")
    _, _, bench_fit_time, bench_train_auc, _ = fit_predict_fold(pool_usable, fold_rows_bench, FULL_FEATURE_COLUMNS,
                                                                  RF_CONFIGS["RF-4"])
    log(f"  fit_time={bench_fit_time:.2f}s train_auc={bench_train_auc:.4f}")
    est_per_config = bench_fit_time * 101 * 0.4  # same conservative scaling heuristic used in v16
    est_total = est_per_config * len(RF_CONFIGS)
    log(f"ESTIMATED runtime: {len(RF_CONFIGS)} configs x 101 folds ~= {est_total:.1f}s (~{est_total/60:.1f} min). "
        f"Proceeding automatically now (no confirmation required per task spec).")

    # --- Run all 7 configs ---
    all_results = {}
    fold_metrics_all = []
    final_models = {}
    for config_name, rf_params in RF_CONFIGS.items():
        log(f"=== Running {config_name}: {rf_params} ===")
        probs, fold_metrics, last_rf = run_one_config(config_name, rf_params, master)
        master[f"{config_name}_probability"] = probs
        fold_metrics_all.extend(fold_metrics)
        final_models[config_name] = last_rf

    runtime = time.time() - t0
    log(f"Total regularization grid runtime: {runtime:.1f}s ({runtime/60:.1f} min) (estimate was {est_total:.1f}s)")

    fold_metrics_df = pd.DataFrame(fold_metrics_all)
    fold_metrics_df.to_csv(os.path.join(OUT_DIR, "rf_regularization_fold_results.csv"), index=False)

    # --- Pooled + generalization summary for every config, plus reused LR/current-RF baselines ---
    summary_rows = []
    lr_eligible = prior_preds[(prior_preds["usable"] == True) & prior_preds["base_lr_probability"].notna()]  # noqa: E712
    lr_metrics = metrics_block(lr_eligible["label"].astype(int).values, lr_eligible["base_lr_probability"].values)
    summary_rows.append({"config": "LR (baseline, 4 feat)", "max_depth": None, "min_samples_leaf": None,
                          "min_samples_split": None, "max_features": None, "train_auc": None,
                          "train_oos_gap": None, **lr_metrics})

    struct_rf_eligible = prior_preds[(prior_preds["usable"] == True) & prior_preds["struct_rf_probability"].notna()]  # noqa: E712
    struct_rf_metrics = metrics_block(struct_rf_eligible["label"].astype(int).values,
                                       struct_rf_eligible["struct_rf_probability"].values)
    prior_train_auc = prior_gap["train_auc"].dropna().mean()
    prior_oos_auc = prior_gap["oos_auc"].dropna().mean()
    summary_rows.append({"config": "RF-current (unregularized, 19 feat)", "max_depth": 8, "min_samples_leaf": 50,
                          "min_samples_split": 100, "max_features": "sqrt", "train_auc": prior_train_auc,
                          "train_oos_gap": prior_train_auc - prior_oos_auc, **struct_rf_metrics})

    for config_name, rf_params in RF_CONFIGS.items():
        eligible = master[(master["usable"] == True) & master[f"{config_name}_probability"].notna()]  # noqa: E712
        m = metrics_block(eligible["label"].astype(int).values, eligible[f"{config_name}_probability"].values)
        cfg_fold = fold_metrics_df[fold_metrics_df["config"] == config_name]
        train_auc_mean = cfg_fold["train_auc"].dropna().mean()
        oos_auc_mean = cfg_fold["oos_fold_auc"].dropna().mean()
        gap = (train_auc_mean - oos_auc_mean) if pd.notna(train_auc_mean) and pd.notna(oos_auc_mean) else None
        summary_rows.append({"config": config_name, **rf_params, "train_auc": train_auc_mean,
                              "train_oos_gap": gap, **m})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "rf_regularization_summary.csv"), index=False)
    log("\n" + summary_df.to_string(index=False))

    # --- Year-by-year for LR, RF-current, and every new config ---
    year_frames = []
    lr_year = year_breakdown(prior_preds, "base_lr_probability")
    lr_year["config"] = "LR (baseline)"
    year_frames.append(lr_year)
    cur_year = year_breakdown(prior_preds, "struct_rf_probability")
    cur_year["config"] = "RF-current"
    year_frames.append(cur_year)
    for config_name in RF_CONFIGS:
        y = year_breakdown(master, f"{config_name}_probability")
        y["config"] = config_name
        year_frames.append(y)
    year_df = pd.concat(year_frames, ignore_index=True)
    year_df.to_csv(os.path.join(OUT_DIR, "rf_yearly_results.csv"), index=False)

    # --- Fold-by-fold delta analysis (RF vs LR) for every new config ---
    lr_fold_renamed = prior_base_fold.rename(columns={"lr_fold_auc": "lr_fold_auc"})[["fold", "lr_fold_auc"]]
    delta_rows = []
    for config_name in RF_CONFIGS:
        cfg_fold = fold_metrics_df[fold_metrics_df["config"] == config_name].copy()
        cfg_fold = cfg_fold.rename(columns={"oos_fold_auc": "oos_fold_auc"})
        d = fold_delta_analysis(cfg_fold, lr_fold_renamed)
        d["config"] = config_name
        delta_rows.append(d)
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(os.path.join(OUT_DIR, "rf_fold_delta_analysis.csv"), index=False)
    log("\nFold-by-fold delta (RF - LR) analysis:\n" + delta_df.to_string(index=False))

    # --- Feature importance for the 2 best configs (by OOS AUC) ---
    best_configs = summary_df[summary_df["config"].isin(RF_CONFIGS.keys())].nlargest(2, "roc_auc")["config"].tolist()
    imp_rows = []
    for config_name in best_configs:
        rf = final_models[config_name]
        if rf is None:
            continue
        for feat, imp in zip(FULL_FEATURE_COLUMNS, rf.feature_importances_):
            imp_rows.append({"config": config_name, "feature": feat, "importance": imp})
    imp_df = pd.DataFrame(imp_rows)
    if len(imp_df):
        imp_df["rank"] = imp_df.groupby("config")["importance"].rank(ascending=False).astype(int)
    imp_df.to_csv(os.path.join(OUT_DIR, "rf_feature_importance.csv"), index=False)

    # --- Decision framework (Section 16) ---
    verdicts = []
    for _, row in summary_df[summary_df["config"].isin(RF_CONFIGS.keys())].iterrows():
        delta_auc = row["roc_auc"] - lr_metrics["roc_auc"]
        gap = row["train_oos_gap"]
        worse_logloss = row["log_loss"] > lr_metrics["log_loss"]
        worse_brier = row["brier"] > lr_metrics["brier"]
        reject = (abs(delta_auc) < 0.003) or (worse_logloss and worse_brier) or (gap is not None and gap > 0.10)
        verdicts.append({"config": row["config"], "delta_auc_vs_lr": delta_auc, "train_oos_gap": gap,
                          "worse_logloss": worse_logloss, "worse_brier": worse_brier,
                          "reject": reject})
    verdict_df = pd.DataFrame(verdicts)
    verdict_df.to_csv(os.path.join(OUT_DIR, "rf_regularization_verdicts.csv"), index=False)
    log("\nPer-config reject/keep signal:\n" + verdict_df.to_string(index=False))

    json.dump({
        "rf_configs": RF_CONFIGS, "n_estimators": N_ESTIMATORS, "seed": SEED,
        "feature_columns": FULL_FEATURE_COLUMNS, "min_pool_rows": MIN_POOL_ROWS,
        "gate_threshold": GATE_THRESHOLD, "benchmark_fit_time_s": bench_fit_time,
        "estimated_runtime_s": est_total, "actual_runtime_s": runtime,
        "experiment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": BASELINE_MASTER_PATH, "structure_features_path": STRUCTURE_FEATURES_PATH,
    }, open(os.path.join(OUT_DIR, "rf_experiment_config.json"), "w"), indent=2, default=str)

    log(f"\nAll artifacts saved to {OUT_DIR}")
    log(f"Total runtime: {runtime:.1f}s ({runtime/60:.1f} min)")


if __name__ == "__main__":
    main()
