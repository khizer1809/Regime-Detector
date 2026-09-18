"""
v15_stageA_expanded_hmm.py -- A/B/C experiment, Stage A (expensive, cacheable).

Per fold: fit HMM-4 on the expanding train window using the EXPANDED
33-feature matrix (18 existing production features + 15 causal
market-structure features at fixed k=3), then CAUSALLY decode only that
fold's own test window -- identical methodology, hyperparameters, and fold
schedule to v2_stage_a_hmm.py (which this is adapted from), differing ONLY
in the feature-column list fed to the HMM. No HMM states/covariance/
restarts/init/training methodology changed -- see the module docstring in
v14/v15's parent task spec.

This is the ONE expensive HMM refit needed for BOTH Test A and Test B (they
share the identical expanded-feature HMM; only the downstream LR's feature
set differs between A and B -- built in v15_stageB.py).

Writes one directory per fold under
Data/hmm_lr_abc_experiment/expanded_hmm_cache/fold_XXX/, crash-safe/
resumable exactly like v2_stage_a_hmm.py (DONE marker, 2-attempt retry,
FAILED marker + continue on persistent failure).

Run: `python v15_stageA_expanded_hmm.py` (from research_archive/src/).
"""

import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v2_common  # noqa: E402
from gap_aware import segment_lengths, valid_values_and_lengths  # noqa: E402
from save_production_model import _fit_one_segmented_with_retry  # noqa: E402

FEATURES_PATH = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "features_expanded_33.csv")
CACHE_DIR = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "expanded_hmm_cache")
LOG_PATH = os.path.join(CACHE_DIR, "stage_a_run_log.jsonl")

# PILOT MODE: if set (via env var FOLD_SUBSET_STEP), only fit every Nth fold
# of the full 101-fold schedule (e.g. step=10 -> folds 0,10,20,...,100), for
# a cheap, non-cherry-picked, date-range-spanning cost/direction pilot before
# committing to the full run. None (default) = every fold, the real experiment.
FOLD_SUBSET_STEP = int(os.environ.get("FOLD_SUBSET_STEP", "0")) or None


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_global_valid_data():
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    model_cols = [c for c in df.columns if c != "valid"]
    valid_df, lengths, valid_index = valid_values_and_lengths(df, model_cols)
    seg_id, _ = v2_common.segment_ids_for_index(valid_index)
    return valid_df, valid_index, seg_id, model_cols


def fit_fold_hmm(train_values: np.ndarray, train_lengths: list):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_values)
    train_scaled = scaler.transform(train_values)

    seeds = [v2_common.RANDOM_STATE + r for r in range(v2_common.N_RESTARTS)]
    best_model, best_ll, best_seed, best_restart = None, float("-inf"), None, None
    restart_log = []
    with ThreadPoolExecutor(max_workers=v2_common.N_WORKERS) as executor:
        futures = {executor.submit(_fit_one_segmented_with_retry, train_scaled, train_lengths,
                                    v2_common.N_STATES, s): (r, s)
                   for r, s in enumerate(seeds)}
        for fut in as_completed(futures):
            r, seed = futures[fut]
            result, n_retries = fut.result()
            failed = result["error"] is not None
            restart_log.append({"restart": r, "seed": seed, "ll": result["ll"], "failed": failed,
                                 "error": result["error"], "runtime_s": result["runtime"]})
            if not failed and result["ll"] > best_ll:
                best_model, best_ll, best_seed, best_restart = result["model"], result["ll"], seed, r
    if best_model is None:
        raise RuntimeError("All restarts failed for this fold -- no model to save.")
    return best_model, scaler, best_ll, best_seed, best_restart, restart_log


def run_fold(fold: dict, valid_df, valid_index, seg_id, model_cols):
    fold_id = fold["fold_id"]
    fold_dir = os.path.join(CACHE_DIR, f"fold_{fold_id:03d}")
    done_marker = os.path.join(fold_dir, "DONE")
    if os.path.exists(done_marker):
        log(f"fold {fold_id:03d}: already done, skipping")
        return

    t0 = time.time()
    os.makedirs(fold_dir, exist_ok=True)

    train_pos_mask = np.array(valid_index < fold["train_end"])
    test_pos_mask = np.array((valid_index >= fold["test_start"]) & (valid_index < fold["test_end"]))
    n_train = int(train_pos_mask.sum())
    n_test = int(test_pos_mask.sum())
    if n_test == 0:
        log(f"fold {fold_id:03d}: 0 test rows -- marking done, no output")
        with open(os.path.join(fold_dir, "metadata.json"), "w") as f:
            json.dump({**{k: str(v) for k, v in fold.items()}, "n_train": n_train, "n_test": 0}, f, indent=2)
        open(done_marker, "w").close()
        return

    train_index = valid_index[train_pos_mask]
    train_values = valid_df.values[train_pos_mask]
    train_lengths = segment_lengths(train_index)
    assert sum(train_lengths) == len(train_values)

    log(f"fold {fold_id:03d}: fitting HMM-4 (33 features) on {n_train:,} train rows "
        f"({fold['train_start'].date()} -> {fold['train_end'].date()})...")
    model, scaler, best_ll, best_seed, best_restart, restart_log = fit_fold_hmm(train_values, train_lengths)
    fit_time = time.time() - t0

    with open(os.path.join(fold_dir, "hmm.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(fold_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    means_df = pd.DataFrame(model.means_, columns=model_cols)
    trend_score_per_state = v2_common.persistence_common.state_trend_scores(means_df)
    trend_sign_per_state = np.sign(trend_score_per_state)
    is_trending_per_state = np.abs(trend_score_per_state) > v2_common.persistence_common.TREND_Z_THRESHOLD

    t1 = time.time()
    test_positions = np.where(test_pos_mask)[0]
    decode_upper_pos = int(test_positions.max()) + 1
    scaled_prefix = scaler.transform(valid_df.values[:decode_upper_pos])
    forward_ret_full = valid_df[v2_common.FORWARD_RET_COL].values

    rows = []
    for p in test_positions:
        dec = v2_common.causal_decode_at(scaled_prefix, seg_id, p, model, v2_common.CONTEXT_BARS)
        j = p + v2_common.LABEL_HORIZON_BARS
        forward_valid = (j < len(seg_id)) and (seg_id[min(j, len(seg_id) - 1)] == seg_id[p])
        forward_ret = float(forward_ret_full[j]) if forward_valid else np.nan
        state = dec["state"]
        rows.append({
            "timestamp": valid_index[p], "fold": fold_id, "hmm_state": state,
            "posterior_state_0": dec["posteriors"][0] if len(dec["posteriors"]) > 0 else np.nan,
            "posterior_state_1": dec["posteriors"][1] if len(dec["posteriors"]) > 1 else np.nan,
            "posterior_state_2": dec["posteriors"][2] if len(dec["posteriors"]) > 2 else np.nan,
            "posterior_state_3": dec["posteriors"][3] if len(dec["posteriors"]) > 3 else np.nan,
            "confidence": dec["confidence"], "margin": dec["margin"], "duration": dec["duration"],
            "duration_left_censored": dec["duration_left_censored"], "stay_prob": dec["stay_prob"],
            "trend_score": float(trend_score_per_state[state]), "trend_sign": float(trend_sign_per_state[state]),
            "is_trending": bool(is_trending_per_state[state]), "segment_id": int(seg_id[p]),
            "forward_ret_24": forward_ret, "forward_valid": forward_valid,
        })
    decode_time = time.time() - t1

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(fold_dir, "causal_output.csv"), index=False)

    metadata = {
        "fold_id": fold_id, "train_start": str(fold["train_start"]), "train_end": str(fold["train_end"]),
        "test_start": str(fold["test_start"]), "test_end": str(fold["test_end"]),
        "embargo_bars": v2_common.EMBARGO_BARS, "n_train": n_train, "n_test": n_test,
        "feature_columns": model_cols, "n_states": v2_common.N_STATES, "covariance_type": v2_common.COVARIANCE_TYPE,
        "n_iter": v2_common.N_ITER, "n_restarts": v2_common.N_RESTARTS,
        "best_seed": best_seed, "best_restart": best_restart, "best_train_ll": best_ll, "restart_log": restart_log,
        "context_bars": v2_common.CONTEXT_BARS, "fit_time_s": fit_time, "decode_time_s": decode_time,
        "total_time_s": time.time() - t0, "code_version": "v15_stageA_expanded_hmm.py:1",
    }
    with open(os.path.join(fold_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    open(done_marker, "w").close()
    log(f"fold {fold_id:03d}: done. n_train={n_train:,} n_test={n_test:,} "
        f"fit={fit_time:.1f}s decode={decode_time:.1f}s total={time.time()-t0:.1f}s")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    log("=== V15 Stage A (expanded 33-feature HMM) starting ===")
    valid_df, valid_index, seg_id, model_cols = load_global_valid_data()
    last_ts = valid_index.max()
    all_folds = v2_common.generate_folds(last_ts)
    if FOLD_SUBSET_STEP:
        folds = [f for f in all_folds if f["fold_id"] % FOLD_SUBSET_STEP == 0]
        log(f"PILOT MODE: FOLD_SUBSET_STEP={FOLD_SUBSET_STEP} -- running {len(folds)}/{len(all_folds)} folds "
            f"(fold_ids={[f['fold_id'] for f in folds]})")
    else:
        folds = all_folds
    log(f"generated {len(all_folds)} total folds, using {len(folds)}, last valid ts={last_ts}, "
        f"feature_columns={len(model_cols)}")

    failed_folds = []
    for fold in folds:
        fold_id = fold["fold_id"]
        fold_dir = os.path.join(CACHE_DIR, f"fold_{fold_id:03d}")
        for attempt in (1, 2):
            try:
                run_fold(fold, valid_df, valid_index, seg_id, model_cols)
                break
            except Exception as e:
                log(f"fold {fold_id:03d}: attempt {attempt} FAILED -- {type(e).__name__}: {e}")
                if attempt == 2:
                    os.makedirs(fold_dir, exist_ok=True)
                    with open(os.path.join(fold_dir, "FAILED"), "w") as f:
                        f.write(f"{type(e).__name__}: {e}\n")
                    failed_folds.append(fold_id)
                    log(f"fold {fold_id:03d}: giving up after 2 attempts, marked FAILED, continuing")

    if failed_folds:
        log(f"=== V15 Stage A complete WITH {len(failed_folds)} FAILED FOLD(S): {failed_folds} ===")
    else:
        log("=== V15 Stage A complete: all folds done, zero failures ===")


if __name__ == "__main__":
    main()
