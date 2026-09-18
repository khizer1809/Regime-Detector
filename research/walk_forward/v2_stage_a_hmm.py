"""
v2_stage_a_hmm.py -- V2 Stage A (the expensive, cacheable part).

Per fold: fit HMM-4 on the expanding train window (identical hyperparameters
to save_production_model.py), then CAUSALLY decode only that fold's own
~1-month test window (never the whole expanding history under one fold's
model -- see v2_common.py's module docstring for why, and for the causal
decode mechanism itself). Writes one directory per fold under
Data/v2_cache/fold_XXX/ containing the fitted HMM, the scaler, the raw
causal decode + forward-return output, and fold metadata, plus a DONE
marker. Crash-safe: on restart, any fold whose DONE marker already exists
is skipped entirely, never recomputed.

This script never trains or evaluates the Logistic Regression -- that's
v2_stage_b_lr.py, which reads only the CSVs this script writes and never
touches the HMM. A future experiment that only changes the LR/threshold/
classifier re-runs Stage B in seconds to minutes; Stage A only needs to
run again if something HMM-related changes (see the module docstring in
v2_stage_b_lr.py for the exact list).

Run: `python v2_stage_a_hmm.py` (from src/).
"""

import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v2_common
from gap_aware import segment_lengths, valid_values_and_lengths
from save_production_model import _fit_one_segmented_with_retry, load_masked_features

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_ROOT, "Data", "v2_cache")
LOG_PATH = os.path.join(CACHE_DIR, "stage_a_run_log.jsonl")


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_global_valid_data():
    df, model_cols = load_masked_features()
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
        log(f"fold {fold_id:03d}: 0 test rows (data ends before test window) -- marking done, no output")
        with open(os.path.join(fold_dir, "metadata.json"), "w") as f:
            json.dump({**{k: str(v) for k, v in fold.items()}, "n_train": n_train, "n_test": 0}, f, indent=2)
        open(done_marker, "w").close()
        return

    train_index = valid_index[train_pos_mask]
    train_values = valid_df.values[train_pos_mask]
    train_lengths = segment_lengths(train_index)
    assert sum(train_lengths) == len(train_values)

    log(f"fold {fold_id:03d}: fitting HMM-4 on {n_train:,} train rows "
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
            "timestamp": valid_index[p],
            "fold": fold_id,
            "hmm_state": state,
            "posterior_state_0": dec["posteriors"][0] if len(dec["posteriors"]) > 0 else np.nan,
            "posterior_state_1": dec["posteriors"][1] if len(dec["posteriors"]) > 1 else np.nan,
            "posterior_state_2": dec["posteriors"][2] if len(dec["posteriors"]) > 2 else np.nan,
            "posterior_state_3": dec["posteriors"][3] if len(dec["posteriors"]) > 3 else np.nan,
            "confidence": dec["confidence"],
            "margin": dec["margin"],
            "duration": dec["duration"],
            "duration_left_censored": dec["duration_left_censored"],
            "stay_prob": dec["stay_prob"],
            "trend_score": float(trend_score_per_state[state]),
            "trend_sign": float(trend_sign_per_state[state]),
            "is_trending": bool(is_trending_per_state[state]),
            "segment_id": int(seg_id[p]),
            "forward_ret_24": forward_ret,
            "forward_valid": forward_valid,
        })
    decode_time = time.time() - t1

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(fold_dir, "causal_output.csv"), index=False)

    metadata = {
        "fold_id": fold_id,
        "train_start": str(fold["train_start"]), "train_end": str(fold["train_end"]),
        "test_start": str(fold["test_start"]), "test_end": str(fold["test_end"]),
        "embargo_bars": v2_common.EMBARGO_BARS,
        "n_train": n_train, "n_test": n_test,
        "feature_columns": model_cols,
        "n_states": v2_common.N_STATES, "covariance_type": v2_common.COVARIANCE_TYPE,
        "n_iter": v2_common.N_ITER, "n_restarts": v2_common.N_RESTARTS,
        "best_seed": best_seed, "best_restart": best_restart, "best_train_ll": best_ll,
        "restart_log": restart_log,
        "context_bars": v2_common.CONTEXT_BARS,
        "fit_time_s": fit_time, "decode_time_s": decode_time, "total_time_s": time.time() - t0,
        "code_version": "v2_stage_a_hmm.py:1",
    }
    with open(os.path.join(fold_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    open(done_marker, "w").close()
    log(f"fold {fold_id:03d}: done. n_train={n_train:,} n_test={n_test:,} "
        f"fit={fit_time:.1f}s decode={decode_time:.1f}s total={time.time()-t0:.1f}s")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    log("=== V2 Stage A starting ===")
    valid_df, valid_index, seg_id, model_cols = load_global_valid_data()
    last_ts = valid_index.max()
    folds = v2_common.generate_folds(last_ts)
    log(f"generated {len(folds)} folds, last valid ts={last_ts}")

    failed_folds = []
    for fold in folds:
        fold_id = fold["fold_id"]
        fold_dir = os.path.join(CACHE_DIR, f"fold_{fold_id:03d}")
        for attempt in (1, 2):  # one retry for transient failures (e.g. thread-pool resource contention)
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
                    log(f"fold {fold_id:03d}: giving up after 2 attempts, marked FAILED, continuing to next fold")

    if failed_folds:
        log(f"=== V2 Stage A complete WITH {len(failed_folds)} FAILED FOLD(S): {failed_folds} -- "
            f"all other folds completed; see fold_XXX/FAILED for details ===")
    else:
        log("=== V2 Stage A complete: all folds done, zero failures ===")


if __name__ == "__main__":
    main()
