"""
v24_30m_hmm4_walkforward.py -- EXPERIMENTAL, NOT PRODUCTION.

Extends the completed 3-way walk-forward validation (v23) with a 4th
configuration: 30m-HMM4. Not part of the original 3-way task (which tested
A=5m-HMM4, B=15m-HMM4, C=30m-HMM6) -- 30m-HMM4 was only ever tested in the
fast screening (full-history development fit, v22), never OOS. This fills
that gap using the EXACT SAME methodology/infrastructure as v23 (identical
fold schedule, embargo, HMM hyperparameters, leakage controls) -- imports
v23's own functions directly rather than reimplementing them, so this is
not a new methodology, just the same one applied to one more config.

Does NOT touch, modify, or re-run any of the 3 already-completed models'
results. Writes only NEW files, all prefixed `30m_hmm4_`, into the same
experiments/final_3way_walkforward/ directory, matching that directory's
existing naming convention.

Run: `python v24_30m_hmm4_walkforward.py` (from research_archive/src/).
"""

import json
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import v23_final_3way_walkforward as v23

EXPERIMENT_DIR = v23.EXPERIMENT_DIR  # same directory as the completed 3-way run
MODEL_KEY = "30m_hmm4"
CFG = {"timeframe": "30m", "n_states": 4, "bar_minutes": 30.0}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    t_start = time.time()
    log("Loading raw 5m OHLCV + building 30m feature table (shared with v23, no new methodology)...")
    v23._RAW5M_CACHE["df"] = v23.v22.load_raw_5m()
    feat_df, valid, bar_interval, feat_cols = v23.build_feature_table("30m")
    v23._AGG_CACHE["30m"] = v23.v22.build_agg_candles(v23._RAW5M_CACHE["df"], "30m")

    last_valid_ts = feat_df.index[valid].max()
    folds = v23.generate_folds(last_valid_ts)
    log(f"Generated {len(folds)} folds (identical schedule to the completed 3-way run: "
        f"expanding, WARMUP={v23.WARMUP_MONTHS}mo, TEST={v23.TEST_MONTHS}mo, EMBARGO={v23.EMBARGO_HOURS}h). "
        f"First test={folds[0]['test_start'].date()}, last test={folds[-1]['test_start'].date()}")

    # --- benchmark (fold 0 + mid fold), then proceed automatically ---
    bench_fold_ids = sorted({0, len(folds) // 2})
    bench_times = []
    for fid in bench_fold_ids:
        t0 = time.time()
        v23.run_fold(MODEL_KEY, CFG, folds[fid], feat_df, valid, bar_interval, feat_cols)
        dt = time.time() - t0
        bench_times.append(dt)
        log(f"  fold {fid} benchmark: {dt:.1f}s")
    per_fold_est = float(np.mean(bench_times))
    est_total = per_fold_est * len(folds)
    log(f"ESTIMATED total runtime: ~{per_fold_est:.1f}s/fold x {len(folds)} folds "
        f"~= {est_total/60:.1f} min. Proceeding automatically now.")

    fold_rows, regime_rows_all, future_ret_all, restart_log_all = [], [], [], []
    skipped = []
    for fold in tqdm(folds, desc=f"{MODEL_KEY} walk-forward"):
        try:
            result = v23.run_fold(MODEL_KEY, CFG, fold, feat_df, valid, bar_interval, feat_cols)
        except Exception as e:
            log(f"  fold {fold['fold_id']} FAILED: {e}")
            skipped.append({"fold_id": fold["fold_id"], "error": str(e)})
            continue
        if result.get("status") != "ok":
            skipped.append(result)
            continue
        fold_rows.append(result["fold_row"])
        regime_rows_all.append(result["regime_rows"])
        if result.get("future_ret_df") is not None:
            future_ret_all.append(result["future_ret_df"])
        restart_log_all.extend([{**r, "model": MODEL_KEY, "fold_id": fold["fold_id"]} for r in result["restart_log"]])

        # checkpoint after every fold, same pattern as v23
        pd.DataFrame(fold_rows).to_csv(os.path.join(EXPERIMENT_DIR, f"{MODEL_KEY}_fold_results.csv"), index=False)
        json.dump({"skipped": skipped, "n_completed": len(fold_rows)},
                   open(os.path.join(EXPERIMENT_DIR, f"{MODEL_KEY}_checkpoint.json"), "w"), indent=2, default=str)

    pd.concat(regime_rows_all, ignore_index=True).to_parquet(
        os.path.join(EXPERIMENT_DIR, f"{MODEL_KEY}_oos_regimes.parquet"), index=False)
    if future_ret_all:
        pd.concat(future_ret_all, ignore_index=True).to_csv(
            os.path.join(EXPERIMENT_DIR, f"{MODEL_KEY}_future_returns.csv"), index=False)
    pd.DataFrame(restart_log_all).to_csv(os.path.join(EXPERIMENT_DIR, f"{MODEL_KEY}_restart_log.csv"), index=False)

    runtime = time.time() - t_start
    log(f"Total runtime: {runtime:.1f}s ({runtime/60:.1f} min)")
    log(f"{len(fold_rows)}/{len(folds)} folds completed, {len(skipped)} skipped/failed")
    log("Done. EXIT_MARKER_V24_DONE")


if __name__ == "__main__":
    main()
