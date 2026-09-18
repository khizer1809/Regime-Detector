"""
v15_calibration_fit.py -- ONE-OFF TIMING CALIBRATION, not part of the actual
experiment. Fits a single full-history HMM-4 on the 33-feature expanded
matrix (5 restarts, identical hyperparameters to production) to get a REAL
wall-clock number for the largest (dominant-cost) fold size, before
committing to launch the full 101-fold expanding walk-forward Stage A run.
Writes nothing except a timing report to stdout.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from gap_aware import valid_values_and_lengths
from scaling import fit_transform_fold
from save_production_model import _fit_one_segmented_with_retry

FEATURES_PATH = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "features_expanded_33.csv")
N_STATES = 4
N_RESTARTS = 5
RANDOM_STATE = 42
N_WORKERS = min(6, os.cpu_count() or 1)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("Loading expanded 33-feature matrix...")
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    model_cols = [c for c in df.columns if c != "valid"]
    log(f"  {len(df):,} rows, {len(model_cols)} feature columns")

    train_vals, lengths, valid_index = valid_values_and_lengths(df, model_cols)
    log(f"  valid rows: {len(train_vals):,}  segments: {len(lengths)}")

    train_matrix, _, scaler = fit_transform_fold(train_vals, train_vals.iloc[:10])
    train_matrix = train_matrix.values

    seeds = [RANDOM_STATE + r for r in range(N_RESTARTS)]
    log(f"Fitting HMM-{N_STATES} on FULL history, {N_RESTARTS} restarts (up to {N_WORKERS} concurrent)...")
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_fit_one_segmented_with_retry, train_matrix, lengths, N_STATES, s): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            result, _ = fut.result()
            status = "FAILED" if result["error"] else f"ll={result['ll']:.1f}"
            log(f"  restart seed={seed}: {status} ({result['runtime']:.1f}s)")

    fit_time = time.time() - t1
    total_time = time.time() - t0
    log(f"\nFULL-HISTORY (largest-fold-equivalent) fit time: {fit_time:.1f}s ({fit_time/60:.2f} min)")
    log(f"Total (incl. load+scale): {total_time:.1f}s ({total_time/60:.2f} min)")


if __name__ == "__main__":
    main()
