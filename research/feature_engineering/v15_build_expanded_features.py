"""
v15_build_expanded_features.py -- EXPERIMENTAL, NOT PRODUCTION.

Builds the 33-column expanded feature matrix (18 existing production HMM
features + 15 causal market-structure features at k=3, REUSED VERBATIM from
v14_market_structure_test.py -- not redefined) aligned to the exact same
row/index set as Data/features_out_masked.csv, for the A/B/C HMM-location
experiment. Written once and cached (structure feature computation itself
is cheap, but this avoids recomputing it on every Stage A/calibration run).

Does not modify Data/features_out_masked.csv or any production file.
Output: research_archive/Data/hmm_lr_abc_experiment/features_expanded_33.csv
"""

import os
import sys
import time

import pandas as pd

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v14_market_structure_test import build_structure_features, STRUCTURE_FEATURE_COLUMNS, RAW_OHLCV_PATH, load_raw_ohlcv  # noqa: E402

MASKED_PATH = os.path.join(_PROJECT_ROOT, "Data", "features_out_masked.csv")
OUT_DIR = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment")
OUT_PATH = os.path.join(OUT_DIR, "features_expanded_33.csv")
K_FIXED = 3  # frozen, per task spec -- reused from the completed market-structure experiment, not reselected


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("Loading existing 18-feature masked matrix (production, unmodified)...")
    masked = pd.read_csv(MASKED_PATH, index_col=0, parse_dates=True)
    orig_cols = [c for c in masked.columns if c != "valid"]
    log(f"  {len(masked):,} rows, {len(orig_cols)} existing feature columns")

    log(f"Building 15 causal market-structure features at FIXED k={K_FIXED} (reused verbatim from v14)...")
    raw_df = load_raw_ohlcv()
    struct_df, n_events = build_structure_features(raw_df, K_FIXED)
    log(f"  {n_events:,} confirmed structural events")

    struct_df = struct_df.set_index("timestamp")
    combined = masked.join(struct_df, how="left")
    assert len(combined) == len(masked), "row count changed on join"

    n_struct_nan = int(combined[STRUCTURE_FEATURE_COLUMNS].isna().any(axis=1).sum())
    log(f"  rows with any NaN structure feature (pre-first-pivot warmup): {n_struct_nan:,} "
        f"({n_struct_nan/len(combined):.3%})")

    combined.to_csv(OUT_PATH)
    log(f"Saved expanded 33-feature matrix -> {OUT_PATH}")
    log(f"Final columns ({len(combined.columns)}): {combined.columns.tolist()}")


if __name__ == "__main__":
    main()
