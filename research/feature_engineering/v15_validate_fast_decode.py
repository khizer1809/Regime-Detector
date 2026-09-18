"""
v15_validate_fast_decode.py -- validates v15_fast_causal_decode.py's
single-pass forward-filter against the existing trusted per-row windowed
causal_decode_at, using fold_000's already-fit HMM (real data, real model,
not synthetic) before trusting the fast path for the real extension pass.
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v2_common  # noqa: E402
from gap_aware import valid_values_and_lengths  # noqa: E402
from v15_fast_causal_decode import validate  # noqa: E402

FEATURES_PATH = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "features_expanded_33.csv")
CACHE_DIR = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "expanded_hmm_cache")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("Loading expanded feature matrix + fold_000's already-fit HMM...")
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    model_cols = [c for c in df.columns if c != "valid"]
    valid_df, lengths, valid_index = valid_values_and_lengths(df, model_cols)
    seg_id, _ = v2_common.segment_ids_for_index(valid_index)

    fold0_dir = os.path.join(CACHE_DIR, "fold_000")
    with open(os.path.join(fold0_dir, "hmm.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(fold0_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    log(f"  loaded fold_000 HMM (n_components={model.n_components})")

    # sample 300 real positions spanning fold_000's own test window
    test_pos_mask = np.array((valid_index >= pd.Timestamp("2018-03-01", tz="UTC")) &
                              (valid_index < pd.Timestamp("2018-04-01", tz="UTC")))
    test_positions = np.where(test_pos_mask)[0]
    rng = np.random.default_rng(42)
    sample = rng.choice(test_positions, size=min(300, len(test_positions)), replace=False)
    sample.sort()
    log(f"  validating on {len(sample)} sampled real positions from fold_000's test window")

    mismatches, n = validate(model, scaler, valid_df, seg_id, valid_index, sample,
                              v2_common.CONTEXT_BARS, v2_common.causal_decode_at)

    log(f"Mismatches: {len(mismatches)}/{n}")
    if mismatches:
        for m in mismatches[:10]:
            log(f"  {m}")
    log(f"Total validation time: {time.time()-t0:.1f}s")
    log(f"VALIDATION {'PASSED' if len(mismatches) == 0 else 'FAILED'}")


if __name__ == "__main__":
    main()
