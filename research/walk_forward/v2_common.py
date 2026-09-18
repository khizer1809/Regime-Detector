"""
v2_common.py -- shared, leakage-safe building blocks for the V2 walk-forward
persistence experiment. No file I/O beyond what's passed in; used by both
v2_stage_a_hmm.py (expensive: HMM fit + causal decode, cached to disk) and
v2_stage_b_lr.py (cheap: expanding LR pool + evaluation, reads the cache).

FOLD SCHEDULE -- reconstructed, not copied verbatim: the original
walk_forward.py that defined the V1 experiment's exact fold boundaries was
deleted during an earlier cleanup pass (before this project was a git repo),
so there is no literal original code left to import. This regenerates an
equivalent schedule from the documented design parameters (recorded in
Data/rolling_vs_expanding_final_report.md and this session's own history):
train start 2017-09-01, a 6-month warm-up before the first test month (so
the ORIGINAL rolling variant's first fold had a full 6-month train window
too -- V2 only uses expanding, but reusing the same test-period boundaries
keeps this comparable to the V1/original experiment), 1-month test, 1-month
step, 48-bar (4h) embargo between train end and test start. This is an
explicit, documented autonomous reconstruction -- flagged here rather than
silently presented as "the same code."

CAUSAL DECODE -- the core leakage-safety mechanism: for every test candle,
decode a window of up to CONTEXT_BARS valid bars ending AT that candle
(clipped to not cross a genuine gap-aware segment boundary, but free to
reach back into the fold's own training period, since that's still past
data) and read out only the LAST row's state/posterior. This is exact, not
an approximation of causality: hmmlearn's forward-backward smoothing
degenerates to pure forward filtering at the terminal position of any
sequence (there is nothing after the last observation to smooth with), and
Viterbi's terminal-state choice is likewise determined entirely by its
forward recursion up to that point. Verified in the V1 leakage audit
(decoding a truncated prefix vs. a full segment gives different
confidence/margin mid-sequence, but the same mechanism guarantees the
terminal read is unaffected by anything after it). A short context window
(not the full available history) is a deliberate, documented approximation
of a full-history causal filter -- the same accepted approximation
infer_regime.py already uses live (CONTEXT_BARS lets the model "forget" an
arbitrary start before trusting the final read), not a new leakage source.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import persistence_common
from gap_aware import BAR_INTERVAL, segment_lengths

N_STATES = 4
DATA_START = pd.Timestamp("2017-09-01", tz="UTC")
WARMUP_MONTHS = 6          # matches the original rolling variant's train window, kept for comparability
TEST_MONTHS = 1
STEP_MONTHS = 1
EMBARGO_BARS = 48          # 4h -- matches this project's existing embargo convention everywhere else
CONTEXT_BARS = 500         # causal-decode window length; comfortably above the ~7-15 bar mean regime duration

# HMM hyperparameters -- copied verbatim from save_production_model.py, not re-derived,
# per the explicit requirement to preserve the validated architecture.
COVARIANCE_TYPE = "diag"
N_ITER = 100
RANDOM_STATE = 42
N_RESTARTS = 5
N_WORKERS = min(6, os.cpu_count() or 1)

FORWARD_RET_COL = "ret_2h"
LABEL_HORIZON_BARS = persistence_common.LABEL_HORIZON_BARS  # 24


def generate_folds(last_valid_ts: pd.Timestamp) -> list:
    """Reconstructed expanding-walk-forward fold schedule (see module
    docstring). Stops once a fold's test window would run past the data's
    actual end, so the fold count is derived from the data, not hardcoded."""
    folds = []
    first_test_start = DATA_START + pd.DateOffset(months=WARMUP_MONTHS)
    k = 0
    while True:
        test_start = first_test_start + pd.DateOffset(months=STEP_MONTHS * k)
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
        if test_start > last_valid_ts:
            break
        test_end = min(test_end, last_valid_ts + BAR_INTERVAL)
        train_end = test_start - EMBARGO_BARS * BAR_INTERVAL  # exclusive upper bound on train
        folds.append({
            "fold_id": k,
            "train_start": DATA_START,
            "train_end": train_end,       # train = [train_start, train_end)
            "test_start": test_start,     # test = [test_start, test_end)
            "test_end": test_end,
        })
        k += 1
    return folds


def segment_ids_for_index(index: pd.DatetimeIndex) -> np.ndarray:
    """Global segment ids over an already-valid-only, chronological index --
    computed once from the raw data's gap structure, independent of any
    fold's model, so it's safe to reuse across every fold."""
    lengths = segment_lengths(index)
    return persistence_common.segment_ids_from_lengths(lengths), lengths


def causal_decode_at(scaled_values: np.ndarray, seg_id: np.ndarray, target_pos: int,
                      model, context_bars: int = CONTEXT_BARS) -> dict:
    """Decodes a window of up to `context_bars` rows ending at and including
    `target_pos` (positions are into the GLOBAL, already-scaled, valid-only
    array), clipped to `target_pos`'s own gap-aware segment so no transition
    is ever modeled across a real gap. Returns the terminal (last-row)
    state/posterior/duration/confidence/margin/stay_prob -- exactly causal,
    per the module docstring. `model` must already be fit (this never fits
    anything)."""
    this_seg = seg_id[target_pos]
    seg_start = target_pos
    while seg_start > 0 and seg_id[seg_start - 1] == this_seg:
        seg_start -= 1
    window_start = max(seg_start, target_pos - context_bars + 1)
    window = scaled_values[window_start:target_pos + 1]
    window_len = window.shape[0]

    states = model.predict(window, lengths=[window_len])
    posteriors = model.predict_proba(window, lengths=[window_len])

    state = int(states[-1])
    confidence = float(posteriors[-1, state])
    margin = float(persistence_common.compute_margin(posteriors)[-1])
    duration_raw = int(persistence_common.compute_running_duration(states, [window_len])[-1])
    stay_prob = float(model.transmat_[state, state])
    left_censored = duration_raw == window_len

    return {
        "state": state, "confidence": confidence, "margin": margin,
        "duration": duration_raw, "stay_prob": stay_prob,
        "posteriors": posteriors[-1].tolist(), "duration_left_censored": left_censored,
    }
