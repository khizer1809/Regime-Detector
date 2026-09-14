"""
persistence_common.py -- pure numeric helpers shared by persistence_dataset.py
(historical training-table construction) and persistence_gate.py (live
inference). No file I/O, no model loading here -- keeps the feature/label
arithmetic defined in exactly one place so training and live serving can
never silently drift apart.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_regime import RETURN_COLS, TREND_Z_THRESHOLD

LABEL_HORIZON_BARS = 24  # ~2h, matches ret_2h's own trailing window exactly


def segment_ids_from_lengths(lengths: list) -> np.ndarray:
    """Which contiguous gap_aware segment each row (in valid-only,
    chronological order) belongs to, e.g. lengths=[3,2] -> [0,0,0,1,1]."""
    return np.repeat(np.arange(len(lengths)), lengths)


def compute_running_duration(states: np.ndarray, lengths: list) -> np.ndarray:
    """Backward-looking count of consecutive bars in the same decoded state,
    1-indexed (the bar a state starts on has duration 1). Resets both on a
    state change AND on a segment/gap boundary -- a state that happens to
    repeat right after a gap is not the same run as before the gap."""
    n = len(states)
    seg_id = segment_ids_from_lengths(lengths)
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = (states[1:] != states[:-1]) | (seg_id[1:] != seg_id[:-1])
    run_id = np.cumsum(changed)
    return pd.Series(run_id).groupby(run_id).cumcount().values + 1


def compute_stay_prob(states: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    """Self-transition probability of the decoded state at each bar --
    a fixed, state-specific scalar from the trained model, not recomputed
    per bar (only which state is active changes)."""
    return transmat[states, states]


def compute_confidence(posteriors: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Posterior mass on the VITERBI-decoded state at each bar -- matches
    what infer_regime.py already reports as "Posterior confidence". This is
    NOT argmax(posteriors) by construction; keep it consistent with that."""
    idx = np.arange(len(states))
    return posteriors[idx, states]


def compute_margin(posteriors: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 posterior probability per bar -- how decisively the
    HMM prefers its best state over the next-best alternative. Deliberately
    independent of which state Viterbi actually picked: a different signal
    from `compute_confidence`, not a restatement of it."""
    sorted_desc = -np.sort(-posteriors, axis=1)
    return sorted_desc[:, 0] - sorted_desc[:, 1]


def state_trend_scores(means_df: pd.DataFrame) -> np.ndarray:
    """Raw trend_score per state -- identical arithmetic to infer_regime.
    characterize_state's trend_score (average of the 6 return columns),
    before any Uptrend/Downtrend/Ranging thresholding."""
    return_cols = [c for c in RETURN_COLS if c in means_df.columns]
    return means_df[return_cols].mean(axis=1).values


def state_trend_signs(means_df: pd.DataFrame) -> np.ndarray:
    """Sign of each state's trend_score, one value per state. Note this is
    +1/-1 even for a "Ranging" state whose trend_score is small in
    magnitude (e.g. +0.006) -- that sign is essentially noise, not a real
    directional call. Use `state_is_trending` to exclude those states from
    anything that needs a genuine trend, don't rely on this sign alone."""
    return np.sign(state_trend_scores(means_df))


def state_is_trending(means_df: pd.DataFrame) -> np.ndarray:
    """Boolean per state: True only for states characterize_state would
    actually call Uptrend/Downtrend (|trend_score| > TREND_Z_THRESHOLD),
    False for Ranging states. The persistence label is only meaningful for
    a state with a real directional characteristic -- a Ranging state's
    trend_sign is a coin flip, not a trend to gate."""
    return np.abs(state_trend_scores(means_df)) > TREND_Z_THRESHOLD


def duration_feature(duration_raw: np.ndarray) -> np.ndarray:
    """log1p transform of raw duration -- duration is an unbounded,
    heavy-tailed count (validated mean regime durations are ~7-15 bars, but
    the tail runs far longer), so this keeps the Logistic Regression's
    linear-in-log-odds assumption reasonable. THE ONE place this transform
    is defined -- persistence_dataset.py and persistence_gate.py both call
    this, never re-derive it inline."""
    return np.log1p(duration_raw)
