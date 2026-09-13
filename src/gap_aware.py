"""
gap_aware.py -- explicit-validity-mask alternative to build_feature_matrix's
blanket dropna(), for HMM fitting that must not treat a verified Binance
data gap as continuous market activity.

features.py is NOT modified. This module calls
features.build_feature_matrix(df, dropna=False) to get the full feature
matrix (every raw timestamp, NaN only where features.py's own logic already
produces it), then builds its own validity mask and segmentation on top.

WHY A NAIVE "ONLY DROP ROWS WITH A LITERAL NaN" MASK ISN'T ENOUGH:
Several features (vol_change, buy_sell_ratio) go NaN immediately at a
zero-volume bar via _safe_divide -- those are already caught by a plain
isna() check. But a *rolling* feature (vol_15m/vol_1h/vol_2h/skew_1h/
skew_4h/etc.) whose lookback window merely *includes* one or two
zero-volume placeholder bars does NOT necessarily go NaN -- it just
silently averages in a fabricated flat/zero observation alongside real
ones, understating volatility without ever producing a NaN to catch. That
is exactly the "gap treated as continuous" failure mode this experiment is
meant to eliminate, per the task's explicit requirement, so masking must be
based on window coverage, not literal NaN-ness.

VALIDITY RULE: a bar t is INVALID if a zero-volume raw bar exists anywhere
in [t - MAX_LOOKBACK_BARS + 1, t] (inclusive of t itself). MAX_LOOKBACK_BARS
= 48 (4h) is the longest lookback used by ANY feature in features.py
(ret_4h via price.diff(48), skew_4h via rolling(48)) -- so no feature at a
valid bar could have been computed from a window touching a gap, regardless
of whether that happened to produce a literal NaN.

SEGMENTATION FOR THE HMM: removing invalid bars from the middle of a
6-month window leaves the remaining valid bars non-contiguous in time.
Concatenating them into one sequence and fitting/scoring an HMM on it as
usual would silently model a "transition" across a multi-hour or
multi-day gap as if it were one ordinary 5-minute step -- exactly the
splicing problem this experiment must avoid. hmmlearn's native multi-
sequence support (a `lengths` array passed to .fit()/.score()/.predict())
is the correct mechanism: each maximal contiguous run of valid bars is
treated as an independent sequence, and no cross-segment transition is
ever modeled internally.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # features.py lives at project root
from features import BARS_PER, build_feature_matrix

MAX_LOOKBACK_BARS = BARS_PER["4h"]  # 48 -- longest lookback used by any feature (ret_4h, skew_4h)
BAR_INTERVAL = pd.Timedelta(minutes=5)


def build_validity_mask(raw_df: pd.DataFrame, feature_index: pd.DatetimeIndex,
                         max_lookback_bars: int = MAX_LOOKBACK_BARS) -> pd.Series:
    """
    Returns a boolean Series indexed like `feature_index`: True where that
    bar's features could not have been computed from a window touching any
    zero-volume raw bar. `raw_df` must be the FULL raw OHLCV+orderflow frame
    (same one features.build_feature_matrix consumes), so it can see
    zero-volume bars even ones that fall just before `feature_index` starts.
    """
    is_zero_vol = (raw_df["volume"] == 0).astype(int)
    # rolling max over the trailing max_lookback_bars window: 1 if a zero-volume
    # bar occurred anywhere in [t-max_lookback_bars+1, t], else 0.
    touched_by_gap = is_zero_vol.rolling(max_lookback_bars, min_periods=1).max()
    valid = (touched_by_gap == 0).reindex(feature_index)
    valid = valid.fillna(False)  # any feature_index timestamp not found in raw_df's window is untrusted
    return valid.astype(bool)


def segment_lengths(index: pd.DatetimeIndex) -> list:
    """Given an already-valid-only DatetimeIndex (non-contiguous in time
    wherever invalid bars were removed), return the length of each maximal
    contiguous (5-minute-spaced) run, in order."""
    if len(index) == 0:
        return []
    diffs = index.to_series().diff()
    new_segment = (diffs != BAR_INTERVAL)
    new_segment.iloc[0] = True
    segment_id = new_segment.cumsum()
    return segment_id.value_counts().sort_index().tolist()


def build_masked_features(raw_df: pd.DataFrame, model_input_cols: list) -> pd.DataFrame:
    """
    Full-history feature matrix (dropna=False) restricted to `model_input_cols`,
    with a `valid` boolean column attached. Does not slice by fold -- callers
    slice this by date range per fold, same as the existing pipeline does
    with features_out.csv.
    """
    feats = build_feature_matrix(raw_df, dropna=False)
    valid = build_validity_mask(raw_df, feats.index)
    out = feats[model_input_cols].copy()
    out["valid"] = valid.values
    return out


def valid_values_and_lengths(fold_df: pd.DataFrame, model_input_cols: list):
    """
    fold_df: a slice of build_masked_features()'s output for one fold's
    TRAIN or TEST date range (must include the `valid` column).
    Returns (valid_subset_df[model_input_cols], lengths, valid_index) ready
    to hand to scaling.fit_transform_fold and then to a GaussianHMM's
    fit/score/predict with lengths=... .

    `valid` (from build_validity_mask) only flags the zero-volume-gap
    criterion -- it does NOT catch NaNs from other causes, chiefly the
    warm-up period at the very start of the whole history (the first
    MAX_LOOKBACK_BARS-1 bars have no zero-volume bar nearby yet a rolling
    window that wide still can't be filled, e.g. skew_4h/ret_4h before
    2017-09-01 09:00). Feeding such a row into GaussianHMM.fit would raise
    ValueError: Input contains NaN. A row is only "genuinely invalid for
    the required features/model" (the task's own bar) if EITHER condition
    holds, so both are combined here rather than trusting `valid` alone.
    """
    valid_mask = fold_df["valid"] & fold_df[model_input_cols].notna().all(axis=1)
    valid_subset = fold_df[valid_mask]
    lengths = segment_lengths(valid_subset.index)
    assert sum(lengths) == len(valid_subset), "segment lengths must sum to the valid row count"
    return valid_subset[model_input_cols], lengths, valid_subset.index
