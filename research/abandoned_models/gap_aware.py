import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # features.py lives at project root
from features import BARS_PER, build_feature_matrix

MAX_LOOKBACK_BARS = BARS_PER["4h"]  # 48 -- longest lookback used by any feature (ret_4h, skew_4h)
BAR_INTERVAL = pd.Timedelta(minutes=5)


def build_validity_mask(raw_df, feature_index, max_lookback_bars=48):

    is_zero_vol = (raw_df["volume"] == 0)
    touched_by_gap = is_zero_vol.rolling(48, min_periods=1).max()
    valid = (touched_by_gap == 0)
    valid = valid.reindex(feature_index)
    valid = valid.fillna(False)

    return valid


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
