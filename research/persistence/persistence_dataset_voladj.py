"""
persistence_dataset_voladj.py -- EXPERIMENTAL, NOT PRODUCTION.

Volatility-adjusted variant of persistence_dataset.py's target. Reuses
persistence_dataset.decode_full_history() directly (byte-identical HMM/
scaler/states/posteriors -- no HMM change, no refit, no redecode logic
duplicated) and changes exactly one thing: how "a meaningful move" is
defined.

    BASELINE:     |forward_ret| > epsilon                (fixed constant)
    EXPERIMENTAL: |forward_ret / past_volatility| > k     (volatility-normalized)

Everything else -- HMM conditioning (trend_sign gates direction), state
eligibility (is_trending), horizon (24 bars), sign-agreement logic for the
label -- is untouched.

CAUSAL VOLATILITY: past_volatility[t] = rolling std of ret_5m over the
trailing 24 bars ending at and including t (same length as the label
horizon -- "how volatile has the market been over a window the same size
as the one we're forecasting" is the most directly comparable choice to
the horizon itself), computed per gap-aware segment (never crosses a gap),
using ONLY ret_5m values at positions <= t. Never uses forward_ret or any
bar after t.

THRESHOLD k: NOT hand-picked. Computed with the exact same rule the
baseline's own epsilon uses (10th percentile of the magnitude distribution,
among trending + forward-valid rows) -- the only change is the *population*
it's computed over: TRAINING-PERIOD-ONLY (strictly before the same
HOLDOUT_START/embargo boundary train_persistence_gate.py uses), never the
holdout. This is a direct methodological analog of epsilon, not a new,
arbitrarily chosen rule -- and it is stricter about leakage than the
baseline's own epsilon, which was computed over the full history (a
pre-existing, minor, now-documented asymmetry -- see the leakage audit in
the final report).

Output: Data/persistence_training_table_voladj.csv -- a NEW file. Does not
touch Data/persistence_training_table.csv or Data/persistence_gate.pkl.

Run: `python persistence_dataset_voladj.py` (from src/).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import persistence_common
import persistence_dataset  # reused directly, not reimplemented

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(_ROOT, "Data", "persistence_training_table_voladj.csv")

VOL_WINDOW_BARS = persistence_common.LABEL_HORIZON_BARS  # 24 -- same length as the forecast horizon
K_PERCENTILE = 10  # identical rule to the baseline's EPSILON_PERCENTILE

# must match train_persistence_gate.py exactly, so "training-only" means the same thing here as there
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
EMBARGO_BARS = 48


def compute_causal_volatility(valid_df: pd.DataFrame, lengths: list) -> np.ndarray:
    """Rolling std of ret_5m over the trailing VOL_WINDOW_BARS bars ending at
    and including each row, computed independently within each gap-aware
    segment (a window never reaches across a genuine data gap). Uses ONLY
    ret_5m values at or before the row itself -- causal by construction."""
    seg_id = persistence_common.segment_ids_from_lengths(lengths)
    ret5 = pd.Series(valid_df["ret_5m"].values)
    vol = ret5.groupby(seg_id).rolling(VOL_WINDOW_BARS, min_periods=VOL_WINDOW_BARS).std()
    vol.index = vol.index.droplevel(0)
    return vol.reindex(range(len(valid_df))).values


def main():
    print(f"Decoding full history (reusing persistence_dataset.decode_full_history -- "
          f"same HMM, same scaler.transform()-only, no refit/redecode)...", flush=True)
    artifact, valid_df, lengths, valid_index, states, posteriors = persistence_dataset.decode_full_history()
    n = len(states)
    print(f"  decoded {n:,} valid bars ({len(lengths)} segments) -- identical to baseline by construction", flush=True)

    means_df = pd.DataFrame(artifact["model"].means_, columns=artifact["feature_columns"])
    confidence = persistence_common.compute_confidence(posteriors, states)
    stay_prob = persistence_common.compute_stay_prob(states, artifact["model"].transmat_)
    duration_raw = persistence_common.compute_running_duration(states, lengths)
    margin = persistence_common.compute_margin(posteriors)
    trend_sign_per_state = persistence_common.state_trend_signs(means_df)
    is_trending_per_state = persistence_common.state_is_trending(means_df)
    trend_sign = trend_sign_per_state[states]
    is_trending = is_trending_per_state[states]

    seg_id = persistence_common.segment_ids_from_lengths(lengths)
    i = np.arange(n)
    j = i + persistence_common.LABEL_HORIZON_BARS
    forward_valid = (j < n) & (seg_id[np.minimum(j, n - 1)] == seg_id)
    forward_ret = valid_df[persistence_dataset.FORWARD_RET_COL].shift(-persistence_common.LABEL_HORIZON_BARS).values

    print("Computing causal past volatility (trailing 24-bar std of ret_5m, gap-aware, backward-only)...", flush=True)
    past_volatility = compute_causal_volatility(valid_df, lengths)
    vol_defined = ~np.isnan(past_volatility) & (past_volatility > 0)

    normalized_move = np.where(vol_defined, forward_ret / past_volatility, np.nan)
    trending_forward_valid_vol = is_trending & forward_valid & vol_defined

    # k computed ONLY from training-period rows (strictly before the same holdout/embargo
    # boundary train_persistence_gate.py uses) -- never from holdout, exact same percentile
    # RULE as the baseline's own epsilon, applied to a different (training-only) population.
    embargo_start = HOLDOUT_START - pd.Timedelta(minutes=5 * EMBARGO_BARS)
    is_train_period = np.array(valid_index) < embargo_start
    k_source_mask = trending_forward_valid_vol & is_train_period
    k_source = np.abs(normalized_move[k_source_mask])
    k_source = k_source[~np.isnan(k_source)]
    k = float(np.percentile(k_source, K_PERCENTILE))
    print(f"  k (p{K_PERCENTILE} of |normalized_move|, TRAINING-PERIOD-ONLY rows, "
          f"train_end={embargo_start}): {k:.4f}", flush=True)
    print(f"  (i.e. a forward-2h move must exceed {k:.2f} standard deviations of the trailing "
          f"2h volatility to count as meaningful)", flush=True)

    usable = trending_forward_valid_vol & (np.abs(normalized_move) > k)
    print(f"  usable rows: {int(usable.sum()):,} (baseline had 405,587 usable rows for comparison)", flush=True)

    label = np.where(np.sign(forward_ret) == trend_sign, 1, 0)
    label_out = np.where(usable, label, np.nan)
    print(f"  label positive rate: {float(np.nanmean(label_out)):.3f}", flush=True)

    out = pd.DataFrame({
        "state": states, "confidence": confidence, "stay_prob": stay_prob, "duration": duration_raw,
        "margin": margin, "trend_sign": trend_sign, "is_trending": is_trending,
        "forward_ret": forward_ret, "forward_valid": forward_valid,
        "past_volatility": past_volatility, "normalized_move": normalized_move,
        "k_threshold": k, "usable": usable, "label": label_out,
    }, index=valid_index)

    out.to_csv(OUT_PATH)
    print(f"\nSaved EXPERIMENTAL training table to:\n  {OUT_PATH}", flush=True)
    print("(Data/persistence_training_table.csv and Data/persistence_gate.pkl untouched)", flush=True)
    return out, k


if __name__ == "__main__":
    main()
