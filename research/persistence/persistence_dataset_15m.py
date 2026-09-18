"""
persistence_dataset_15m.py -- 15m-HMM4 counterpart of persistence_dataset.py.
One-shot full-history decode of genuine 15m candles with
Data/production_model_15m_hmm4.pkl, building a labeled training table for
the persistence gate, using the EXACT same labeling design already
validated for the 5m gate (persistence_dataset.py's own docstring):
trend-states-only (Ranging excluded -- diluted an earlier 5m run to
near-uninformative, holdout AUC 0.511) + a data-driven epsilon deadband on
the smallest ~10% of forward-return magnitudes.

TIMEFRAME-CORRECTED CONSTANTS (the one real difference from the 5m script):
persistence_common.LABEL_HORIZON_BARS=24 means ~2h ONLY at 5m granularity
(24 x 5min). Reusing it unmodified for 15m data would silently turn the
"~2h persistence" label into a ~6h one. This script instead uses
LABEL_HORIZON_BARS_15M=8 (8 x 15min = 2h, matching the ORIGINAL economic
horizon, and exactly the "ret_2h" column already in the 15m feature set --
same pattern as v22_fast_timeframe_screening.py's RETURN_HORIZON_BARS).

ACCEPTED LOOK-AHEAD LIMITATION (same as the 5m script, not fixed here):
decodes the entire history with a single production model fit on all
history through train_end -- confidence/margin for old bars are somewhat
cleaner than a live gate would have seen at the time; duration is
unaffected.

Run: `python persistence_dataset_15m.py` (from src/). Writes
Data/persistence_training_table_15m.csv.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "research_archive", "src"))

import persistence_common  # noqa: E402
import v22_fast_timeframe_screening as v22  # noqa: E402

ARTIFACT_PATH = os.path.join(_ROOT, "Data", "production_model_15m_hmm4.pkl")
OUT_PATH = os.path.join(_ROOT, "Data", "persistence_training_table_15m.csv")

TIMEFRAME = "15m"
N_STATES = 4
LABEL_HORIZON_BARS_15M = 8  # 8 x 15min = 2h -- the SAME real-world horizon as the 5m gate's 24 bars
FORWARD_RET_COL = "ret_2h"  # already an 8-bar column in the 15m feature set (v22.RETURN_HORIZON_BARS["15m"])
EPSILON_PERCENTILE = 10


def decode_full_history():
    with open(ARTIFACT_PATH, "rb") as f:
        artifact = pickle.load(f)
    raw5m = v22.load_raw_5m()
    agg = v22.build_agg_candles(raw5m, TIMEFRAME)
    feat_df = v22.build_tf_features(agg, TIMEFRAME)
    if list(feat_df.columns) != artifact["feature_columns"]:
        raise ValueError("Feature schema mismatch between rebuilt 15m features and production_model_15m_hmm4.pkl")
    max_lookback = max(v22.EMA_SLOW_BARS, max(v22.RETURN_HORIZON_BARS[TIMEFRAME].values()))
    valid_flag = v22.build_validity_mask_native(agg, feat_df.index, max_lookback)
    valid_mask = valid_flag & feat_df.notna().all(axis=1)
    valid_index = feat_df.index[valid_mask]
    valid_df = feat_df.loc[valid_index]
    bar_interval = pd.Timedelta(minutes=v22.TF_MINUTES[TIMEFRAME])
    lengths = v22.segment_lengths_native(valid_index, bar_interval)
    assert sum(lengths) == len(valid_df)

    scaled = artifact["scaler"].transform(valid_df.values)  # TRANSFORM ONLY -- never refit here
    states = artifact["model"].predict(scaled, lengths=lengths)
    posteriors = artifact["model"].predict_proba(scaled, lengths=lengths)
    return artifact, valid_df, lengths, valid_index, states, posteriors


def compute_epsilon(forward_ret_valid_only: pd.Series, percentile: int = EPSILON_PERCENTILE) -> float:
    return float(np.percentile(forward_ret_valid_only.abs(), percentile))


def build_training_table() -> pd.DataFrame:
    print(f"Decoding full 15m history with production_model_15m_hmm4.pkl...", flush=True)
    artifact, valid_df, lengths, valid_index, states, posteriors = decode_full_history()
    n = len(states)
    print(f"  decoded {n:,} valid 15m bars ({len(lengths)} contiguous segments)", flush=True)

    means_df = pd.DataFrame(artifact["model"].means_, columns=artifact["feature_columns"])

    confidence = persistence_common.compute_confidence(posteriors, states)
    stay_prob = persistence_common.compute_stay_prob(states, artifact["model"].transmat_)
    duration_raw = persistence_common.compute_running_duration(states, lengths)
    margin = persistence_common.compute_margin(posteriors)
    trend_sign_per_state = persistence_common.state_trend_signs(means_df)
    is_trending_per_state = persistence_common.state_is_trending(means_df)
    trend_sign = trend_sign_per_state[states]
    is_trending = is_trending_per_state[states]
    print(f"  trending states: {[s for s in range(N_STATES) if is_trending_per_state[s]]} "
          f"of {N_STATES} (Ranging states excluded from labeling)", flush=True)

    seg_id = persistence_common.segment_ids_from_lengths(lengths)
    i = np.arange(n)
    j = i + LABEL_HORIZON_BARS_15M
    forward_valid = (j < n) & (seg_id[np.minimum(j, n - 1)] == seg_id)
    forward_ret = valid_df[FORWARD_RET_COL].shift(-LABEL_HORIZON_BARS_15M).values

    n_forward_dropped = int((~forward_valid).sum())
    print(f"  forward-valid: {int(forward_valid.sum()):,} / {n:,}  "
          f"(dropped {n_forward_dropped:,} at segment tails)", flush=True)

    trending_and_forward_valid = forward_valid & is_trending
    eps_source = pd.Series(forward_ret[trending_and_forward_valid & ~np.isnan(forward_ret)])
    epsilon = compute_epsilon(eps_source)
    print(f"  epsilon (p{EPSILON_PERCENTILE} of |{FORWARD_RET_COL}|, trending+forward-valid rows) = {epsilon:.6f}", flush=True)

    usable = trending_and_forward_valid & (np.abs(forward_ret) > epsilon)
    n_ranging_dropped = int((forward_valid & ~is_trending).sum())
    n_deadband_dropped = int((trending_and_forward_valid & (np.abs(forward_ret) <= epsilon)).sum())
    print(f"  dropped as Ranging (not trending): {n_ranging_dropped:,}", flush=True)
    print(f"  dropped by epsilon deadband: {n_deadband_dropped:,}", flush=True)
    print(f"  usable rows: {int(usable.sum()):,}", flush=True)

    label = np.where(np.sign(forward_ret) == trend_sign, 1, 0)
    label_out = np.where(usable, label, np.nan)

    pos_rate_all = float(np.nanmean(label_out))
    pos_rate_up = float(np.nanmean(np.where(usable & (trend_sign > 0), label_out, np.nan)))
    pos_rate_down = float(np.nanmean(np.where(usable & (trend_sign < 0), label_out, np.nan)))
    print(f"  label positive rate: overall={pos_rate_all:.3f}  "
          f"trend_sign=+1: {pos_rate_up:.3f}  trend_sign=-1: {pos_rate_down:.3f}", flush=True)

    out = pd.DataFrame({
        "state": states, "confidence": confidence, "stay_prob": stay_prob, "duration": duration_raw,
        "margin": margin, "trend_sign": trend_sign, "is_trending": is_trending,
        "forward_ret": forward_ret, "forward_valid": forward_valid, "usable": usable, "label": label_out,
    }, index=valid_index)

    out.to_csv(OUT_PATH)
    print(f"\nSaved training table to:\n  {OUT_PATH}", flush=True)
    return out


if __name__ == "__main__":
    build_training_table()
