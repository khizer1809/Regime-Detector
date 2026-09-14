"""
persistence_dataset.py -- one-shot full-history decode of
Data/features_out_masked.csv with the current production model
(Data/production_model_hmm4.pkl), building a labeled training table for the
persistence gate.

Label definition (outcome-based): for bar t, label=1 if the forward 24-bar
(~2h) return continues in the SAME direction as bar t's decoded state's own
trend sign (persistence_common.state_trend_signs, same arithmetic as
infer_regime.characterize_state's trend_score), label=0 otherwise. Only bars
in a genuinely trending state (Uptrend/Downtrend, |trend_score| >
TREND_Z_THRESHOLD) are used -- a Ranging state's trend_sign is a near-zero,
essentially arbitrary sign (e.g. +0.006), not a real directional call, and
an earlier run confirmed including it dilutes the label to near-uninformative
(holdout AUC 0.511). Rows whose forward return magnitude falls in the
smallest ~10% (noise-dominated near-zero drift) are also excluded via a
data-driven epsilon deadband -- not because their label is wrong, but
because it isn't a meaningful outcome to learn from.

ACCEPTED LOOK-AHEAD LIMITATION (documented, not fixed here): this decodes
the ENTIRE history with a single production model that was itself fit on
all history through train_end. For an old bar (e.g. 2018), confidence/margin
are therefore somewhat "cleaner" than what a live gate would actually have
seen back then, since the model decoding it already "knows" the future.
This is an explicit, accepted tradeoff (fast: seconds, not a multi-hour
walk-forward refit) -- duration is unaffected, since it depends only on the
decoded path's own run-lengths, not on model quality.

Run: `python persistence_dataset.py [N_STATES]` (from src/). Writes to
Data/persistence_training_table.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import persistence_common
from gap_aware import valid_values_and_lengths
from infer_regime import load_artifact
from save_production_model import load_masked_features

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(_ROOT, "Data", "persistence_training_table.csv")

N_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
FORWARD_RET_COL = "ret_2h"  # log(close).diff(24) -- exact match to LABEL_HORIZON_BARS=24
EPSILON_PERCENTILE = 10


def decode_full_history(n_states: int = N_STATES):
    artifact = load_artifact(n_states)
    df, model_cols = load_masked_features()
    if model_cols != artifact["feature_columns"]:
        raise ValueError(
            f"Feature schema mismatch: features_out_masked.csv gives {model_cols}, "
            f"but production_model_hmm{n_states}.pkl was trained on {artifact['feature_columns']}"
        )
    valid_df, lengths, valid_index = valid_values_and_lengths(df, model_cols)

    scaled = artifact["scaler"].transform(valid_df.values)  # TRANSFORM ONLY -- never refit here
    states = artifact["model"].predict(scaled, lengths=lengths)
    posteriors = artifact["model"].predict_proba(scaled, lengths=lengths)
    return artifact, valid_df, lengths, valid_index, states, posteriors


def compute_epsilon(forward_ret_valid_only: pd.Series, percentile: int = EPSILON_PERCENTILE) -> float:
    return float(np.percentile(forward_ret_valid_only.abs(), percentile))


def build_training_table(n_states: int = N_STATES) -> pd.DataFrame:
    print(f"Decoding full history with production_model_hmm{n_states}.pkl...", flush=True)
    artifact, valid_df, lengths, valid_index, states, posteriors = decode_full_history(n_states)
    n = len(states)
    print(f"  decoded {n:,} valid bars ({len(lengths)} contiguous segments)", flush=True)

    means_df = pd.DataFrame(artifact["model"].means_, columns=artifact["feature_columns"])

    confidence = persistence_common.compute_confidence(posteriors, states)
    stay_prob = persistence_common.compute_stay_prob(states, artifact["model"].transmat_)
    duration_raw = persistence_common.compute_running_duration(states, lengths)
    margin = persistence_common.compute_margin(posteriors)
    trend_sign_per_state = persistence_common.state_trend_signs(means_df)
    is_trending_per_state = persistence_common.state_is_trending(means_df)
    trend_sign = trend_sign_per_state[states]
    is_trending = is_trending_per_state[states]
    print(f"  trending states: {[s for s in range(n_states) if is_trending_per_state[s]]} "
          f"of {n_states} (Ranging states excluded from labeling)", flush=True)

    seg_id = persistence_common.segment_ids_from_lengths(lengths)
    i = np.arange(n)
    j = i + persistence_common.LABEL_HORIZON_BARS
    forward_valid = (j < n) & (seg_id[np.minimum(j, n - 1)] == seg_id)
    forward_ret = valid_df[FORWARD_RET_COL].shift(-persistence_common.LABEL_HORIZON_BARS).values

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
        "state": states,
        "confidence": confidence,
        "stay_prob": stay_prob,
        "duration": duration_raw,
        "margin": margin,
        "trend_sign": trend_sign,
        "is_trending": is_trending,
        "forward_ret": forward_ret,
        "forward_valid": forward_valid,
        "usable": usable,
        "label": label_out,
    }, index=valid_index)

    out.to_csv(OUT_PATH)
    print(f"\nSaved training table to:\n  {OUT_PATH}", flush=True)
    return out


if __name__ == "__main__":
    build_training_table()
