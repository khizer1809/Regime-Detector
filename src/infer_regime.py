"""
infer_regime.py -- loads a saved production model (Data/production_model_
hmm<N>.pkl) and answers "what regime does the HMM believe the market is in
right now?" using live data fetched directly from Binance.

Why this is causally safe for live use (unlike decoding a whole backtest
test-fold at once, which was flagged as a look-ahead risk during the audit
of the walk-forward pipeline): there is no future data to leak here -- we
only ever fetch candles up through the current moment, so Viterbi decoding
over that window cannot see anything a live system wouldn't already have.

CONTEXT_BARS controls how much recent history is fed into the decode
alongside the newest bar. This is NOT about needing lookback for feature
computation (that only needs the ~48-bar/4h max rolling window, always
included) -- it is about giving the HMM's own state-filtering enough bars
to "forget" an arbitrary starting assumption and settle onto the
distribution the data actually supports, before trusting its answer at
the very last bar. A few hundred bars is generous relative to the ~7-15
bar mean regime durations observed during validation (see
Data/rolling_vs_expanding_final_report.md).

Gap-aware: if the fetched window contains a zero-volume/gap bar, the same
masking + segment-aware (`lengths=`) approach used throughout this project
is applied, so no transition is decoded across a genuine data gap. If the
most recent bar itself is invalid (e.g. a live zero-volume moment), that
is reported explicitly rather than guessed at.

Run: `python infer_regime.py [N_STATES] [CONTEXT_BARS]` (from src/).
Defaults: N_STATES=4, CONTEXT_BARS=2000 (~7 days of 5m bars).
"""

import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # features.py at project root

from binance_fetch import fetch_recent
from features import build_feature_matrix
from gap_aware import MAX_LOOKBACK_BARS, build_validity_mask, segment_lengths

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CONTEXT_BARS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

RETURN_COLS = ["ret_5m", "ret_15m", "ret_30m", "ret_1h", "ret_2h", "ret_4h"]
VOL_COLS = ["vol_15m", "vol_1h", "vol_2h"]
TREND_Z_THRESHOLD = 0.3
VOL_Z_THRESHOLD = 0.3


def characterize_state(state_mean: pd.Series) -> str:
    """Minimal standalone re-implementation of the (now-removed research-
    only) regime_labeling.characterize_state -- same thresholds/logic,
    kept small since production only needs the label, not the full
    diagnostic report those research scripts produced."""
    return_cols = [c for c in RETURN_COLS if c in state_mean.index]
    vol_cols = [c for c in VOL_COLS if c in state_mean.index]
    trend_score = float(state_mean[return_cols].mean()) if return_cols else 0.0
    vol_score = float(state_mean[vol_cols].mean()) if vol_cols else 0.0

    if trend_score > TREND_Z_THRESHOLD:
        trend = "Uptrend"
    elif trend_score < -TREND_Z_THRESHOLD:
        trend = "Downtrend"
    else:
        trend = "Ranging"

    if vol_score > VOL_Z_THRESHOLD:
        vol = "High-Vol"
    elif vol_score < -VOL_Z_THRESHOLD:
        vol = "Low-Vol"
    else:
        vol = "Mid-Vol"

    return f"{trend} / {vol}"


def load_artifact(n_states: int) -> dict:
    path = os.path.join(_ROOT, "Data", f"production_model_hmm{n_states}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No production model found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def decode_current_regime(n_states: int = None, context_bars: int = None):
    """Fetches live data and decodes the current regime -- the same fetch->
    features->mask->scale->predict pipeline `infer()` prints, but returned
    as a dict for reuse (e.g. by persistence_gate.py) instead of only being
    printed. Returns None if the most recent bar is invalid (zero-volume/
    gap or NaN feature): there is nothing to report for it, same condition
    that used to end `infer()` early with the "[UNAVAILABLE]" message."""
    n_states = N_STATES if n_states is None else n_states
    context_bars = CONTEXT_BARS if context_bars is None else context_bars

    artifact = load_artifact(n_states)
    model, scaler, feature_cols = artifact["model"], artifact["scaler"], artifact["feature_columns"]
    print(f"Loaded {n_states}-state production model "
          f"(trained {artifact['train_start']} -> {artifact['train_end']}, "
          f"{artifact['n_train_valid']:,} valid bars, strategy={artifact['training_strategy']})", flush=True)

    n_fetch = context_bars + MAX_LOOKBACK_BARS + 12  # + feature lookback + small margin
    print(f"\nFetching last {n_fetch} bars from Binance ({context_bars} for decode context "
          f"+ {MAX_LOOKBACK_BARS}-bar feature lookback)...", flush=True)
    raw = fetch_recent(n_fetch)
    print(f"Fetched {len(raw)} bars, {raw.index.min()} -> {raw.index.max()}", flush=True)

    feats = build_feature_matrix(raw, dropna=False)
    valid = build_validity_mask(raw, feats.index)
    feats = feats[feature_cols].copy()
    feats["valid"] = valid.values

    context = feats.iloc[-context_bars:] if len(feats) > context_bars else feats

    last_bar_ts = context.index[-1]
    last_bar_valid = bool(context["valid"].iloc[-1])

    if not last_bar_valid:
        print(f"\n[UNAVAILABLE] The most recent bar ({last_bar_ts}) is INVALID for this model "
              f"(touches a zero-volume/gap window, or has a NaN feature) -- no regime can be "
              f"reported for it. This is a genuine data-availability gap, not an error.")
        return None

    valid_context = context[context["valid"]]
    lengths = segment_lengths(valid_context.index)

    scaled = pd.DataFrame(scaler.transform(valid_context[feature_cols]),
                           index=valid_context.index, columns=feature_cols)
    states = model.predict(scaled.values, lengths=lengths)
    posteriors = model.predict_proba(scaled.values, lengths=lengths)

    current_state = int(states[-1])
    confidence = float(posteriors[-1, current_state])

    means_df = pd.DataFrame(model.means_, columns=feature_cols)
    label = characterize_state(means_df.iloc[current_state])

    return {
        "artifact": artifact, "model": model, "scaler": scaler, "feature_cols": feature_cols,
        "valid_context": valid_context, "lengths": lengths, "states": states, "posteriors": posteriors,
        "current_state": current_state, "confidence": confidence, "last_bar_ts": last_bar_ts,
        "label": label, "means_df": means_df,
    }


def infer():
    decode = decode_current_regime(N_STATES, CONTEXT_BARS)
    if decode is None:
        return

    print(f"\n{'='*60}")
    print(f"CURRENT REGIME as of {decode['last_bar_ts']}")
    print(f"{'='*60}")
    print(f"  State: {decode['current_state']} (of {N_STATES})")
    print(f"  Label: {decode['label']}")
    print(f"  Posterior confidence: {decode['confidence']:.1%}")
    print(f"  Decoded using {len(decode['valid_context'])} valid bars of context "
          f"({len(decode['lengths'])} contiguous segment(s))")

    print(f"\nAll {N_STATES} states, for reference:")
    for s in range(N_STATES):
        marker = " <- current" if s == decode['current_state'] else ""
        print(f"  state {s}: {characterize_state(decode['means_df'].iloc[s])}{marker}")


if __name__ == "__main__":
    infer()
