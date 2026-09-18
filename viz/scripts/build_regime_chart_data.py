"""
build_regime_chart_data.py -- preprocessing for the HMM regime visualization
app. Builds an efficient, pre-joined, pre-segmented dataset for the
TradingView-style chart, reusing the EXISTING production pipeline verbatim:

  - src/infer_regime.py's load_artifact() / characterize_state() -- the
    project's own, already-established state -> regime-name mapping
    (TREND_Z_THRESHOLD/VOL_Z_THRESHOLD = 0.3, same as production).
  - src/persistence_dataset.py's decode_full_history() -- the project's own
    one-shot full-history decode (scaler.transform()-only, never refit) of
    the current production HMM (Data/production_model_hmm<N>.pkl).
  - src/persistence_common.py's compute_confidence/compute_running_duration/
    segment_ids_from_lengths -- unchanged.

Does NOT retrain or modify the HMM, features, scaling, or any research
result. Does NOT touch the raw OHLCV source file. Reads-only.

CANONICAL DATA SOURCE DECISION (documented, not silent): candlesticks come
from the raw 5m OHLCV file already used elsewhere in this project this
session for causal feature work (BTCUSDT_5M.csv -- verified this session to
be perfectly contiguous, 933,924 5-minute bars, zero missing rows,
2017-09-01 -> 2026-07-18). HMM state assignments come from decode_full_
history(), which operates on Data/features_out_masked.csv (the same
933,924-row, gap-aware-masked feature matrix the production model was
fit on) -- so both sources share the same underlying timestamp range,
confirmed by explicit merge-and-report below, not assumed.

Produces, per available HMM model (HMM-4, and HMM-6 if its feature schema
matches the current production feature set):
    Data/chart/hmm<N>_candles.parquet     -- one row per RAW 5m bar (OHLCV +
                                              state/regime/confidence/duration,
                                              NaN state over genuine gaps --
                                              never forward-filled)
    Data/chart/hmm<N>_segments.parquet    -- one row per contiguous state run
    Data/chart/hmm<N>_transitions.parquet -- one row per state change
    Data/chart/models.json                -- which models are available +
                                              their regime legend

Run: `python scripts/build_regime_chart_data.py` (from the project root).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import persistence_common  # noqa: E402
from infer_regime import characterize_state, load_artifact  # noqa: E402
from persistence_dataset import decode_full_history  # noqa: E402

RAW_OHLCV_PATH = r"C:\Users\MohammedkhezerK\Downloads\New System\ICT_STRUCTURE_RESEARCH\data\BTCUSDT_5M.csv"
OUT_DIR = os.path.join(_ROOT, "Data", "chart")
CANDIDATE_N_STATES = [4, 6]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_raw_ohlcv() -> pd.DataFrame:
    log(f"Loading raw OHLCV from {RAW_OHLCV_PATH} ...")
    df = pd.read_csv(RAW_OHLCV_PATH, usecols=["timestamp", "open", "high", "low", "close", "volume"],
                      parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    log(f"  {len(df):,} raw candles, {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df


def build_segments(candles: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of the same non-null state. A run breaks on a state
    change OR whenever state becomes null (a genuine data gap) -- so a gap
    never silently joins two segments, and no state is ever forward-filled
    across it (Section 22)."""
    state = candles["state"]
    is_valid = state.notna()
    changed = np.empty(len(candles), dtype=bool)
    changed[0] = True
    changed[1:] = (state.values[1:] != state.values[:-1]) | (is_valid.values[1:] != is_valid.values[:-1])
    # also treat NaN != NaN comparisons correctly: two consecutive NaNs must NOT count as "changed"
    both_nan = (~is_valid.values[1:]) & (~is_valid.values[:-1])
    changed[1:] = changed[1:] & ~both_nan
    run_id = np.cumsum(changed)

    tmp = candles.assign(run_id=run_id)
    valid_runs = tmp[tmp["state"].notna()]
    segs = valid_runs.groupby("run_id").agg(
        state=("state", "first"), regime=("regime", "first"),
        start_timestamp=("timestamp", "first"), end_timestamp=("timestamp", "last"),
        number_of_bars=("timestamp", "size"),
    ).reset_index(drop=True)
    segs["duration_minutes"] = (segs["end_timestamp"] - segs["start_timestamp"]).dt.total_seconds() / 60 + 5
    segs.insert(0, "segment_id", np.arange(len(segs)))
    return segs


def build_transitions(segments: pd.DataFrame) -> pd.DataFrame:
    """One row per state change between consecutive valid segments (gaps in
    between don't erase the transition -- Section 8)."""
    if len(segments) < 2:
        return pd.DataFrame(columns=["timestamp", "from_state", "from_regime", "to_state", "to_regime"])
    rows = []
    for i in range(1, len(segments)):
        prev, cur = segments.iloc[i - 1], segments.iloc[i]
        if prev["state"] != cur["state"]:
            rows.append({
                "timestamp": cur["start_timestamp"], "from_state": int(prev["state"]), "from_regime": prev["regime"],
                "to_state": int(cur["state"]), "to_regime": cur["regime"],
            })
    return pd.DataFrame(rows)


def build_for_model(n_states: int, raw: pd.DataFrame) -> dict | None:
    log(f"=== HMM-{n_states} ===")
    try:
        artifact, valid_df, lengths, valid_index, states, posteriors = decode_full_history(n_states)
    except FileNotFoundError:
        log(f"  Data/production_model_hmm{n_states}.pkl not found -- skipping")
        return None
    except ValueError as e:
        log(f"  feature schema mismatch, skipping HMM-{n_states}: {e}")
        return None

    means_df = pd.DataFrame(artifact["model"].means_, columns=artifact["feature_columns"])
    regime_labels = [characterize_state(means_df.iloc[s]) for s in range(n_states)]
    log(f"  state -> regime mapping: {dict(enumerate(regime_labels))}")

    confidence = persistence_common.compute_confidence(posteriors, states)
    duration_raw = persistence_common.compute_running_duration(states, lengths)

    state_df = pd.DataFrame({
        "timestamp": pd.DatetimeIndex(valid_index), "state": states.astype(float),
        "regime": [regime_labels[s] for s in states], "confidence": confidence, "state_duration": duration_raw,
    })

    # --- explicit timestamp alignment (Section 21) -- never assume row order ---
    merged = raw.merge(state_df, on="timestamp", how="left")
    n_ohlc, n_hmm = len(raw), len(state_df)
    n_matched = int(merged["state"].notna().sum())
    unmatched_hmm = n_hmm - n_matched
    log(f"  OHLC rows: {n_ohlc:,}")
    log(f"  HMM rows: {n_hmm:,}")
    log(f"  Matched: {n_matched:,}")
    log(f"  Unmatched OHLC (no HMM state -- gap/warmup, expected): {n_ohlc - n_matched:,}")
    log(f"  Unmatched HMM (HMM timestamp missing from raw OHLC -- should be 0): {unmatched_hmm:,}")
    if unmatched_hmm != 0:
        log(f"  WARNING: {unmatched_hmm} HMM-decoded timestamps have no matching raw OHLC row -- investigate, "
            f"not silently dropped.")

    segments = build_segments(merged)
    transitions = build_transitions(segments)
    log(f"  {len(segments):,} contiguous state segments, {len(transitions):,} state transitions")

    merged.to_parquet(os.path.join(OUT_DIR, f"hmm{n_states}_candles.parquet"), index=False)
    segments.to_parquet(os.path.join(OUT_DIR, f"hmm{n_states}_segments.parquet"), index=False)
    transitions.to_parquet(os.path.join(OUT_DIR, f"hmm{n_states}_transitions.parquet"), index=False)

    return {
        "n_states": n_states, "regime_labels": {int(s): r for s, r in enumerate(regime_labels)},
        "n_candles": int(n_ohlc), "n_matched": int(n_matched), "n_segments": int(len(segments)),
        "n_transitions": int(len(transitions)),
        "date_range": [str(raw["timestamp"].min()), str(raw["timestamp"].max())],
        "train_end": artifact.get("train_end"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = load_raw_ohlcv()

    models = []
    for n_states in CANDIDATE_N_STATES:
        info = build_for_model(n_states, raw)
        if info is not None:
            models.append(info)

    if not models:
        raise RuntimeError("No HMM model could be decoded -- check Data/production_model_hmm*.pkl and "
                            "Data/features_out_masked.csv exist and their feature schemas match.")

    json.dump({"models": models}, open(os.path.join(OUT_DIR, "models.json"), "w"), indent=2, default=str)
    log(f"\nAvailable models: {[m['n_states'] for m in models]}")
    log(f"All artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
