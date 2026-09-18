"""
build_regime_chart_data_15m.py -- same preprocessing as
build_regime_chart_data.py, for the 15m-HMM4 model from the fast timeframe
screening (research_archive/src/v22_fast_timeframe_screening.py).

Reuses, does NOT recompute:
  - results/model_15m_hmm4.pkl (already-fit model, from v22)
  - results/regime_15m_hmm4.parquet (already-decoded per-bar state/regime/
    confidence, from v22)

Only NEW computation here is cheap: rebuilding the 15m OHLCV candles (a
pandas resample of the raw 5m file, ~seconds, no HMM fit) and re-deriving
segments/transitions/state_duration from the already-decoded state sequence
-- consistent with this session's "reuse existing frozen results, do not
retrain" convention (the 3-way walk-forward test running in parallel is the
only thing in this project currently doing new HMM fitting).

Produces (does not touch or rename the existing 5m hmm4_/hmm6_ files):
    Data/chart/15m_hmm4_candles.parquet
    Data/chart/15m_hmm4_segments.parquet
    Data/chart/15m_hmm4_transitions.parquet
    Data/chart/models.json -- REWRITTEN to add model_id/timeframe/file_prefix
    metadata to all entries (5m ones included) and append this new model.

Run: `python scripts/build_regime_chart_data_15m.py` (from the project root).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "research_archive", "src"))

import persistence_common  # noqa: E402
import v22_fast_timeframe_screening as v22  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "Data", "chart")
RESULTS_DIR = os.path.join(_ROOT, "results")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_segments(candles: pd.DataFrame) -> pd.DataFrame:
    """Identical logic to build_regime_chart_data.py's own build_segments."""
    state = candles["state"]
    is_valid = state.notna()
    changed = np.empty(len(candles), dtype=bool)
    changed[0] = True
    changed[1:] = (state.values[1:] != state.values[:-1]) | (is_valid.values[1:] != is_valid.values[:-1])
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
    segs["duration_minutes"] = (segs["end_timestamp"] - segs["start_timestamp"]).dt.total_seconds() / 60 + 15
    segs.insert(0, "segment_id", np.arange(len(segs)))
    return segs


def build_transitions(segments: pd.DataFrame) -> pd.DataFrame:
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log("Loading raw 5m OHLCV and building genuine 15m candles (no HMM computation)...")
    raw5m = v22.load_raw_5m()
    agg15 = v22.build_agg_candles(raw5m, "15m")
    raw = agg15.reset_index().rename(columns={"index": "timestamp"})[["timestamp", "open", "high", "low", "close", "volume"]]

    log("Loading already-decoded 15m-HMM4 regime assignments (reused, not recomputed)...")
    regime_df = pd.read_parquet(os.path.join(RESULTS_DIR, "regime_15m_hmm4.parquet"))
    regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"], utc=True)

    bar_interval = pd.Timedelta(minutes=15)
    lengths = v22.segment_lengths_native(pd.DatetimeIndex(regime_df["timestamp"]), bar_interval)
    duration_raw = persistence_common.compute_running_duration(regime_df["state"].values, lengths)
    regime_df["state_duration"] = duration_raw

    merged = raw.merge(regime_df[["timestamp", "state", "regime", "confidence", "state_duration"]],
                        on="timestamp", how="left")
    n_ohlc, n_hmm = len(raw), len(regime_df)
    n_matched = int(merged["state"].notna().sum())
    unmatched_hmm = n_hmm - n_matched
    log(f"  15m OHLC rows: {n_ohlc:,}")
    log(f"  15m HMM rows: {n_hmm:,}")
    log(f"  Matched: {n_matched:,}")
    log(f"  Unmatched OHLC (gap/warmup, expected): {n_ohlc - n_matched:,}")
    log(f"  Unmatched HMM (should be 0): {unmatched_hmm:,}")
    if unmatched_hmm != 0:
        log(f"  WARNING: {unmatched_hmm} HMM-decoded 15m timestamps have no matching OHLC row -- investigate.")

    segments = build_segments(merged)
    transitions = build_transitions(segments)
    log(f"  {len(segments):,} contiguous state segments, {len(transitions):,} state transitions")

    merged.to_parquet(os.path.join(OUT_DIR, "15m_hmm4_candles.parquet"), index=False)
    segments.to_parquet(os.path.join(OUT_DIR, "15m_hmm4_segments.parquet"), index=False)
    transitions.to_parquet(os.path.join(OUT_DIR, "15m_hmm4_transitions.parquet"), index=False)

    model_stats = json.load(open(os.path.join(RESULTS_DIR, "model_stats_15m_hmm4.json")))
    new_entry = {
        "model_id": "15m_hmm4", "timeframe": "15m", "file_prefix": "15m_hmm4", "n_states": 4,
        "regime_labels": model_stats["regime_labels"],
        "n_candles": int(n_ohlc), "n_matched": int(n_matched), "n_segments": int(len(segments)),
        "n_transitions": int(len(transitions)),
        "date_range": [str(raw["timestamp"].min()), str(raw["timestamp"].max())],
        "train_end": str(raw["timestamp"].max()),
        "note": "From the fast timeframe screening (full-history development fit, NOT a walk-forward/OOS model).",
    }

    models_path = os.path.join(OUT_DIR, "models.json")
    meta = json.load(open(models_path))
    for m in meta["models"]:
        if "model_id" not in m:
            m["model_id"] = f"5m_hmm{m['n_states']}"
            m["timeframe"] = "5m"
            m["file_prefix"] = f"hmm{m['n_states']}"
    meta["models"] = [m for m in meta["models"] if m["model_id"] != "15m_hmm4"] + [new_entry]
    json.dump(meta, open(models_path, "w"), indent=2, default=str)
    log(f"Updated {models_path} -- models now: {[m['model_id'] for m in meta['models']]}")


if __name__ == "__main__":
    main()
