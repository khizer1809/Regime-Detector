"""
build_overview_chart_data.py -- full-history (2017-2026) daily-aggregated
overview for the regime viz app, so the whole dataset can be viewed at a
glance instead of only a 30-day window.

Rendering all ~934,000 5m candles (and their tens of thousands of regime-
band divs) at once would be both slow and visually meaningless (each pixel
would represent hundreds of candles). Instead this resamples to ONE candle
per day (~3,242 candles for the full history -- trivially small) and colors
each day by whichever regime occupied the MOST 5m bars that day (the day's
statistical mode, not just its closing regime).

Reuses, does not recompute: Data/chart/hmm4_candles.parquet (the existing
5m-HMM4 decode) -- purely a re-aggregation of already-decoded output, no
HMM/model computation.

Produces:
    Data/chart/5m_hmm4_overview_candles.parquet
    Data/chart/5m_hmm4_overview_segments.parquet
    Data/chart/5m_hmm4_overview_transitions.parquet
    Data/chart/models.json -- appended with the overview model entry

Run: `python scripts/build_overview_chart_data.py` (from the project root).
"""

import json
import os
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "Data", "chart")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_transitions(segments: pd.DataFrame) -> pd.DataFrame:
    if len(segments) < 2:
        return pd.DataFrame(columns=["timestamp", "from_state", "from_regime", "to_state", "to_regime"])
    rows = []
    for i in range(1, len(segments)):
        prev, cur = segments.iloc[i - 1], segments.iloc[i]
        rows.append({
            "timestamp": cur["start_timestamp"], "from_state": int(prev["state"]), "from_regime": prev["regime"],
            "to_state": int(cur["state"]), "to_regime": cur["regime"],
        })
    return pd.DataFrame(rows)


def main():
    log("Loading existing 5m-HMM4 decode (hmm4_candles.parquet) -- no recomputation...")
    candles = pd.read_parquet(os.path.join(OUT_DIR, "hmm4_candles.parquet"))
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    candles = candles.set_index("timestamp")

    models_path = os.path.join(OUT_DIR, "models.json")
    meta = json.load(open(models_path))
    base = next(m for m in meta["models"] if m["model_id"] == "5m_hmm4")
    regime_labels = base["regime_labels"]

    log("Resampling to 1 candle per day (open/high/low/close/volume)...")
    g = candles.resample("1D")
    daily = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    })

    log("Computing each day's dominant (most-occupied) regime from its own 5m bars...")

    def day_mode_state(s: pd.Series):
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return s.mode().iloc[0]

    daily_state = g["state"].apply(day_mode_state)
    daily["state"] = daily_state
    daily["regime"] = daily["state"].map(lambda s: regime_labels.get(str(int(s))) if pd.notna(s) else None)
    daily = daily.dropna(subset=["open"])  # drop any day with literally no OHLC (shouldn't happen, contiguous data)
    daily = daily.reset_index().rename(columns={"index": "timestamp"})

    n_days = len(daily)
    n_matched = int(daily["state"].notna().sum())
    log(f"  {n_days:,} days, {n_matched:,} with a decoded dominant regime "
        f"({n_days - n_matched} with no valid HMM bars that day)")

    # segments/transitions at daily granularity
    state = daily["state"]
    is_valid = state.notna()
    changed = np.empty(len(daily), dtype=bool)
    changed[0] = True
    changed[1:] = (state.values[1:] != state.values[:-1]) | (is_valid.values[1:] != is_valid.values[:-1])
    both_nan = (~is_valid.values[1:]) & (~is_valid.values[:-1])
    changed[1:] = changed[1:] & ~both_nan
    run_id = np.cumsum(changed)
    tmp = daily.assign(run_id=run_id)
    valid_runs = tmp[tmp["state"].notna()]
    segments = valid_runs.groupby("run_id").agg(
        state=("state", "first"), regime=("regime", "first"),
        start_timestamp=("timestamp", "first"), end_timestamp=("timestamp", "last"),
        number_of_bars=("timestamp", "size"),
    ).reset_index(drop=True)
    segments["duration_minutes"] = segments["number_of_bars"] * 1440.0
    segments.insert(0, "segment_id", np.arange(len(segments)))
    transitions = build_transitions(segments)
    log(f"  {len(segments):,} daily-regime segments, {len(transitions):,} transitions")

    daily["state"] = daily["state"].astype(float)
    daily["confidence"] = np.nan  # not meaningful at daily granularity (day is a mode over many bars, not one decode)
    daily["state_duration"] = np.nan
    daily.to_parquet(os.path.join(OUT_DIR, "5m_hmm4_overview_candles.parquet"), index=False)
    segments.to_parquet(os.path.join(OUT_DIR, "5m_hmm4_overview_segments.parquet"), index=False)
    transitions.to_parquet(os.path.join(OUT_DIR, "5m_hmm4_overview_transitions.parquet"), index=False)

    new_entry = {
        "model_id": "5m_hmm4_overview", "timeframe": "1D", "file_prefix": "5m_hmm4_overview", "n_states": 4,
        "regime_labels": regime_labels,
        "n_candles": int(n_days), "n_matched": n_matched, "n_segments": int(len(segments)),
        "n_transitions": int(len(transitions)),
        "date_range": [str(daily["timestamp"].min()), str(daily["timestamp"].max())],
        "train_end": base["train_end"],
        "note": ("Full-history daily overview: 1 candle/day, colored by that day's MOST-OCCUPIED 5m-HMM4 "
                 "regime (not a separate model fit). For seeing the whole 2017-2026 history at a glance -- "
                 "zoom into the regular 5m/15m models for bar-level detail."),
    }
    meta["models"] = [m for m in meta["models"] if m["model_id"] != "5m_hmm4_overview"] + [new_entry]
    json.dump(meta, open(models_path, "w"), indent=2, default=str)
    log(f"Updated {models_path} -- models now: {[m['model_id'] for m in meta['models']]}")


if __name__ == "__main__":
    main()
