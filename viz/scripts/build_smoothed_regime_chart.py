"""
build_smoothed_regime_chart.py -- adds a "smoothed" viewing option to the
regime viz app: post-hoc minimum-duration merging of the ALREADY-DECODED
HMM state sequence, so isolated short blips stop registering as their own
regime change on the chart.

This is a DISPLAY/interpretation aid only:
  - Does NOT retrain, refit, or touch the HMM/scaler/features in any way.
  - Does NOT change the underlying `production_model_hmm4.pkl` decode; it
    post-processes the already-built Data/chart/hmm4_candles.parquet.
  - Is NOT causal/online-safe as implemented (a segment is only known to be
    "too short" once it has ENDED, which requires seeing bars after it) --
    fine for a historical chart, but if this logic is ever wanted for LIVE
    inference, it needs a different, forward-only formulation (e.g. "require
    K consecutive bars of the new state before displaying the regime
    change"), which is NOT what this script does.

ALGORITHM (single left-to-right pass over the existing contiguous segments,
merge-with-a-stack -- see inline comments for why one pass is sufficient):
    For each segment in chronological order:
      - if the output stack is non-empty AND (this segment's duration is
        below MIN_MINUTES, OR its state equals the top-of-stack's state),
        merge it into the top-of-stack segment (extend end time / bar count,
        keep the TOP's state -- i.e. a short blip is swallowed by whatever
        regime was already running).
      - otherwise, push it as a new segment.
    Merging never needs to look further back than the current top-of-stack:
    by construction the existing segment list already has no two adjacent
    segments sharing a state, so extending the top-of-stack can only ever
    newly match the NEXT segment in the loop, which the loop's own re-check
    of the (now-updated) top-of-stack on the following iteration handles --
    no multi-pass iteration required.

MIN_MINUTES=30 default: chosen as a round, clearly-stated threshold roughly
at the short end of this project's own segment-duration stats (5m-HMM4
median segment = 30min already; this only removes segments SHORTER than
that, not the median itself). Configurable via the constant below --
"correct" threshold is a matter of how much noise-suppression is wanted,
not a statistically derived number, and is disclosed as such.

Produces (does not touch or overwrite the existing raw hmm4_/hmm6_/15m_hmm4_
files):
    Data/chart/<tf>_hmm<N>_smoothed_candles.parquet
    Data/chart/<tf>_hmm<N>_smoothed_segments.parquet
    Data/chart/<tf>_hmm<N>_smoothed_transitions.parquet
    Data/chart/models.json -- appended with the smoothed variant(s)

Run: `python scripts/build_smoothed_regime_chart.py` (from the project root).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

OUT_DIR = os.path.join(_ROOT, "Data", "chart")
MIN_MINUTES = 30  # segments shorter than this get absorbed into the preceding regime


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_raw_segments(candles: pd.DataFrame) -> pd.DataFrame:
    """Same logic as build_regime_chart_data.py's build_segments -- contiguous
    runs of the same non-null state, never merging across a genuine gap
    (state becomes null)."""
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
    return segs


def smooth_segments(segments: pd.DataFrame, bar_minutes: float, min_minutes: float) -> pd.DataFrame:
    min_bars = min_minutes / bar_minutes
    out = []
    for _, seg in segments.iterrows():
        seg = seg.to_dict()
        if out and (seg["number_of_bars"] < min_bars or out[-1]["state"] == seg["state"]):
            out[-1]["end_timestamp"] = seg["end_timestamp"]
            out[-1]["number_of_bars"] += seg["number_of_bars"]
        else:
            out.append(seg)
    result = pd.DataFrame(out)
    result["duration_minutes"] = result["number_of_bars"] * bar_minutes
    result.insert(0, "segment_id", np.arange(len(result)))
    return result[["segment_id", "state", "regime", "start_timestamp", "end_timestamp",
                    "number_of_bars", "duration_minutes"]]


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


def apply_smoothed_state_to_candles(candles: pd.DataFrame, smoothed_segments: pd.DataFrame) -> pd.DataFrame:
    """Re-labels each valid (non-null-state) candle according to which
    SMOOTHED segment it falls in. Vectorized: the smoothed segments cover
    exactly the same valid rows as the original decode, in the same
    chronological order, just re-grouped -- so a single np.repeat expansion
    (segment state/regime, segment bar count) reproduces the full per-bar
    array in one pass, without any per-segment mask over the whole frame
    (which would be O(n_segments x n_candles), far too slow at this scale).
    Bars with no HMM decode at all (gap/warmup) stay null, never fabricated."""
    out = candles.copy()
    valid_mask = candles["state"].notna()
    n_valid = int(valid_mask.sum())
    assert int(smoothed_segments["number_of_bars"].sum()) == n_valid, (
        "smoothed segments must cover exactly the original valid bar count -- mismatch indicates a bug"
    )
    state_expanded = np.repeat(smoothed_segments["state"].values, smoothed_segments["number_of_bars"].values)
    regime_expanded = np.repeat(smoothed_segments["regime"].values, smoothed_segments["number_of_bars"].values)
    out["state"] = np.nan
    out["regime"] = None
    out.loc[valid_mask, "state"] = state_expanded
    out.loc[valid_mask, "regime"] = regime_expanded
    return out


def process_one(tag: str, bar_minutes: float):
    log(f"=== {tag} (bar_minutes={bar_minutes}) ===")
    candles = pd.read_parquet(os.path.join(OUT_DIR, f"{tag}_candles.parquet"))
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)

    raw_segments = build_raw_segments(candles)
    log(f"  raw: {len(raw_segments):,} segments")
    smoothed_segments = smooth_segments(raw_segments, bar_minutes, MIN_MINUTES)
    log(f"  smoothed (min={MIN_MINUTES}min): {len(smoothed_segments):,} segments "
        f"({len(raw_segments) - len(smoothed_segments):,} short segments absorbed, "
        f"{(1 - len(smoothed_segments)/len(raw_segments))*100:.1f}% reduction)")

    smoothed_candles = apply_smoothed_state_to_candles(candles, smoothed_segments)
    transitions = build_transitions(smoothed_segments)

    out_tag = f"{tag}_smoothed"
    smoothed_candles.to_parquet(os.path.join(OUT_DIR, f"{out_tag}_candles.parquet"), index=False)
    smoothed_segments.to_parquet(os.path.join(OUT_DIR, f"{out_tag}_segments.parquet"), index=False)
    transitions.to_parquet(os.path.join(OUT_DIR, f"{out_tag}_transitions.parquet"), index=False)

    n_ohlc = len(candles)
    n_matched = int(smoothed_candles["state"].notna().sum())
    return {
        "n_candles": int(n_ohlc), "n_matched": n_matched, "n_segments": int(len(smoothed_segments)),
        "n_transitions": int(len(transitions)),
        "raw_n_segments": int(len(raw_segments)),
        "pct_segments_removed": round((1 - len(smoothed_segments) / len(raw_segments)) * 100, 1),
    }


def main():
    models_path = os.path.join(OUT_DIR, "models.json")
    meta = json.load(open(models_path))

    targets = [
        {"base_model_id": "5m_hmm4", "file_prefix": "hmm4", "timeframe": "5m", "n_states": 4, "bar_minutes": 5.0},
        {"base_model_id": "15m_hmm4", "file_prefix": "15m_hmm4", "timeframe": "15m", "n_states": 4, "bar_minutes": 15.0},
    ]

    new_entries = []
    for t in targets:
        base = next(m for m in meta["models"] if m["model_id"] == t["base_model_id"])
        stats = process_one(t["file_prefix"], t["bar_minutes"])
        new_entries.append({
            "model_id": f"{t['base_model_id']}_smoothed", "timeframe": t["timeframe"],
            "file_prefix": f"{t['file_prefix']}_smoothed", "n_states": t["n_states"],
            "regime_labels": base["regime_labels"],
            "n_candles": stats["n_candles"], "n_matched": stats["n_matched"],
            "n_segments": stats["n_segments"], "n_transitions": stats["n_transitions"],
            "date_range": base["date_range"], "train_end": base["train_end"],
            "note": (f"Post-hoc minimum-duration smoothing (segments <{MIN_MINUTES}min absorbed into the "
                     f"preceding regime) of {t['base_model_id']}'s own decode -- NOT a different HMM fit, "
                     f"NOT causal/live-safe. {stats['raw_n_segments']:,} raw segments -> {stats['n_segments']:,} "
                     f"smoothed ({stats['pct_segments_removed']}% removed)."),
        })

    meta["models"] = [m for m in meta["models"] if m["model_id"] not in {e["model_id"] for e in new_entries}] + new_entries
    json.dump(meta, open(models_path, "w"), indent=2, default=str)
    log(f"Updated {models_path} -- models now: {[m['model_id'] for m in meta['models']]}")


if __name__ == "__main__":
    main()
