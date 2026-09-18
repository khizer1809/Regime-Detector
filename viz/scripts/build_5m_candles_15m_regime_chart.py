"""
build_5m_candles_15m_regime_chart.py -- chart data source with 5m-candle
detail but regime coloring/state driven by the PRODUCTION 15m-HMM4 model
(Data/production_model_15m_hmm4.pkl), not the 5m model and not the fast-
screening's research artifact.

Mechanism: decode the full history once with the production 15m-HMM4 model
(genuine 15m candles, its own 18 features) -> every 15m decoded bar's
state/regime/confidence is broadcast to the 3 raw 5m candles inside it
(floor each 5m timestamp to its 15-minute bucket to find which 15m bar it
belongs to). A 5m candle whose 15m bucket wasn't valid for the 15m model
(gap/warmup) gets a null state, never fabricated. Segments/transitions are
then built directly on this broadcasted 5m-level state array, so a
"regime segment" always starts/ends exactly at a real 5m candle boundary
even though the underlying regime decision was made at 15m granularity.

state_duration is expressed in 5m-candle-EQUIVALENT units (the 15m-bar
duration x3), not raw 15m-bar count, so it reads consistently against the
5m candles actually shown on this chart (documented here, not left
ambiguous).

Produces:
    Data/chart/5m_candles_15m_regime_candles.parquet
    Data/chart/5m_candles_15m_regime_segments.parquet
    Data/chart/5m_candles_15m_regime_transitions.parquet
    Data/chart/models.json -- appended with this model entry

Run: `python scripts/build_5m_candles_15m_regime_chart.py` (from the project root).
"""

import json
import os
import pickle
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
ARTIFACT_PATH = os.path.join(_ROOT, "Data", "production_model_15m_hmm4.pkl")
MODEL_ID = "5m_candles_15m_regime"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_segments(candles: pd.DataFrame) -> pd.DataFrame:
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
    segs["duration_minutes"] = (segs["end_timestamp"] - segs["start_timestamp"]).dt.total_seconds() / 60 + 5
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
    log("Loading production 15m-HMM4 model and decoding full history...")
    with open(ARTIFACT_PATH, "rb") as f:
        artifact = pickle.load(f)

    raw5m = v22.load_raw_5m()
    agg15 = v22.build_agg_candles(raw5m, "15m")
    feat_df = v22.build_tf_features(agg15, "15m")
    assert list(feat_df.columns) == artifact["feature_columns"]
    max_lookback = max(v22.EMA_SLOW_BARS, max(v22.RETURN_HORIZON_BARS["15m"].values()))
    valid_flag = v22.build_validity_mask_native(agg15, feat_df.index, max_lookback)
    valid_mask = valid_flag & feat_df.notna().all(axis=1)
    valid_index_15m = feat_df.index[valid_mask]
    bar_interval_15m = pd.Timedelta(minutes=15)
    lengths = v22.segment_lengths_native(valid_index_15m, bar_interval_15m)

    scaled = artifact["scaler"].transform(feat_df.loc[valid_index_15m].values)
    states = artifact["model"].predict(scaled, lengths=lengths)
    posteriors = artifact["model"].predict_proba(scaled, lengths=lengths)
    confidence = persistence_common.compute_confidence(posteriors, states)
    duration_15m_bars = persistence_common.compute_running_duration(states, lengths)

    means_df = pd.DataFrame(artifact["model"].means_, columns=artifact["feature_columns"])
    regime_labels = [v22.characterize_state_tf(means_df.iloc[s], "15m")[0] for s in range(artifact["model"].n_components)]
    log(f"  15m state -> regime mapping: {dict(enumerate(regime_labels))}")

    decode_15m = pd.DataFrame({
        "timestamp_15m": valid_index_15m, "state": states,
        "regime": [regime_labels[s] for s in states], "confidence": confidence,
        "state_duration": duration_15m_bars * 3,  # 5m-candle-equivalent units, documented above
    })
    log(f"  {len(decode_15m):,} valid 15m bars decoded ({len(lengths)} contiguous segments)")

    log("Broadcasting each 15m regime onto its 3 underlying 5m candles...")
    raw = raw5m.reset_index().rename(columns={"timestamp": "timestamp"})[["timestamp", "open", "high", "low", "close", "volume"]]
    raw["timestamp_15m"] = raw["timestamp"].dt.floor("15min")
    merged = raw.merge(decode_15m, on="timestamp_15m", how="left").drop(columns=["timestamp_15m"])

    n_ohlc = len(raw)
    n_matched = int(merged["state"].notna().sum())
    log(f"  5m OHLC rows: {n_ohlc:,}  matched to a valid 15m regime: {n_matched:,} "
        f"(unmatched: {n_ohlc - n_matched:,}, expected -- 15m warm-up/gap bars)")

    segments = build_segments(merged)
    transitions = build_transitions(segments)
    log(f"  {len(segments):,} contiguous regime segments (5m-candle-aligned boundaries), "
        f"{len(transitions):,} transitions")

    merged.to_parquet(os.path.join(OUT_DIR, f"{MODEL_ID}_candles.parquet"), index=False)
    segments.to_parquet(os.path.join(OUT_DIR, f"{MODEL_ID}_segments.parquet"), index=False)
    transitions.to_parquet(os.path.join(OUT_DIR, f"{MODEL_ID}_transitions.parquet"), index=False)

    new_entry = {
        "model_id": MODEL_ID, "timeframe": "5m", "file_prefix": MODEL_ID, "n_states": artifact["model"].n_components,
        "regime_labels": {int(s): r for s, r in enumerate(regime_labels)},
        "n_candles": int(n_ohlc), "n_matched": n_matched, "n_segments": int(len(segments)),
        "n_transitions": int(len(transitions)),
        "date_range": [str(raw["timestamp"].min()), str(raw["timestamp"].max())],
        "train_end": artifact["train_end"],
        "note": ("5m candles (full price detail) colored by the PRODUCTION 15m-HMM4 model's regime -- "
                 "each 15m decode broadcast to its 3 underlying 5m bars. state_duration is in 5m-candle-"
                 "equivalent units (15m-bar duration x3) for display consistency with the 5m candles shown, "
                 "not raw 15m-bar count. Not the same artifact as the 'production' 5m-HMM4 model or the "
                 "fast-screening's 15m_hmm4 research entry -- this is Data/production_model_15m_hmm4.pkl."),
    }
    models_path = os.path.join(OUT_DIR, "models.json")
    meta = json.load(open(models_path))
    meta["models"] = [m for m in meta["models"] if m["model_id"] != MODEL_ID] + [new_entry]
    json.dump(meta, open(models_path, "w"), indent=2, default=str)
    log(f"Updated {models_path} -- models now: {[m['model_id'] for m in meta['models']]}")


if __name__ == "__main__":
    main()
