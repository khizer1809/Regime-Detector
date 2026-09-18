"""
FastAPI backend for the HMM regime visualization app.

Serves the pre-processed, pre-segmented chart data built by
scripts/build_regime_chart_data.py (Data/chart/hmm<N>_*.parquet). Loads
each available model's data into memory once at startup (a few hundred MB
for the full multi-year history is well within reach; this is a local
research tool, not a multi-tenant service) and slices by date range per
request -- no HMM inference and no data reprocessing happens on any request.

Does not modify, retrain, or re-decode anything. Read-only over the
pre-built Parquet files.

Run: `uvicorn main:app --reload --port 8000` (from viz/backend/).
"""

import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_BACKEND_DIR))
CHART_DATA_DIR = os.path.join(_ROOT, "Data", "chart")

app = FastAPI(title="BTCUSDT HMM Regime Viewer", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_MODELS: dict[str, dict] = {}


def _load_all_models():
    import json
    models_path = os.path.join(CHART_DATA_DIR, "models.json")
    if not os.path.exists(models_path):
        raise RuntimeError(
            f"{models_path} not found -- run `python scripts/build_regime_chart_data.py` from the project root first.")
    meta = json.load(open(models_path))
    for m in meta["models"]:
        # model_id is a string key (e.g. "5m_hmm4", "15m_hmm4") so models at
        # different base timeframes can coexist -- older models.json entries
        # (pre-multi-timeframe) may lack it, so fall back to "5m_hmm<N>".
        model_id = m.get("model_id") or f"5m_hmm{m['n_states']}"
        m.setdefault("model_id", model_id)
        m.setdefault("timeframe", "5m")
        prefix = m.get("file_prefix") or f"hmm{m['n_states']}"
        # NOTE: Parquet round-trips pandas timestamps at MICROSECOND precision
        # (datetime64[us]), not the nanosecond precision astype("int64")//10**9
        # below assumes -- normalizing to datetime64[ns, UTC] here, once, keeps
        # every downstream unix-seconds conversion correct regardless of the
        # source file's stored unit.
        candles = pd.read_parquet(os.path.join(CHART_DATA_DIR, f"{prefix}_candles.parquet"))
        candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        candles = candles.sort_values("timestamp").reset_index(drop=True)
        segments = pd.read_parquet(os.path.join(CHART_DATA_DIR, f"{prefix}_segments.parquet"))
        segments["start_timestamp"] = pd.to_datetime(segments["start_timestamp"], utc=True).astype("datetime64[ns, UTC]")
        segments["end_timestamp"] = pd.to_datetime(segments["end_timestamp"], utc=True).astype("datetime64[ns, UTC]")
        transitions = pd.read_parquet(os.path.join(CHART_DATA_DIR, f"{prefix}_transitions.parquet"))
        if len(transitions):
            transitions["timestamp"] = pd.to_datetime(transitions["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        _MODELS[model_id] = {"meta": m, "candles": candles, "segments": segments, "transitions": transitions}
    print(f"Loaded models: {list(_MODELS.keys())}")


@app.on_event("startup")
def startup():
    _load_all_models()


def _get_model(model_id: str) -> dict:
    if model_id not in _MODELS:
        raise HTTPException(404, f"Model '{model_id}' not available. Available: {list(_MODELS.keys())}")
    return _MODELS[model_id]


def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": list(_MODELS.keys())}


@app.get("/api/models")
def list_models():
    return {"models": [m["meta"] for m in _MODELS.values()]}


@app.get("/api/regimes")
def regimes(model: str = Query(..., description="Model id, e.g. 5m_hmm4, 5m_hmm6, 15m_hmm4")):
    m = _get_model(model)
    return {"model": model, "regime_labels": m["meta"]["regime_labels"]}


@app.get("/api/candles")
def candles(model: str = Query(...), start: Optional[str] = None, end: Optional[str] = None,
            limit: int = Query(20000, le=100000)):
    m = _get_model(model)
    df = m["candles"]
    start_ts, end_ts = _parse_ts(start), _parse_ts(end)
    if start_ts is not None:
        df = df[df["timestamp"] >= start_ts]
    if end_ts is not None:
        df = df[df["timestamp"] <= end_ts]
    truncated = len(df) > limit
    if start_ts is None and end_ts is None:
        # default view (Section 26): most recent window
        df = df.tail(limit)
    else:
        df = df.head(limit)
    out = df.copy()
    out["time"] = (out["timestamp"].astype("int64") // 10**9).astype(int)
    out["state"] = out["state"].astype("Int64")
    # NOTE: a plain `.where(pd.notnull(out), None)` on a float64 column is a
    # no-op -- pandas silently re-casts the assigned `None` back to NaN in a
    # float dtype Series, so NaN confidence/state_duration never actually
    # became JSON null. Never triggered before because every date range
    # tested so far happened to avoid gap/warmup rows; the full-history
    # overview model (entirely-NaN confidence/state_duration by design)
    # surfaced it immediately -- FastAPI's JSON encoder correctly rejects a
    # bare NaN (ValueError: Out of range float values are not JSON
    # compliant). Fixed by casting to object dtype first, which lets None
    # stick instead of being coerced back to NaN.
    sel = out[["time", "open", "high", "low", "close", "volume", "state", "regime", "confidence",
               "state_duration"]]
    records = sel.astype(object).where(pd.notnull(sel), None).to_dict(orient="records")
    return {"model": model, "count": len(records), "truncated": truncated, "candles": records}


@app.get("/api/segments")
def segments(model: str = Query(...), start: Optional[str] = None, end: Optional[str] = None,
              limit: int = Query(5000, le=20000)):
    m = _get_model(model)
    df = m["segments"]
    start_ts, end_ts = _parse_ts(start), _parse_ts(end)
    if start_ts is not None:
        df = df[df["end_timestamp"] >= start_ts]
    if end_ts is not None:
        df = df[df["start_timestamp"] <= end_ts]
    df = df.tail(limit)
    out = df.copy()
    out["start_time"] = (out["start_timestamp"].astype("int64") // 10**9).astype(int)
    out["end_time"] = (out["end_timestamp"].astype("int64") // 10**9).astype(int)
    records = out[["segment_id", "state", "regime", "start_time", "end_time", "number_of_bars",
                   "duration_minutes"]].to_dict(orient="records")
    return {"model": model, "count": len(records), "segments": records}


@app.get("/api/transitions")
def transitions(model: str = Query(...), start: Optional[str] = None, end: Optional[str] = None,
                  limit: int = Query(200, le=2000)):
    m = _get_model(model)
    df = m["transitions"]
    if len(df) == 0:
        return {"model": model, "count": 0, "transitions": []}
    start_ts, end_ts = _parse_ts(start), _parse_ts(end)
    if start_ts is not None:
        df = df[df["timestamp"] >= start_ts]
    if end_ts is not None:
        df = df[df["timestamp"] <= end_ts]
    df = df.tail(limit)
    out = df.copy()
    out["time"] = (out["timestamp"].astype("int64") // 10**9).astype(int)
    records = out[["time", "from_state", "from_regime", "to_state", "to_regime"]].to_dict(orient="records")
    return {"model": model, "count": len(records), "transitions": records}


@app.get("/api/state-stats")
def state_stats(model: str = Query(...), start: Optional[str] = None, end: Optional[str] = None):
    m = _get_model(model)
    candles_df = m["candles"]
    start_ts, end_ts = _parse_ts(start), _parse_ts(end)
    view = candles_df
    if start_ts is not None:
        view = view[view["timestamp"] >= start_ts]
    if end_ts is not None:
        view = view[view["timestamp"] <= end_ts]
    valid = view[view["state"].notna()]

    seg_df = m["segments"]
    seg_view = seg_df
    if start_ts is not None:
        seg_view = seg_view[seg_df["end_timestamp"] >= start_ts]
    if end_ts is not None:
        seg_view = seg_view[seg_df["start_timestamp"] <= end_ts]

    trans_df = m["transitions"]
    trans_view = trans_df
    if len(trans_df):
        if start_ts is not None:
            trans_view = trans_view[trans_df["timestamp"] >= start_ts]
        if end_ts is not None:
            trans_view = trans_view[trans_df["timestamp"] <= end_ts]

    regime_dist = {}
    if len(valid):
        counts = valid["regime"].value_counts(normalize=True)
        regime_dist = {k: round(float(v) * 100, 2) for k, v in counts.items()}

    longest = None
    if len(seg_view):
        row = seg_view.loc[seg_view["duration_minutes"].idxmax()]
        longest = {"regime": row["regime"], "duration_minutes": float(row["duration_minutes"])}

    current = None
    if len(valid):
        last = valid.iloc[-1]
        # state_duration is intentionally NaN for the daily overview model
        # (a day's "duration in bars" isn't a meaningful concept for a
        # dominant-regime-per-day aggregate) -- guard instead of int(NaN).
        duration_bars = None if pd.isna(last["state_duration"]) else int(last["state_duration"])
        current = {"state": int(last["state"]), "regime": last["regime"],
                   "duration_bars": duration_bars}

    return {
        "model": model, "bars": int(len(view)), "labeled_bars": int(len(valid)),
        "state_transitions": int(len(trans_view)), "current_state": current,
        "longest_segment": longest, "regime_distribution_pct": regime_dist,
    }


@app.get("/api/nearest-candle")
def nearest_candle(model: str = Query(...), timestamp: str = Query(...)):
    m = _get_model(model)
    target = _parse_ts(timestamp)
    df = m["candles"]
    idx = (df["timestamp"] - target).abs().idxmin()
    row = df.loc[idx]
    exact = row["timestamp"] == target
    return {"requested": str(target), "matched_timestamp": str(row["timestamp"]), "exact_match": bool(exact),
            "time": int(row["timestamp"].value // 10**9)}
