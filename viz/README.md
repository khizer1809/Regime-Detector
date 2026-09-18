# BTCUSDT HMM Regime Viewer

A local, TradingView-style candlestick chart that visualizes which HMM
regime/state the production model assigns to every BTCUSDT 5-minute candle
-- built as a research tool for answering: **"where did the HMM think the
market regime changed, and what did BTC price do around that change?"**

This is a **visualization/research tool only**. No trading execution, order
placement, or exchange connectivity of any kind.

## What it reuses vs. what's new

Reuses the existing production pipeline verbatim, unmodified:
- `src/infer_regime.py` -- `load_artifact()` / `characterize_state()` (the
  project's real state -> regime-name mapping, TREND_Z_THRESHOLD=0.3 /
  VOL_Z_THRESHOLD=0.3, same as production).
- `src/persistence_dataset.py` -- `decode_full_history()` (one-shot decode
  of the production HMM, `scaler.transform()`-only, never refit).
- `src/persistence_common.py` -- confidence/duration computation.
- `Data/production_model_hmm4.pkl` and `Data/production_model_hmm6.pkl`.

New, built for this tool only:
- `scripts/build_regime_chart_data.py` -- preprocessing (see below).
- `viz/backend/` -- FastAPI server, read-only over the preprocessed data.
- `viz/frontend/` -- React + TypeScript + [lightweight-charts](https://github.com/tradingview/lightweight-charts)
  (TradingView's own open-source charting library).

Nothing here retrains, modifies, or touches the HMM, the raw OHLCV source,
feature definitions, scaling, or any prior research result.

## 1. Install

**Backend (Python):**
```bash
pip install fastapi "uvicorn[standard]" pyarrow
```
(pandas/numpy/scikit-learn/hmmlearn are already required by the existing project.)

**Frontend (Node):**
```bash
cd viz/frontend
npm install
```

## 2. Data configuration

Candlesticks come from the raw 5m OHLCV file already used elsewhere in this
project (`BTCUSDT_5M.csv`). If your copy lives somewhere else, update
`RAW_OHLCV_PATH` at the top of `scripts/build_regime_chart_data.py`.

## 3. Model configuration

By default, both HMM-4 and HMM-6 are built (`CANDIDATE_N_STATES = [4, 6]` in
the preprocessing script) -- whichever of `Data/production_model_hmm4.pkl` /
`hmm6.pkl` exist and whose feature schema matches the current
`Data/features_out_masked.csv` will be included. If only one is available,
the model dropdown in the UI shows only that one.

## 4. Preprocessing (run once, and again any time the production model is retrained)

```bash
python scripts/build_regime_chart_data.py
```

This decodes full history with each available production model, aligns it
against the raw OHLCV by **explicit timestamp merge** (never by row
position), computes contiguous state segments and state-change transitions,
and writes:

```
Data/chart/
    hmm4_candles.parquet       hmm6_candles.parquet
    hmm4_segments.parquet      hmm6_segments.parquet
    hmm4_transitions.parquet   hmm6_transitions.parquet
    models.json
```

The script prints an explicit alignment report (OHLC rows / HMM rows /
matched / unmatched) -- check this after any rerun. On the full history as
of this build: 933,924 raw candles, 923,766 HMM-decoded rows, 0 unmatched
HMM timestamps (the ~10K unmatched OHLC rows are gap/warm-up bars the HMM
pipeline already excludes -- expected, not an error).

## 5. Start the backend

```bash
cd viz/backend
uvicorn main:app --reload --port 8000
```
Health check: http://127.0.0.1:8000/api/health

## 6. Start the frontend

```bash
cd viz/frontend
npm run dev
```
Open http://localhost:5173

**Or, once both are installed and preprocessing has run once**, start both
together:
```bash
python viz/run_app.py
```

## 7. Using the app

- **Model dropdown** switches between HMM-4 / HMM-6 -- reloads state
  assignments, regime legend, transitions, and stats for that model.
- **Prev / Next** shift the loaded 30-day window; **date picker + Go** jumps
  to a specific time (snaps to the nearest available candle if the exact
  timestamp isn't there, and tells you so).
- **Fit** re-fits the chart to the loaded data; **Latest** jumps to the most
  recent available window.
- Candle color = price direction (green/red) only. Background color bands =
  HMM regime. Small orange dots = state-change events. Hover any candle for
  its OHLCV + HMM state/regime/confidence/duration in the right panel.
- The right panel also shows the regime legend, live-computed stats for the
  currently loaded window (bars, transitions, current regime/duration,
  longest segment, regime distribution), and a clickable list of recent
  state changes that jumps the chart to that moment.

## Documented scope decisions

- **Windowed loading, not infinite virtual scroll.** The dataset spans
  ~9 years (933,924 5-minute bars). Rather than trying to virtualize the
  entire history into the chart library's live viewport, the app loads a
  30-day window at a time (smooth pan/zoom within it) and uses
  Prev/Next/date-jump for broader navigation. This was a deliberate
  simplicity/robustness tradeoff, not an oversight -- flagged here per the
  project's own "document reasonable technical choices" instruction.
- **Marker text.** State-change markers are drawn as small dots (not
  always-on text labels) -- with hundreds to low-thousands of transitions
  in a typical 30-day window, always-rendered text became unreadable
  clutter in testing. Full FROM/TO detail is available via the Recent State
  Changes panel and by hovering the surrounding candles.
- **HMM-6 label collisions.** HMM-6 (more states than HMM-4) produces
  several numerically-distinct states that map to the *same* regime name
  under the project's existing `characterize_state()` labeling (e.g. three
  different "Ranging / Low-Vol" states). This is real, existing behavior of
  the project's labeling function, not something this tool alters --
  transitions are computed from the raw state index (`prev_state !=
  cur_state`), so a same-label transition between two different underlying
  HMM-6 states is still correctly marked as a transition, even though the
  displayed labels on both sides read the same.

## Validation performed

- Data alignment: explicit timestamp merge, 0 unmatched HMM timestamps
  (printed by the preprocessing script every run).
- Backend: all endpoints (`/api/health`, `/api/models`, `/api/candles`,
  `/api/segments`, `/api/transitions`, `/api/state-stats`,
  `/api/nearest-candle`) smoke-tested against real data via curl.
- Frontend: driven with a headless-browser (Playwright) test pass --
  candles render, regime bands render, crosshair hover updates the state
  panel with real OHLC/state data, mouse-wheel zoom and drag-pan both
  work, model switching (HMM-4 <-> HMM-6) reloads correctly, clicking a
  transition doesn't crash, zero browser console errors and zero page
  errors observed.
