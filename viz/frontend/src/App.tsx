import { useCallback, useEffect, useRef, useState } from "react";
import TradingChart from "./components/TradingChart";
import ControlBar from "./components/ControlBar";
import RegimeLegend from "./components/RegimeLegend";
import StatePanel from "./components/StatePanel";
import StatsPanel from "./components/StatsPanel";
import TransitionPanel from "./components/TransitionPanel";
import { fetchModels, fetchCandles, fetchSegments, fetchTransitions, fetchStateStats, fetchNearestCandle } from "./services/api";
import type { Candle, Segment, Transition, ModelInfo, StateStats } from "./types";
import "./App.css";

const WINDOW_DAYS = 30;
const DAY_SECONDS = 86400;

function isOverviewModel(m: ModelInfo): boolean {
  return m.model_id.endsWith("_overview");
}

// Regular models default to the most recent 30-day window; an "overview"
// model (one daily candle per day, full 2017-2026 history) instead defaults
// to its ENTIRE date range at once -- it's only ~3,200 candles total, so
// there's no performance reason to window it, and a 30-day slice of daily
// candles would be a nearly-empty, useless view.
function defaultWindowFor(m: ModelInfo): { start: Date; end: Date } {
  const end = new Date(m.date_range[1]);
  if (isOverviewModel(m)) {
    return { start: new Date(m.date_range[0]), end };
  }
  return { start: new Date(end.getTime() - WINDOW_DAYS * DAY_SECONDS * 1000), end };
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [windowStart, setWindowStart] = useState<Date | null>(null);
  const [windowEnd, setWindowEnd] = useState<Date | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [transitions, setTransitions] = useState<Transition[]>([]);
  const [stats, setStats] = useState<StateStats | null>(null);
  const [hovered, setHovered] = useState<Candle | null>(null);
  const [dateInput, setDateInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [snapNote, setSnapNote] = useState<string | null>(null);
  const [fitSignal, setFitSignal] = useState(0);

  const jumpFnRef = useRef<((time: number) => void) | null>(null);

  // initial load: models, then default to the most recent window of the first model
  useEffect(() => {
    (async () => {
      try {
        const { models: m } = await fetchModels();
        if (m.length === 0) throw new Error("No HMM models available (run scripts/build_regime_chart_data.py first)");
        setModels(m);
        const first = m[0];
        setSelectedModel(first.model_id);
        const { start, end } = defaultWindowFor(first);
        setWindowStart(start);
        setWindowEnd(end);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const loadWindow = useCallback(async (model: string, start: Date, end: Date) => {
    setLoading(true);
    setError(null);
    try {
      const startIso = start.toISOString();
      const endIso = end.toISOString();
      const [candlesRes, segmentsRes, transitionsRes, statsRes] = await Promise.all([
        fetchCandles(model, startIso, endIso, 50000),
        fetchSegments(model, startIso, endIso),
        fetchTransitions(model, startIso, endIso, 500),
        fetchStateStats(model, startIso, endIso),
      ]);
      setCandles(candlesRes.candles);
      setSegments(segmentsRes.segments);
      setTransitions(transitionsRes.transitions);
      setStats(statsRes);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedModel !== null && windowStart && windowEnd) {
      loadWindow(selectedModel, windowStart, windowEnd);
    }
  }, [selectedModel, windowStart, windowEnd, loadWindow]);

  function handleModelChange(id: string) {
    setSnapNote(null);
    setSelectedModel(id);
    const info = models.find((m) => m.model_id === id);
    if (info) {
      const { start, end } = defaultWindowFor(info);
      setWindowStart(start);
      setWindowEnd(end);
    }
  }

  function shiftWindow(days: number) {
    if (!windowStart || !windowEnd) return;
    setSnapNote(null);
    setWindowStart(new Date(windowStart.getTime() + days * DAY_SECONDS * 1000));
    setWindowEnd(new Date(windowEnd.getTime() + days * DAY_SECONDS * 1000));
  }

  async function handleJump() {
    if (!dateInput || selectedModel === null) return;
    setSnapNote(null);
    try {
      const nearest = await fetchNearestCandle(selectedModel, new Date(dateInput).toISOString());
      if (!nearest.exact_match) {
        setSnapNote(`Requested time not available -- snapped to nearest candle: ${nearest.matched_timestamp}`);
      }
      const center = new Date(nearest.time * 1000);
      const half = (WINDOW_DAYS * DAY_SECONDS * 1000) / 2;
      setWindowStart(new Date(center.getTime() - half));
      setWindowEnd(new Date(center.getTime() + half));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function handleLatest() {
    const model = models.find((m) => m.model_id === selectedModel);
    if (!model) return;
    setSnapNote(null);
    const { start, end } = defaultWindowFor(model);
    setWindowStart(start);
    setWindowEnd(end);
  }

  function handleFit() {
    setFitSignal((s) => s + 1);
  }

  function handleTransitionSelect(time: number) {
    jumpFnRef.current?.(time);
  }

  const currentModelInfo = models.find((m) => m.model_id === selectedModel);
  const lastCandle = candles.length > 0 ? candles[candles.length - 1] : null;
  const windowLabel =
    windowStart && windowEnd
      ? `${windowStart.toISOString().slice(0, 10)} → ${windowEnd.toISOString().slice(0, 10)}`
      : "";

  return (
    <div className="app-root">
      {models.length > 0 && selectedModel !== null && (
        <ControlBar
          models={models}
          selectedModel={selectedModel}
          onModelChange={handleModelChange}
          dateInput={dateInput}
          onDateInputChange={setDateInput}
          onJump={handleJump}
          onPrevWindow={() => shiftWindow(-WINDOW_DAYS)}
          onNextWindow={() => shiftWindow(WINDOW_DAYS)}
          onFit={handleFit}
          onLatest={handleLatest}
          windowLabel={windowLabel}
          timeframe={currentModelInfo?.timeframe ?? "5m"}
        />
      )}

      {error && <div className="error-banner">⚠ {error}</div>}
      {snapNote && <div className="info-banner">{snapNote}</div>}
      {loading && <div className="loading-banner">Loading...</div>}

      <div className="main-layout">
        <div className="chart-area">
          <TradingChart
            candles={candles}
            segments={segments}
            transitions={transitions}
            onCrosshairMove={setHovered}
            onJumpRequest={(fn) => (jumpFnRef.current = fn)}
            fitSignal={fitSignal}
            logScale={currentModelInfo ? isOverviewModel(currentModelInfo) : false}
          />
        </div>
        <div className="side-panel">
          <StatePanel hovered={hovered} fallback={lastCandle} />
          {currentModelInfo && <RegimeLegend regimeLabels={currentModelInfo.regime_labels} />}
          <StatsPanel stats={stats} />
          <TransitionPanel transitions={transitions} onSelect={handleTransitionSelect} />
        </div>
      </div>
    </div>
  );
}
