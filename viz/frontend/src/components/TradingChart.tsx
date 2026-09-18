import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  createSeriesMarkers,
  ColorType,
  CrosshairMode,
  PriceScaleMode,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type Time,
} from "lightweight-charts";
import type { Candle, Segment, Transition } from "../types";
import { regimeBandColor } from "../services/regimeColors";

interface Props {
  candles: Candle[];
  segments: Segment[];
  transitions: Transition[];
  onCrosshairMove?: (candle: Candle | null) => void;
  onJumpRequest?: (jump: (time: number) => void) => void;
  fitSignal?: number;
  logScale?: boolean;
}

// Candles = price direction (handled by the series' own up/down colors).
// Background bands = HMM regime (drawn as a semi-transparent DOM overlay on
// top of the chart canvas, pointer-events:none so pan/zoom/crosshair still
// reach the chart underneath -- lightweight-charts has no native continuous
// "background band across a time range" primitive, this is the standard
// pattern for it). Markers = state-change events only, never per-candle.
export default function TradingChart({ candles, segments, transitions, onCrosshairMove, onJumpRequest, fitSignal, logScale }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const markersApiRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const segmentsRef = useRef<Segment[]>([]);
  const candlesRef = useRef<Candle[]>([]);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: { background: { type: ColorType.Solid, color: "#0d0f14" }, textColor: "#c7cdd9", fontSize: 11 },
      grid: { vertLines: { color: "#181c24" }, horzLines: { color: "#181c24" } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#262b36" },
      timeScale: { borderColor: "#262b36", timeVisible: true, secondsVisible: false },
      autoSize: true,
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });
    candleSeries.priceScale().applyOptions({ scaleMargins: { top: 0.06, bottom: 0.24 } });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: "#385263",
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

    markersApiRef.current = createSeriesMarkers(candleSeries, []);

    chartRef.current = chart;
    seriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    const redraw = () => drawRegimeBands();
    chart.timeScale().subscribeVisibleLogicalRangeChange(redraw);
    const resizeObs = new ResizeObserver(redraw);
    if (containerRef.current) resizeObs.observe(containerRef.current);

    chart.subscribeCrosshairMove((param) => {
      if (!onCrosshairMove) return;
      if (!param.time) {
        onCrosshairMove(null);
        return;
      }
      const match = candlesRef.current.find((c) => c.time === (param.time as number));
      onCrosshairMove(match ?? null);
    });

    if (onJumpRequest) {
      onJumpRequest((time: number) => {
        chart.timeScale().scrollToPosition(0, false);
        const logicalRange = chart.timeScale().getVisibleLogicalRange();
        void logicalRange;
        candleSeries.priceScale();
        // center the view on the requested time
        const idx = candlesRef.current.findIndex((c) => c.time >= time);
        if (idx >= 0) {
          const from = Math.max(0, idx - 100);
          const to = Math.min(candlesRef.current.length - 1, idx + 100);
          chart.timeScale().setVisibleRange({
            from: candlesRef.current[from].time as Time,
            to: candlesRef.current[to].time as Time,
          });
        }
      });
    }

    return () => {
      resizeObs.disconnect();
      chart.remove();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    candlesRef.current = candles;
    if (!seriesRef.current || !volumeSeriesRef.current) return;
    seriesRef.current.setData(
      candles.map((c) => ({ time: c.time as Time, open: c.open, high: c.high, low: c.low, close: c.close }))
    );
    volumeSeriesRef.current.setData(
      candles.map((c) => ({
        time: c.time as Time,
        value: c.volume,
        color: c.close >= c.open ? "rgba(38,166,154,0.5)" : "rgba(239,83,80,0.5)",
      }))
    );
    chartRef.current?.timeScale().fitContent();
    requestAnimationFrame(() => drawRegimeBands());
  }, [candles]);

  useEffect(() => {
    if (!markersApiRef.current) return;
    const markers = transitions
      .slice()
      .sort((a, b) => a.time - b.time)
      // No text label per marker (Section 8 wants every transition marked,
      // but with 100s-1000s of transitions in a typical window, always-on
      // text turns into unreadable clutter -- shape marks WHERE a change
      // happened; the side Transition Panel + hover-tooltip give the FROM/TO
      // detail on demand, which stays legible at any zoom/density).
      .map((t) => ({
        time: t.time as Time,
        position: "aboveBar" as const,
        // Deliberately NOT a color used by any regime band (green/red/amber/
        // purple, see regimeColors.ts) -- these mark WHERE a transition
        // happened, regardless of which regime it's to/from, so the color
        // must never look like it's indicating a specific regime.
        color: "#ffffff",
        shape: "circle" as const,
        size: 0.6,
        text: "",
      }));
    markersApiRef.current.setMarkers(markers);
  }, [transitions]);

  useEffect(() => {
    segmentsRef.current = segments;
    requestAnimationFrame(() => drawRegimeBands());
  }, [segments]);

  useEffect(() => {
    if (fitSignal === undefined) return;
    chartRef.current?.timeScale().fitContent();
    requestAnimationFrame(() => drawRegimeBands());
  }, [fitSignal]);

  // Multi-year BTC history spans ~$4.6k -> ~$140k -- on a linear scale the
  // early years are squashed unreadably close to zero. Log scale (the
  // standard fix for a high-appreciation asset's full history) keeps every
  // era equally readable; only applied where the caller opts in (the
  // full-history overview), not the normal windowed/detail views.
  useEffect(() => {
    seriesRef.current?.priceScale().applyOptions({
      mode: logScale ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal,
    });
  }, [logScale]);

  function drawRegimeBands() {
    const chart = chartRef.current;
    const overlay = overlayRef.current;
    if (!chart || !overlay) return;
    const timeScale = chart.timeScale();
    const width = overlay.clientWidth;
    overlay.innerHTML = "";

    // Bands: draw every visible segment (even short ones -- the color
    // striping itself is informative at any zoom level and cheap to render).
    // Labels: only the widest, most legible ones -- skip anything too
    // narrow to read AND anything that would overlap the previously placed
    // label, so zooming out on a choppy period never produces unreadable
    // stacked text (Section 9's "labels must remain readable while zooming").
    const MIN_LABEL_WIDTH = 90;
    const LABEL_CHAR_WIDTH = 6.2; // approx px per monospace char at 10px font
    let lastLabelRight = -Infinity;

    for (const seg of segmentsRef.current) {
      const x1 = timeScale.timeToCoordinate(seg.start_time as Time);
      const x2 = timeScale.timeToCoordinate(seg.end_time as Time);
      if (x1 === null && x2 === null) continue;
      const left = x1 ?? 0;
      const right = x2 ?? width;
      if (right < 0 || left > width) continue;

      const band = document.createElement("div");
      band.style.position = "absolute";
      band.style.left = `${left}px`;
      band.style.top = "0";
      band.style.width = `${Math.max(right - left, 1)}px`;
      band.style.height = "100%";
      band.style.background = regimeBandColor(seg.regime);
      band.style.pointerEvents = "none";
      overlay.appendChild(band);

      const segWidth = right - left;
      const labelLeft = Math.max(left, 2);
      const estLabelWidth = seg.regime.length * LABEL_CHAR_WIDTH;
      const fitsSegment = segWidth > MIN_LABEL_WIDTH;
      const noOverlap = labelLeft > lastLabelRight + 8;
      if (fitsSegment && noOverlap) {
        const label = document.createElement("div");
        label.textContent = seg.regime;
        label.style.position = "absolute";
        label.style.left = `${labelLeft}px`;
        label.style.top = "4px";
        label.style.fontSize = "10px";
        label.style.fontFamily = "'JetBrains Mono', monospace";
        label.style.color = "rgba(210,218,232,0.75)";
        label.style.background = "rgba(13,15,20,0.55)";
        label.style.padding = "1px 4px";
        label.style.borderRadius = "2px";
        label.style.whiteSpace = "nowrap";
        label.style.pointerEvents = "none";
        overlay.appendChild(label);
        lastLabelRight = labelLeft + estLabelWidth + 8;
      }
    }
  }

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div ref={containerRef} style={{ position: "absolute", inset: 0, zIndex: 1 }} />
      <div ref={overlayRef} style={{ position: "absolute", inset: 0, zIndex: 2, pointerEvents: "none" }} />
    </div>
  );
}
