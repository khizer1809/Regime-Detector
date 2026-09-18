import type { Candle } from "../types";

interface Props {
  hovered: Candle | null;
  fallback: Candle | null; // most recent candle in the loaded window, shown when nothing is hovered
}

function fmtTime(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toISOString().replace("T", " ").replace(".000Z", " UTC");
}

export default function StatePanel({ hovered, fallback }: Props) {
  const c = hovered ?? fallback;
  if (!c) {
    return (
      <div className="state-panel empty">
        <span>No data loaded</span>
      </div>
    );
  }
  return (
    <div className="state-panel">
      <div className="state-panel-header">{hovered ? "HOVERED CANDLE" : "CURRENT (latest in view)"}</div>
      <div className="state-panel-time">{fmtTime(c.time)}</div>
      <div className="ohlc-row">
        <span>O {c.open.toFixed(2)}</span>
        <span>H {c.high.toFixed(2)}</span>
        <span>L {c.low.toFixed(2)}</span>
        <span>C {c.close.toFixed(2)}</span>
      </div>
      <div className="ohlc-row muted">Vol {c.volume.toFixed(3)}</div>

      {c.regime ? (
        <>
          <div className="regime-name">{c.regime}</div>
          <div className="state-meta">
            <span>State {c.state}</span>
            {c.confidence != null && <span>Confidence {(c.confidence * 100).toFixed(1)}%</span>}
            {c.state_duration != null && <span>Duration {c.state_duration} bars</span>}
          </div>
        </>
      ) : (
        <div className="regime-name muted">No HMM state (data gap / warm-up)</div>
      )}
    </div>
  );
}
