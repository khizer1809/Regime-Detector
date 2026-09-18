import type { Transition } from "../types";
import { regimeSolidColor } from "../services/regimeColors";

interface Props {
  transitions: Transition[];
  onSelect: (time: number) => void;
}

function fmtTime(unixSeconds: number): string {
  const d = new Date(unixSeconds * 1000);
  return d.toISOString().slice(0, 16).replace("T", " ");
}

export default function TransitionPanel({ transitions, onSelect }: Props) {
  const recent = transitions.slice(-30).reverse();
  return (
    <div className="transition-panel">
      <div className="panel-title">RECENT STATE CHANGES</div>
      <div className="transition-list">
        {recent.length === 0 && <div className="muted">No transitions in loaded window</div>}
        {recent.map((t, i) => (
          <button key={`${t.time}-${i}`} className="transition-item" onClick={() => onSelect(t.time)}>
            <span className="transition-time">{fmtTime(t.time)}</span>
            <span className="transition-flow">
              <span style={{ color: regimeSolidColor(t.from_regime) }}>{t.from_regime}</span>
              <span className="arrow"> → </span>
              <span style={{ color: regimeSolidColor(t.to_regime) }}>{t.to_regime}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
