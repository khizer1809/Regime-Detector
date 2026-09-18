import { regimeSolidColor } from "../services/regimeColors";

interface Props {
  regimeLabels: Record<string, string>;
}

export default function RegimeLegend({ regimeLabels }: Props) {
  const uniqueRegimes = Array.from(new Set(Object.values(regimeLabels)));
  return (
    <div className="regime-legend">
      {uniqueRegimes.map((r) => (
        <span key={r} className="legend-item">
          <span className="legend-dot" style={{ background: regimeSolidColor(r) }} />
          {r}
        </span>
      ))}
    </div>
  );
}
