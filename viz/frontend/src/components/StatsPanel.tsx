import type { StateStats } from "../types";
import { regimeSolidColor } from "../services/regimeColors";

interface Props {
  stats: StateStats | null;
}

function fmtDuration(minutes: number): string {
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

export default function StatsPanel({ stats }: Props) {
  if (!stats) return <div className="stats-panel empty">Loading stats...</div>;
  return (
    <div className="stats-panel">
      <div className="stats-row">
        <div className="stat-tile">
          <div className="stat-value">{stats.bars.toLocaleString()}</div>
          <div className="stat-label">Bars</div>
        </div>
        <div className="stat-tile">
          <div className="stat-value">{stats.state_transitions.toLocaleString()}</div>
          <div className="stat-label">State transitions</div>
        </div>
        <div className="stat-tile">
          <div className="stat-value">{stats.current_state?.regime ?? "-"}</div>
          <div className="stat-label">Current regime</div>
        </div>
        <div className="stat-tile">
          <div className="stat-value">{stats.current_state?.duration_bars ?? "-"}</div>
          <div className="stat-label">Current duration (bars)</div>
        </div>
        <div className="stat-tile">
          <div className="stat-value">{stats.longest_segment ? fmtDuration(stats.longest_segment.duration_minutes) : "-"}</div>
          <div className="stat-label">Longest segment</div>
        </div>
      </div>
      <div className="regime-dist">
        {Object.entries(stats.regime_distribution_pct).map(([regime, pct]) => (
          <div key={regime} className="dist-row">
            <span className="dist-dot" style={{ background: regimeSolidColor(regime) }} />
            <span className="dist-label">{regime}</span>
            <div className="dist-bar-bg">
              <div className="dist-bar-fill" style={{ width: `${pct}%`, background: regimeSolidColor(regime) }} />
            </div>
            <span className="dist-pct">{pct.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
