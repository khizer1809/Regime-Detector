import type { ModelInfo } from "../types";

interface Props {
  models: ModelInfo[];
  selectedModel: string;
  onModelChange: (id: string) => void;
  dateInput: string;
  onDateInputChange: (v: string) => void;
  onJump: () => void;
  onPrevWindow: () => void;
  onNextWindow: () => void;
  onFit: () => void;
  onLatest: () => void;
  windowLabel: string;
  timeframe: string;
}

export default function ControlBar({
  models,
  selectedModel,
  onModelChange,
  dateInput,
  onDateInputChange,
  onJump,
  onPrevWindow,
  onNextWindow,
  onFit,
  onLatest,
  windowLabel,
  timeframe,
}: Props) {
  return (
    <div className="control-bar">
      <div className="control-group">
        <span className="symbol-badge">BTCUSDT</span>
        <span className="tf-badge">{timeframe}</span>
        <select value={selectedModel} onChange={(e) => onModelChange(e.target.value)} className="model-select">
          {models.map((m) => (
            <option key={m.model_id} value={m.model_id}>
              {m.model_id.endsWith("_overview")
                ? "Overview (full history, daily)"
                : m.model_id === "5m_candles_15m_regime"
                ? "5m candles (production 15m regime)"
                : `${m.timeframe}-HMM${m.n_states}${m.model_id.endsWith("_smoothed") ? " (smoothed)" : ""}`}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <button onClick={onPrevWindow} title="Previous window">
          ← Prev
        </button>
        <span className="window-label">{windowLabel}</span>
        <button onClick={onNextWindow} title="Next window">
          Next →
        </button>
      </div>

      <div className="control-group">
        <input
          type="datetime-local"
          value={dateInput}
          onChange={(e) => onDateInputChange(e.target.value)}
          className="date-input"
        />
        <button onClick={onJump}>Go</button>
      </div>

      <div className="control-group">
        <button onClick={onFit}>Fit</button>
        <button onClick={onLatest}>Latest</button>
      </div>
    </div>
  );
}
