export interface Candle {
  time: number; // unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  state: number | null;
  regime: string | null;
  confidence: number | null;
  state_duration: number | null;
}

export interface Segment {
  segment_id: number;
  state: number;
  regime: string;
  start_time: number;
  end_time: number;
  number_of_bars: number;
  duration_minutes: number;
}

export interface Transition {
  time: number;
  from_state: number;
  from_regime: string;
  to_state: number;
  to_regime: string;
}

export interface ModelInfo {
  model_id: string;
  timeframe: string;
  n_states: number;
  regime_labels: Record<string, string>;
  n_candles: number;
  n_matched: number;
  n_segments: number;
  n_transitions: number;
  date_range: [string, string];
  train_end: string;
}

export interface StateStats {
  model: string;
  bars: number;
  labeled_bars: number;
  state_transitions: number;
  current_state: { state: number; regime: string; duration_bars: number | null } | null;
  longest_segment: { regime: string; duration_minutes: number } | null;
  regime_distribution_pct: Record<string, number>;
}
