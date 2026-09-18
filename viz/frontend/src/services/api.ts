import type { Candle, Segment, Transition, ModelInfo, StateStats } from "../types";

const BASE_URL = "http://127.0.0.1:8000";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status} for ${path}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchModels(): Promise<{ models: ModelInfo[] }> {
  return getJSON("/api/models");
}

export async function fetchCandles(
  model: string,
  start?: string,
  end?: string,
  limit = 20000
): Promise<{ candles: Candle[]; truncated: boolean; count: number }> {
  const params = new URLSearchParams({ model: String(model), limit: String(limit) });
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  return getJSON(`/api/candles?${params.toString()}`);
}

export async function fetchSegments(
  model: string,
  start?: string,
  end?: string
): Promise<{ segments: Segment[] }> {
  const params = new URLSearchParams({ model: String(model) });
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  return getJSON(`/api/segments?${params.toString()}`);
}

export async function fetchTransitions(
  model: string,
  start?: string,
  end?: string,
  limit = 200
): Promise<{ transitions: Transition[] }> {
  const params = new URLSearchParams({ model: String(model), limit: String(limit) });
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  return getJSON(`/api/transitions?${params.toString()}`);
}

export async function fetchStateStats(
  model: string,
  start?: string,
  end?: string
): Promise<StateStats> {
  const params = new URLSearchParams({ model: String(model) });
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  return getJSON(`/api/state-stats?${params.toString()}`);
}

export async function fetchNearestCandle(
  model: string,
  timestamp: string
): Promise<{ requested: string; matched_timestamp: string; exact_match: boolean; time: number }> {
  const params = new URLSearchParams({ model: String(model), timestamp });
  return getJSON(`/api/nearest-candle?${params.toString()}`);
}
