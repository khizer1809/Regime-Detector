"""
v20_mtf_hmm_experiment.py -- EXPERIMENTAL, NOT PRODUCTION.

DEVELOPMENT / FEATURE-SCREENING EXPERIMENT -- NOT OOS WALK-FORWARD VALIDATION.

Tests whether adding causal 15m/30m EMA-based trend context to the existing
18 production features changes HMM regime-switching behavior (full-history
fits only, NOT the expensive 101-fold walk-forward).

REUSED, UNMODIFIED:
  - Data/features_out_masked.csv's 18 existing production feature columns
    (never redefined, never recomputed).
  - src/gap_aware.py's build_validity_mask() / segment_lengths() (same
    function, called with a WIDER max_lookback_bars for configs whose new
    features have longer lookback than the original 48-bar buffer -- see
    "GAP-AWARE BUFFER WIDENING" below).
  - src/infer_regime.py's characterize_state() (same TREND/VOL_Z_THRESHOLD
    labeling, unchanged).
  - HMM hyperparameters copied verbatim from save_production_model.py:
    GaussianHMM, covariance_type="diag", n_iter=100, 5 restarts (seeds
    42-46), best restart selected by TRAINING log-likelihood only.

NEW FEATURES (Section 6 -- avoiding duplication):
  The existing 18 features ALREADY include ret_15m and ret_30m (confirmed
  by reading Data/features_out_masked.csv's actual columns before designing
  anything) -- so plain 15m/30m returns would be pure duplicates. Instead:

    ema_dist_15m  = log(close / EMA(close, span=60, min_periods=60))
    ema_slope_15m = log(EMA_60[t] / EMA_60[t-3])
    ema_dist_30m  = log(close / EMA(close, span=120, min_periods=120))
    ema_slope_30m = log(EMA_120[t] / EMA_120[t-6])

  span=60 bars (5h) approximates a 20-period EMA on 15m-resampled bars
  (20 x 3 5m-bars-per-15m-bar = 60); span=120 (10h) approximates a
  20-period EMA on 30m bars (20 x 6 = 120) -- computed directly on the
  causal 5m close series (no literal resampling), per the task's explicit
  "keep 5m as the base frequency, do not resample the entire dataset"
  instruction. Both EMA_dist and EMA_slope are log-ratios -- scale-free,
  immune to BTC's multi-year price-level drift (the same non-stationarity
  class of issue already found and fixed for a different feature earlier
  this session). Nothing in the existing 18 features is EMA-based or
  operates on a >4h effective lookback -- these are genuinely new
  information, not redundant with any existing column (correlation-checked
  in Part 17 of the report, not assumed).

  4 new features total: B = 18+2 (15m only), C = 18+2 (30m only),
  D = 18+4 (15m+30m combined).

CAUSALITY: .ewm(adjust=False) is a pure forward recursion (each value
depends only on the current observation and the previous EMA value, never
anything after t) -- no centering, no lookahead by construction. Slopes
compare EMA[t] to EMA[t-lag], both already-causal values.

GAP-AWARE BUFFER WIDENING: the original MAX_LOOKBACK_BARS=48 was
specifically sized to match the ORIGINAL 18 features' longest lookback
(ret_4h/skew_4h = 48 bars). The new EMA features have longer effective
lookback (60/120 bars) -- reusing the 48-bar buffer would under-protect
them relative to how the original design protects its own longest
feature. gap_aware.build_validity_mask() already accepts max_lookback_bars
as a parameter, so this widens it per-config (A=48 unchanged, B=60,
C=120, D=120) rather than inventing a new masking mechanism.

NO walk-forward, NO OOS validation, NO train/test split -- full-history
fits only, exactly mirroring save_production_model.py's own methodology,
per the task's explicit scope.

Run: `python v20_mtf_hmm_experiment.py` (from research_archive/src/).
"""

import json
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import persistence_common  # noqa: E402
from gap_aware import segment_lengths  # noqa: E402
from infer_regime import RETURN_COLS, VOL_COLS, characterize_state  # noqa: E402
from save_production_model import load_masked_features  # noqa: E402

# ===========================================================================
# KNOWN BUG FOUND WHILE BUILDING THIS EXPERIMENT (disclosed, not fixed here):
# gap_aware.build_validity_mask(raw_df, feature_index, max_lookback_bars=48)
# accepts max_lookback_bars as a parameter but its body hardcodes the literal
# 48 on the `is_zero_vol.rolling(48, ...)` line -- the parameter is dead,
# never read. This means calling the EXISTING function with a larger value
# would silently still use a 48-bar buffer, under-protecting the new EMA
# features (60/120-bar lookback) from gap contamination. Per this task's
# explicit "do not modify existing gap-aware methodology" instruction, that
# function is NOT touched. Instead, a correctly-parametrized local
# equivalent is used for THIS experiment only (identical logic, the one
# fix: max_lookback_bars actually reaches the rolling window). Flagged here
# and in the final report as a real, pre-existing issue worth fixing in
# gap_aware.py itself, separately from this experiment.
# ===========================================================================

def build_validity_mask_fixed(raw_df, feature_index, max_lookback_bars):
    is_zero_vol = (raw_df["volume"] == 0)
    touched_by_gap = is_zero_vol.rolling(max_lookback_bars, min_periods=1).max()
    valid = (touched_by_gap == 0)
    valid = valid.reindex(feature_index)
    valid = valid.fillna(False)
    return valid

RAW_OHLCV_PATH = r"C:\Users\MohammedkhezerK\Downloads\New System\ICT_STRUCTURE_RESEARCH\data\BTCUSDT_5M.csv"
OUT_DIR = os.path.join(_PROJECT_ROOT, "results")
CHART_OUT_DIR = os.path.join(_PROJECT_ROOT, "results", "chart_data")
REPORT_PATH = os.path.join(_PROJECT_ROOT, "reports", "mtf_hmm_A_to_D_report.md")

COVARIANCE_TYPE = "diag"
N_ITER = 100
RANDOM_STATE = 42
N_RESTARTS = 5
N_WORKERS = min(6, os.cpu_count() or 1)
N_STATES_LIST = [4, 6]
CONFIGS = ["A", "B", "C", "D"]

EMA_15M_SPAN = 60   # ~5h, "20-period EMA on 15m bars" equivalent
EMA_15M_SLOPE_LAG = 3
EMA_30M_SPAN = 120  # ~10h, "20-period EMA on 30m bars" equivalent
EMA_30M_SLOPE_LAG = 6

FUTURE_HORIZONS = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHART_OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Feature construction
# ===========================================================================

def load_raw_ohlcv() -> pd.DataFrame:
    log(f"Loading raw OHLCV from {RAW_OHLCV_PATH} ...")
    df = pd.read_csv(RAW_OHLCV_PATH, usecols=["timestamp", "open", "high", "low", "close", "volume"],
                      parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_new_features(raw: pd.DataFrame) -> pd.DataFrame:
    close = raw["close"]
    log_close = np.log(close)

    ema15 = close.ewm(span=EMA_15M_SPAN, adjust=False, min_periods=EMA_15M_SPAN).mean()
    ema_dist_15m = np.log(close / ema15)
    ema_slope_15m = np.log(ema15 / ema15.shift(EMA_15M_SLOPE_LAG))

    ema30 = close.ewm(span=EMA_30M_SPAN, adjust=False, min_periods=EMA_30M_SPAN).mean()
    ema_dist_30m = np.log(close / ema30)
    ema_slope_30m = np.log(ema30 / ema30.shift(EMA_30M_SLOPE_LAG))

    return pd.DataFrame({
        "timestamp": raw["timestamp"], "ema_dist_15m": ema_dist_15m, "ema_slope_15m": ema_slope_15m,
        "ema_dist_30m": ema_dist_30m, "ema_slope_30m": ema_slope_30m,
    })


def build_config_matrix(existing18: pd.DataFrame, new_feats: pd.DataFrame, config: str):
    if config == "A":
        feat_cols = [c for c in existing18.columns if c != "valid"]
        max_lookback = 48
    elif config == "B":
        feat_cols = [c for c in existing18.columns if c != "valid"] + ["ema_dist_15m", "ema_slope_15m"]
        max_lookback = max(48, EMA_15M_SPAN + EMA_15M_SLOPE_LAG)
    elif config == "C":
        feat_cols = [c for c in existing18.columns if c != "valid"] + ["ema_dist_30m", "ema_slope_30m"]
        max_lookback = max(48, EMA_30M_SPAN + EMA_30M_SLOPE_LAG)
    elif config == "D":
        feat_cols = [c for c in existing18.columns if c != "valid"] + [
            "ema_dist_15m", "ema_slope_15m", "ema_dist_30m", "ema_slope_30m"]
        max_lookback = max(48, EMA_30M_SPAN + EMA_30M_SLOPE_LAG)
    else:
        raise ValueError(config)
    return feat_cols, max_lookback


# ===========================================================================
# HMM fit + decode
# ===========================================================================

def _fit_one(train_values, lengths, n_states, seed):
    model = GaussianHMM(n_components=n_states, covariance_type=COVARIANCE_TYPE, n_iter=N_ITER, random_state=seed)
    t0 = time.time()
    try:
        model.fit(train_values, lengths=lengths)
        ll = model.score(train_values, lengths=lengths)
        return {"ll": ll, "model": model, "error": None, "runtime": time.time() - t0,
                "converged": bool(model.monitor_.converged), "n_iter_run": int(model.monitor_.iter)}
    except Exception as e:
        return {"ll": None, "model": None, "error": f"{type(e).__name__}: {e}", "runtime": time.time() - t0}


def fit_hmm(train_values, lengths, n_states):
    seeds = [RANDOM_STATE + r for r in range(N_RESTARTS)]
    best = {"ll": float("-inf"), "model": None, "seed": None}
    restart_log = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_fit_one, train_values, lengths, n_states, s): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            r = fut.result()
            restart_log.append({"seed": seed, "ll": r["ll"], "error": r["error"], "runtime_s": r["runtime"],
                                 "converged": r.get("converged"), "n_iter_run": r.get("n_iter_run")})
            if r["error"] is None and r["ll"] > best["ll"]:
                best = {"ll": r["ll"], "model": r["model"], "seed": seed}
    return best["model"], best["ll"], best["seed"], restart_log


def build_segments(states: np.ndarray, seg_id: np.ndarray, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(states)
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = (states[1:] != states[:-1]) | (seg_id[1:] != seg_id[:-1])
    run_id = np.cumsum(changed)
    tmp = pd.DataFrame({"state": states, "run_id": run_id, "timestamp": timestamps})
    segs = tmp.groupby("run_id").agg(state=("state", "first"), start_timestamp=("timestamp", "first"),
                                      end_timestamp=("timestamp", "last"), number_of_bars=("timestamp", "size")
                                      ).reset_index(drop=True)
    segs["duration_minutes"] = segs["number_of_bars"] * 5.0
    return segs


def build_transitions(segments: pd.DataFrame) -> pd.DataFrame:
    if len(segments) < 2:
        return pd.DataFrame(columns=["timestamp", "from_state", "to_state"])
    rows = []
    for i in range(1, len(segments)):
        prev, cur = segments.iloc[i - 1], segments.iloc[i]
        if prev["state"] != cur["state"]:
            rows.append({"timestamp": cur["start_timestamp"], "from_state": int(prev["state"]),
                         "to_state": int(cur["state"])})
    return pd.DataFrame(rows)


def future_return_stats(states, forward_rets_by_horizon, n_states):
    """Section 19/20: evaluation-only, never fed to the HMM."""
    rows = []
    for state in range(n_states):
        mask = states == state
        for h_name, fr in forward_rets_by_horizon.items():
            vals = fr[mask]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            rows.append({
                "state": state, "horizon": h_name, "n": len(vals),
                "mean_ret": float(np.mean(vals)), "median_ret": float(np.median(vals)),
                "std_ret": float(np.std(vals)), "p_positive": float((vals > 0).mean()),
                "p_gt_0.5pct": float((vals > 0.005).mean()), "p_lt_-0.5pct": float((vals < -0.005).mean()),
            })
    return pd.DataFrame(rows)


def state_separation_tests(states, forward_rets_by_horizon, n_states):
    """Kruskal-Wallis across states per horizon -- descriptive only, not proof of trading usefulness."""
    rows = []
    for h_name, fr in forward_rets_by_horizon.items():
        groups = []
        for state in range(n_states):
            mask = states == state
            vals = fr[mask]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 10:
                groups.append(vals)
        if len(groups) >= 2:
            try:
                stat, p = stats.kruskal(*groups)
                rows.append({"horizon": h_name, "kruskal_stat": float(stat), "p_value": float(p),
                             "n_groups": len(groups),
                             "note": "descriptive only -- large n inflates significance; not proof of tradability"})
            except Exception as e:
                rows.append({"horizon": h_name, "error": str(e)})
    return pd.DataFrame(rows)


def run_one_model(config, n_states, feat_cols, max_lookback, raw, existing18, new_feats, feature_corr_cache):
    fold_dir_tag = f"{config}-HMM{n_states}"
    log(f"=== {fold_dir_tag}: features={feat_cols} max_lookback={max_lookback} ===")

    # existing18 is indexed by timestamp (from features_out_masked.csv); align new_feats (a timestamp COLUMN,
    # built from raw) onto that same index via an explicit join, never positional concatenation.
    combined = existing18.drop(columns=["valid"]).join(new_feats.set_index("timestamp"), how="left")
    # build_validity_mask_fixed needs raw_df indexed by timestamp too, matching how it's used correctly
    # elsewhere in this project (e.g. binance_fetch.py's output) -- raw here has a plain RangeIndex instead
    # (timestamp is a column), which would silently reindex to all-False if passed directly.
    raw_indexed = raw.set_index("timestamp")
    valid_flag = build_validity_mask_fixed(raw_indexed, existing18.index, max_lookback_bars=max_lookback)
    feat_df = combined[feat_cols]
    valid_mask = valid_flag.reindex(feat_df.index).fillna(False) & feat_df.notna().all(axis=1)
    valid_index = feat_df.index[valid_mask]
    valid_values = feat_df.loc[valid_mask].values
    lengths = segment_lengths(valid_index)
    assert sum(lengths) == len(valid_values)
    seg_id = persistence_common.segment_ids_from_lengths(lengths)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(valid_values)

    t0 = time.time()
    model, best_ll, best_seed, restart_log = fit_hmm(scaled, lengths, n_states)
    fit_time = time.time() - t0
    if model is None:
        return {"config": config, "n_states": n_states, "error": "all restarts failed", "restart_log": restart_log}

    t1 = time.time()
    states = model.predict(scaled, lengths=lengths)
    posteriors = model.predict_proba(scaled, lengths=lengths)
    decode_time = time.time() - t1

    means_df = pd.DataFrame(model.means_, columns=feat_cols)
    # characterize_state only looks at RETURN_COLS/VOL_COLS -- present in every config since the 18 base features
    # are always included, so this works unmodified regardless of the extra new columns.
    regime_labels = [characterize_state(means_df.iloc[s]) for s in range(n_states)]

    segments = build_segments(states, seg_id, valid_index)
    segments["regime"] = segments["state"].map(dict(enumerate(regime_labels)))
    transitions = build_transitions(segments)

    n_valid = len(valid_index)
    bars = segments["number_of_bars"]
    total_days = (valid_index.max() - valid_index.min()).total_seconds() / 86400

    # forward returns (evaluation only)
    close_full = raw.set_index("timestamp")["close"].reindex(valid_index)
    log_close = np.log(close_full.values)
    fwd_by_h = {}
    for h_name, h_bars in FUTURE_HORIZONS.items():
        j = np.arange(n_valid) + h_bars
        fwd_valid = (j < n_valid) & (seg_id[np.minimum(j, n_valid - 1)] == seg_id)
        fr = np.full(n_valid, np.nan)
        fr[fwd_valid] = log_close[j[fwd_valid]] - log_close[np.arange(n_valid)[fwd_valid]]
        fwd_by_h[h_name] = fr

    future_ret_df = future_return_stats(states, fwd_by_h, n_states)
    future_ret_df["config"] = config
    future_ret_df["n_states_model"] = n_states
    future_ret_df["regime"] = future_ret_df["state"].map(dict(enumerate(regime_labels)))

    separation_df = state_separation_tests(states, fwd_by_h, n_states)
    separation_df["config"] = config
    separation_df["n_states_model"] = n_states

    state_stats_rows = []
    for s in range(n_states):
        sub = segments[segments["state"] == s]
        n_bars_state = int((states == s).sum())
        state_stats_rows.append({
            "config": config, "n_states_model": n_states, "state": s, "regime": regime_labels[s],
            "n_segments": len(sub), "n_bars": n_bars_state, "pct_bars": n_bars_state / n_valid * 100,
            "mean_duration_bars": float(sub["number_of_bars"].mean()) if len(sub) else None,
            "median_duration_bars": float(sub["number_of_bars"].median()) if len(sub) else None,
            "max_duration_bars": int(sub["number_of_bars"].max()) if len(sub) else None,
            "stay_prob": float(model.transmat_[s, s]),
        })

    n_transitions = len(transitions)
    model_stats = {
        "config": config, "n_states_model": n_states, "n_features": len(feat_cols),
        "feature_columns": feat_cols, "max_lookback_bars": int(max_lookback),
        "n_valid_bars": n_valid, "total_days": total_days,
        "n_transitions": n_transitions, "n_segments": len(segments),
        "transitions_per_1k_bars": n_transitions / n_valid * 1000,
        "transitions_per_day": n_transitions / total_days if total_days > 0 else None,
        "mean_segment_bars": float(bars.mean()), "median_segment_bars": float(bars.median()),
        "min_segment_bars": int(bars.min()), "max_segment_bars": int(bars.max()),
        "pct_1bar": float((bars == 1).mean() * 100), "pct_2bar": float((bars == 2).mean() * 100),
        "pct_3bar": float((bars == 3).mean() * 100), "pct_le5bar": float((bars <= 5).mean() * 100),
        "pct_le10bar": float((bars <= 10).mean() * 100),
        "pct_gt1h": float((bars > 12).mean() * 100), "pct_gt2h": float((bars > 24).mean() * 100),
        "pct_gt4h": float((bars > 48).mean() * 100), "pct_gt8h": float((bars > 96).mean() * 100),
        "n_1bar": int((bars == 1).sum()), "n_2bar": int((bars == 2).sum()), "n_3bar": int((bars == 3).sum()),
        "n_le5bar": int((bars <= 5).sum()), "n_le10bar": int((bars <= 10).sum()),
        "n_gt1h": int((bars > 12).sum()), "n_gt2h": int((bars > 24).sum()),
        "n_gt4h": int((bars > 48).sum()), "n_gt8h": int((bars > 96).sum()),
        "train_ll": best_ll, "train_ll_per_obs": best_ll / n_valid, "best_seed": best_seed,
        "fit_time_s": fit_time, "decode_time_s": decode_time, "distinct_regime_labels": len(set(regime_labels)),
        "regime_labels": {int(s): r for s, r in enumerate(regime_labels)},
        "transmat": model.transmat_.tolist(),
    }

    # feature correlation of NEW features vs EXISTING 18 (computed once per config, cached)
    if config != "A" and config not in feature_corr_cache:
        new_cols = [c for c in feat_cols if c.startswith("ema_")]
        corr = feat_df.loc[valid_mask, feat_cols].corr()
        feature_corr_cache[config] = corr.loc[new_cols, [c for c in feat_cols if c not in new_cols]]

    # save artifacts (checkpoint)
    tag = f"{config}_HMM{n_states}"
    with open(os.path.join(OUT_DIR, f"model_{tag}.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_columns": feat_cols}, f)
    pd.DataFrame(state_stats_rows).to_csv(os.path.join(OUT_DIR, f"state_stats_{tag}.csv"), index=False)
    segments.to_csv(os.path.join(OUT_DIR, f"segments_{tag}.csv"), index=False)
    future_ret_df.to_csv(os.path.join(OUT_DIR, f"future_returns_{tag}.csv"), index=False)
    separation_df.to_csv(os.path.join(OUT_DIR, f"separation_{tag}.csv"), index=False)
    json.dump(model_stats, open(os.path.join(OUT_DIR, f"model_stats_{tag}.json"), "w"), indent=2, default=str)
    pd.DataFrame(restart_log).to_csv(os.path.join(OUT_DIR, f"restarts_{tag}.csv"), index=False)

    # chart-format export (for potential future visualization, not wired into the live app)
    candles = raw[["timestamp", "open", "high", "low", "close", "volume"]].merge(
        pd.DataFrame({"timestamp": valid_index, "state": states, "regime": per_bar_labels(states, regime_labels)}),
        on="timestamp", how="left")
    candles.to_parquet(os.path.join(CHART_OUT_DIR, f"{tag}_candles.parquet"), index=False)
    segments.to_parquet(os.path.join(CHART_OUT_DIR, f"{tag}_segments.parquet"), index=False)
    transitions.to_parquet(os.path.join(CHART_OUT_DIR, f"{tag}_transitions.parquet"), index=False)

    log(f"  {tag}: n_valid={n_valid:,} transitions={n_transitions:,} "
        f"({model_stats['transitions_per_1k_bars']:.2f}/1k bars) median_seg={model_stats['median_segment_bars']}bars "
        f"LL/obs={model_stats['train_ll_per_obs']:.4f} distinct_labels={model_stats['distinct_regime_labels']}/{n_states}")

    return model_stats


def per_bar_labels(states, regime_labels):
    return [regime_labels[s] for s in states]


def main():
    t0 = time.time()
    raw = load_raw_ohlcv()
    log(f"  {len(raw):,} raw candles")

    existing18, model_cols = load_masked_features()
    new_feats = build_new_features(raw)
    log(f"  existing 18 features loaded, {len(model_cols)} columns: {model_cols}")
    log(f"  new EMA features built: ema_dist_15m, ema_slope_15m (span={EMA_15M_SPAN}), "
        f"ema_dist_30m, ema_slope_30m (span={EMA_30M_SPAN})")

    # --- benchmark, then proceed automatically (Section 27) ---
    feat_cols_D, max_lb_D = build_config_matrix(existing18, new_feats, "D")
    combined_D = existing18.drop(columns=["valid"]).join(new_feats.set_index("timestamp"), how="left")
    raw_indexed = raw.set_index("timestamp")
    valid_flag_D = build_validity_mask_fixed(raw_indexed, existing18.index, max_lookback_bars=max_lb_D)
    feat_df_D = combined_D[feat_cols_D]
    valid_mask_D = valid_flag_D.reindex(feat_df_D.index).fillna(False) & feat_df_D.notna().all(axis=1)
    bench_values = feat_df_D.loc[valid_mask_D].values
    bench_lengths = segment_lengths(feat_df_D.index[valid_mask_D])
    bench_scaled = StandardScaler().fit_transform(bench_values)
    log(f"Benchmarking D-HMM6 (largest config) on {len(bench_values):,} rows...")
    tb0 = time.time()
    _fit_one(bench_scaled, bench_lengths, 6, RANDOM_STATE)
    bench_time = time.time() - tb0
    n_jobs = len(CONFIGS) * len(N_STATES_LIST)
    est_total = bench_time * n_jobs * 0.75  # most configs cheaper than D-HMM6 (fewer features/states)
    log(f"  1 restart of D-HMM6 took {bench_time:.1f}s")
    log(f"ESTIMATED total runtime: {n_jobs} configs x ~{bench_time:.0f}s ~= {est_total:.0f}s (~{est_total/60:.1f} min). "
        f"Proceeding automatically now.")

    all_model_stats = []
    feature_corr_cache = {}
    failed = []
    jobs = [(c, n) for c in CONFIGS for n in N_STATES_LIST]
    pbar = tqdm(jobs, desc="MTF HMM experiment (A-D x HMM4/6)")
    for config, n_states in pbar:
        pbar.set_postfix_str(f"{config}-HMM{n_states}")
        feat_cols, max_lookback = build_config_matrix(existing18, new_feats, config)
        try:
            result = run_one_model(config, n_states, feat_cols, max_lookback, raw, existing18, new_feats,
                                    feature_corr_cache)
            all_model_stats.append(result)
        except Exception as e:
            tb = traceback.format_exc()
            log(f"  {config}-HMM{n_states} FAILED: {e}\n{tb}")
            failed.append({"config": config, "n_states": n_states, "error": str(e), "traceback": tb})
        # checkpoint after every job
        json.dump({"completed": all_model_stats, "failed": failed}, open(os.path.join(OUT_DIR, "mtf_checkpoint.json"), "w"),
                   indent=2, default=str)

    runtime = time.time() - t0
    log(f"Total runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    # feature correlation report
    corr_rows = []
    for config, corr_df in feature_corr_cache.items():
        for new_feat in corr_df.index:
            for old_feat in corr_df.columns:
                corr_rows.append({"config": config, "new_feature": new_feat, "existing_feature": old_feat,
                                  "correlation": float(corr_df.loc[new_feat, old_feat])})
    corr_out = pd.DataFrame(corr_rows)
    corr_out.to_csv(os.path.join(OUT_DIR, "mtf_feature_correlations.csv"), index=False)

    pd.DataFrame(all_model_stats).to_csv(os.path.join(OUT_DIR, "mtf_hmm_model_results.csv"), index=False)
    json.dump({"completed": all_model_stats, "failed": failed, "runtime_s": runtime},
               open(os.path.join(OUT_DIR, "mtf_hmm_final_results.json"), "w"), indent=2, default=str)

    log(f"Completed {len(all_model_stats)}/{n_jobs} configs. Failed: {len(failed)}")
    log(f"All artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
