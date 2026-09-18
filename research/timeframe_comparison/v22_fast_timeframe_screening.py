"""
v22_fast_timeframe_screening.py -- EXPERIMENTAL, NOT PRODUCTION.

FAST TIMEFRAME SCREENING -- full-history DEVELOPMENT fits only, NOT the
expensive walk-forward. Purpose: quickly identify whether 15m-base, 30m-base,
or a 15m+30m weighted MTF score produces more coherent HMM regimes than the
existing 5m production model, so only the survivors get sent to the (much
more expensive) walk-forward validation later. Does NOT modify any
production file.

CONFIGS: A=5m (existing, decode-only, NOT refit) / B=15m / C=30m /
D=15m+30m weighted score (built from B and C's own outputs, no new HMM fit).
Each of A/B/C run at HMM-4 and HMM-6.

===========================================================================
REUSED, UNMODIFIED
===========================================================================
- HMM hyperparameters copied verbatim from save_production_model.py:
  GaussianHMM, covariance_type="diag", n_iter=100, 5 restarts (seeds 42-46),
  best restart selected by TRAINING log-likelihood only.
- infer_regime.py's TREND_Z_THRESHOLD/VOL_Z_THRESHOLD=0.3 labeling
  convention (re-applied to each timeframe's own return/vol columns --
  see characterize_state_tf below).
- Results/model_A_HMM4.pkl, results/model_A_HMM6.pkl -- the pure-18-feature,
  full-history 5m fits already completed in v20_mtf_hmm_experiment.py's
  config "A". Per this task's explicit "do not retrain A unless absolutely
  necessary" instruction, these are loaded and RE-DECODED ONLY (cheap --
  no new HMM fit for the 5m baseline).

===========================================================================
TWO REAL, PRE-EXISTING HARDCODED-ASSUMPTION BUGS FOUND WHILE BUILDING THIS
(disclosed, NOT fixed in the original files, per "do not modify existing
methodology" -- local corrected equivalents used here instead, same pattern
already used in v20_mtf_hmm_experiment.py for the first one):

1. gap_aware.build_validity_mask(raw_df, feature_index, max_lookback_bars)
   accepts max_lookback_bars but its body hardcodes `.rolling(48, ...)` --
   the parameter is dead. (Same bug already disclosed in v20's report.)

2. NEW, found here: gap_aware.segment_lengths(index) hardcodes
   BAR_INTERVAL = pd.Timedelta(minutes=5) at module level and compares
   consecutive-row time diffs against that fixed 5-minute constant to decide
   where a contiguous run breaks. Called on 15m or 30m data, EVERY row would
   be (incorrectly) flagged as starting a new segment, because real 15m/30m
   spacing (15min/30min) never equals the hardcoded 5-minute constant. Not
   a "dead parameter" like #1 -- here there is no way to override it at all
   from the caller. A local `segment_lengths_native()` (parameterized by the
   caller's own bar interval) is used instead for anything that isn't 5m.

===========================================================================
FEATURE SET (18 per timeframe, mirrors production's 18-feature count) --
mapped to the closest existing production concept (features.py), NOT a new
feature-engineering approach, with windows re-expressed in ECONOMIC time
(Section 6) rather than the old 5m bar counts:

  RETURNS (5):     N-bar log(close) diffs at economic 15m/30m/1h/2h/4h
                    (15m-base) or 30m/1h/2h/4h/8h (30m-base).
                    -> production concept: add_returns
  VOLATILITY (4):  rolling std of the 1-bar log return, same horizons minus
                    the timeframe's own native 1-bar step.
                    -> production concept: add_volatility
  TREND (3):       ema_dist_fast, ema_dist_slow, ema_slope_fast (causal
                    EMA log-distance/slope -- production itself has no EMA
                    family, so this reuses v20_mtf_hmm_experiment.py's own
                    already-vetted EMA design, the closest existing concept
                    in THIS codebase, rather than inventing a new one).
                    Section 5's momentum_4/momentum_8 are mathematically
                    IDENTICAL to N-bar log(close) diffs at bars 4/8 -- i.e.
                    identical to columns already present in the RETURNS
                    family above (ret_1h/ret_2h at 15m-base, ret_2h/ret_4h
                    at 30m-base). Rather than feed a diagonal-covariance
                    Gaussian HMM two literally-duplicate columns (extra
                    weight on that axis of the likelihood with zero new
                    information), they are treated as aliases of the
                    existing return columns and NOT added a second time --
                    disclosed here and in the report, not silently dropped.
  VOLUME (2):      vol_change, vol_zscore. -> production concept:
                    add_volume_orderflow (buy_sell_ratio and vwap_dist are
                    NOT in this experiment's Section 5 feature list and are
                    intentionally excluded, per the task's own explicit
                    compact list).
  ORDER FLOW (2):  ofi_raw, ofi_zscore, RECOMPUTED at the aggregated
                    timeframe as sum(buy_vol) - sum(sell_vol) (matching
                    binance_fetch.py's own documented ofi=buy_vol-sell_vol
                    formula) rather than summing the raw 5m `ofi` column --
                    the raw file's `ofi` column was found (verified
                    directly against the file) to carry an undocumented
                    1-bar lag relative to its own buy_vol-sell_vol
                    (ofi[i] == buy_vol[i-1]-sell_vol[i-1] for the entire
                    file); summing that lagged column would propagate a
                    source-file quirk into a new artifact rather than a
                    genuine aggregated order-flow measure. buy_vol/sell_vol
                    are directly additive trade-volume counts, so this
                    recomputation is not an invented feature -- it is the
                    correct aggregation of the same underlying data.
  DISTRIBUTIONAL(2): skew_1h (rolling skew of the 1-bar log return over the
                    economic-1h window), updown_asymmetry (same formula as
                    features.py's add_skew, same winsorization).
                    -> production concept: add_skew

===========================================================================
GENUINE 15m/30m CANDLES (Section 3/4)
===========================================================================
Built directly from the raw 5m OHLCV file via clock-aligned resampling
(open=first, high=max, low=min, close=last, volume=sum). The raw 5m file
was verified (programmatically, not assumed) to be a single perfectly
contiguous 5-minute series from 2017-09-01 05:00 UTC to 2026-07-18 23:55 UTC
(933,924 rows, zero missing timestamps -- only 2,183 zero-VOLUME "inactivity"
bars, which is the project's own existing definition of a gap, not a missing
row). Because the series starts on a clock-aligned boundary and its length
divides evenly by 3 (15m) and 6 (30m), every resample bucket contains exactly
the expected number of real 5m bars; any bucket that doesn't (only possible
at the very start/end of an arbitrary date range) is DROPPED, not fabricated
from partial data.

===========================================================================
MTF SCORE (Section 11/12) -- D config
===========================================================================
No combined-feature HMM is trained. Instead, for each of B (15m) and C (30m),
every state s has a continuous z-scored "trend_score" (the same
RETURN_COLS-mean quantity characterize_state_tf uses to decide Uptrend/
Downtrend/Ranging) and every bar has a posterior "confidence" (probability
mass on its Viterbi-decoded state). Two derived series:

  DIRECTION_SCORE_t = 0.6 * trend_score_15m[state_15m_t]
                     + 0.4 * trend_score_30m[state_30m_t]     (unweighted by
                       confidence -- used to bin into Bullish/Ranging/Bearish
                       via the SAME +/-0.3 threshold convention used
                       throughout this project, so >0.3=bullish, <-0.3=
                       bearish, else ranging, directly matching Section 11's
                       "positive=bullish/negative=bearish/near-zero=range.")

  MTF_SCORE_t        = 0.6 * (trend_score_15m[state_15m_t] * confidence_15m_t)
                     + 0.4 * (trend_score_30m[state_30m_t] * confidence_30m_t)
                       (confidence-weighted continuous signal -- used only
                       for the future-return coherence check, Section 14).

CAUSAL 30m->15m ALIGNMENT (Section 12): the 30m aggregate candle at index
timestamp t0 (label="left") covers [t0, t0+30m) and is only fully KNOWN at
t0+30m (its close time). Its score is therefore time-stamped as available at
t0+30m, and for every 15m timestamp t, the 30m score used is the one from
the most recently COMPLETED 30m bar (merge_asof, direction="backward" on
this "available_at" timestamp) -- never the 30m bar currently forming. E.g.
a 15m bar at 10:15 sees the 30m bar covering [09:30,10:00) (available at
10:00), never the one covering [10:00,10:30) (not available until 10:30).

Run: `python v22_fast_timeframe_screening.py` (from research_archive/src/).
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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import persistence_common  # noqa: E402
from save_production_model import load_masked_features  # noqa: E402

RAW_OHLCV_PATH = r"C:\Users\MohammedkhezerK\Downloads\New System\ICT_STRUCTURE_RESEARCH\data\BTCUSDT_5M.csv"
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
REPORT_PATH = os.path.join(_PROJECT_ROOT, "reports", "fast_timeframe_hmm_screening_report.md")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# ---------------------------------------------------------------------------
# HMM hyperparameters -- copied verbatim from save_production_model.py
# ---------------------------------------------------------------------------
COVARIANCE_TYPE = "diag"
N_ITER = 100
RANDOM_STATE = 42
N_RESTARTS = 5
N_WORKERS = min(6, os.cpu_count() or 1)
N_STATES_LIST = [4, 6]

TREND_Z_THRESHOLD = 0.3
VOL_Z_THRESHOLD = 0.3

TF_MINUTES = {"15m": 15, "30m": 30}
TF_BARS_PER_5M = {"15m": 3, "30m": 6}

# Section 6: economic-time-horizon bar counts, NOT copied 5m bar counts.
RETURN_HORIZON_BARS = {
    "15m": {"15m": 1, "30m": 2, "1h": 4, "2h": 8, "4h": 16},
    "30m": {"30m": 1, "1h": 2, "2h": 4, "4h": 8, "8h": 16},
}
RETURN_COLS_TF = {
    "15m": ["ret_15m", "ret_30m", "ret_1h", "ret_2h", "ret_4h"],
    "30m": ["ret_30m", "ret_1h", "ret_2h", "ret_4h", "ret_8h"],
}
# mirrors production's own choice of the 3 SHORTEST vol windows for
# characterize_state (production excludes vol_4h from VOL_COLS)
VOL_COLS_TF = {
    "15m": ["vol_30m", "vol_1h", "vol_2h"],
    "30m": ["vol_1h", "vol_2h", "vol_4h"],
}
FUTURE_HORIZON_BARS = RETURN_HORIZON_BARS  # same economic horizons, reused for forward-return eval

EMA_FAST_BARS = 8
EMA_SLOW_BARS = 24
EMA_SLOPE_LAG_BARS = 3

# Section 5's "rolling skewness over an economically reasonable horizon":
# the literal economic-1h window is only 2 bars at 30m-base
# (RETURN_HORIZON_BARS["30m"]["1h"]==2) -- pandas' rolling skew is
# STRUCTURALLY undefined (NaN 100% of the time, not just noisy) for any
# window < 3, since the bias-corrected skew formula divides by (n-2).
# Verified directly: build_tf_features(30m) produced skew_1h NaN for all
# 5,000/5,000 rows of a smoke-test slice, which zeroed out validity
# entirely. "Economically reasonable" is read here as "the shortest
# horizon in this timeframe's own ladder wide enough to define skew at
# all" -- both timeframes land on 8 bars this way (15m's "2h", 30m's
# "4h"), keeping the two timeframes' distributional feature on
# comparable sample size rather than comparable wall-clock label.
SKEW_WINDOW_LABEL = {"15m": "2h", "30m": "4h"}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def segment_lengths_native(index: pd.DatetimeIndex, bar_interval: pd.Timedelta) -> list:
    """Local equivalent of gap_aware.segment_lengths() -- see module
    docstring bug #2. Same maximal-contiguous-run logic, parameterized by
    the caller's own bar spacing instead of a hardcoded 5-minute constant."""
    if len(index) == 0:
        return []
    diffs = index.to_series().diff()
    new_segment = (diffs != bar_interval)
    new_segment.iloc[0] = True
    segment_id = new_segment.cumsum()
    return segment_id.value_counts().sort_index().tolist()


# ===========================================================================
# 1. Genuine 15m/30m candle construction (Section 3/4)
# ===========================================================================

def load_raw_5m() -> pd.DataFrame:
    log("Loading raw 5m OHLCV + order-flow columns...")
    df = pd.read_csv(RAW_OHLCV_PATH,
                      usecols=["timestamp", "open", "high", "low", "close", "volume", "avg_price",
                               "buy_vol", "sell_vol"],
                      parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    diffs = df["timestamp"].diff().dropna().unique()
    assert len(diffs) == 1 and pd.Timedelta(diffs[0]) == pd.Timedelta(minutes=5), (
        f"raw 5m data is not perfectly contiguous (found diffs {diffs}) -- candle aggregation "
        f"below assumes contiguity and must not silently fabricate across a real gap"
    )
    df = df.set_index("timestamp")
    log(f"  {len(df):,} contiguous 5m bars, {df.index.min()} -> {df.index.max()}")
    return df


def build_agg_candles(raw5m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Genuine tf-minute candles from real 5m OHLCV -- see module docstring."""
    minutes = TF_MINUTES[tf]
    rule = f"{minutes}min"
    g = raw5m.resample(rule, label="left", closed="left")
    n_per_bucket = g["close"].count()
    expected = TF_BARS_PER_5M[tf]
    complete = n_per_bucket == expected
    n_incomplete = int((~complete).sum())

    agg = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "buy_vol": g["buy_vol"].sum(),
        "sell_vol": g["sell_vol"].sum(),
    })
    quote_vol = (raw5m["avg_price"] * raw5m["volume"]).resample(rule, label="left", closed="left").sum()
    agg["avg_price"] = (quote_vol / agg["volume"]).where(agg["volume"] > 0, agg["close"])
    agg["ofi"] = agg["buy_vol"] - agg["sell_vol"]
    agg = agg[complete].copy()
    log(f"  {tf}: {len(agg):,} genuine candles built from {len(raw5m):,} 5m bars "
        f"({n_incomplete} incomplete edge bucket(s) dropped, not fabricated)")
    return agg


# ===========================================================================
# 2. Feature construction (Section 5/6/7)
# ===========================================================================

def build_tf_features(agg: pd.DataFrame, tf: str) -> pd.DataFrame:
    horizon_bars = RETURN_HORIZON_BARS[tf]
    close = agg["close"]
    log_close = np.log(close)
    ret1 = log_close.diff(1)

    feats = {}
    for label, nbars in horizon_bars.items():
        feats[f"ret_{label}"] = log_close.diff(nbars)

    vol_horizon_bars = {k: v for k, v in horizon_bars.items() if v > 1}
    for label, nbars in vol_horizon_bars.items():
        feats[f"vol_{label}"] = ret1.rolling(nbars).std()

    ema_fast = close.ewm(span=EMA_FAST_BARS, adjust=False, min_periods=EMA_FAST_BARS).mean()
    ema_slow = close.ewm(span=EMA_SLOW_BARS, adjust=False, min_periods=EMA_SLOW_BARS).mean()
    feats["ema_dist_fast"] = np.log(close / ema_fast)
    feats["ema_dist_slow"] = np.log(close / ema_slow)
    feats["ema_slope_fast"] = np.log(ema_fast / ema_fast.shift(EMA_SLOPE_LAG_BARS))

    volume = agg["volume"]
    vol_change = _winsorize(_safe_div(volume - volume.shift(1), volume.shift(1)))
    roll_1h = horizon_bars["1h"]
    mean_v = volume.rolling(roll_1h).mean()
    std_v = volume.rolling(roll_1h).std()
    feats["vol_change"] = vol_change
    feats["vol_zscore"] = _safe_div(volume - mean_v, std_v)

    ofi = agg["ofi"]
    ofi_mean = ofi.rolling(roll_1h).mean()
    ofi_std = ofi.rolling(roll_1h).std()
    feats["ofi_raw"] = _winsorize(ofi)
    feats["ofi_zscore"] = _winsorize(_safe_div(ofi - ofi_mean, ofi_std))

    skew_label = SKEW_WINDOW_LABEL[tf]
    roll_skew = horizon_bars[skew_label]
    feats[f"skew_{skew_label}"] = ret1.rolling(roll_skew).skew()
    up = ret1.clip(lower=0)
    down = ret1.clip(upper=0).abs()
    avg_up = up.rolling(roll_skew).mean()
    avg_down = down.rolling(roll_skew).mean()
    feats["updown_asymmetry"] = _winsorize(_safe_div(avg_up, avg_up + avg_down))

    return pd.DataFrame(feats, index=agg.index)


def build_validity_mask_native(agg: pd.DataFrame, feature_index, max_lookback_bars):
    """Own-timeframe adaptation of gap_aware.py's zero-volume-touch concept
    (re-applied at native 15m/30m bar spacing, not reusing 5m zero flags
    directly, and NOT calling the buggy build_validity_mask -- see bug #1)."""
    is_zero_vol = (agg["volume"] == 0)
    touched = is_zero_vol.rolling(max_lookback_bars, min_periods=1).max()
    valid = (touched == 0)
    valid = valid.reindex(feature_index).fillna(False)
    return valid


def characterize_state_tf(state_mean: pd.Series, tf: str):
    return_cols = RETURN_COLS_TF[tf]
    vol_cols = VOL_COLS_TF[tf]
    trend_score = float(state_mean[return_cols].mean())
    vol_score = float(state_mean[vol_cols].mean())
    if trend_score > TREND_Z_THRESHOLD:
        trend = "Uptrend"
    elif trend_score < -TREND_Z_THRESHOLD:
        trend = "Downtrend"
    else:
        trend = "Ranging"
    if vol_score > VOL_Z_THRESHOLD:
        vol = "High-Vol"
    elif vol_score < -VOL_Z_THRESHOLD:
        vol = "Low-Vol"
    else:
        vol = "Mid-Vol"
    return f"{trend} / {vol}", trend_score, vol_score


# ===========================================================================
# 3. HMM fit (identical architecture/hyperparameters at every timeframe)
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


# ===========================================================================
# 4. Segments / transitions / stats (Section 13/14/15) -- clock-time based
# ===========================================================================

def build_segments(states: np.ndarray, seg_id: np.ndarray, timestamps: pd.DatetimeIndex, tf_minutes: float) -> pd.DataFrame:
    n = len(states)
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = (states[1:] != states[:-1]) | (seg_id[1:] != seg_id[:-1])
    run_id = np.cumsum(changed)
    tmp = pd.DataFrame({"state": states, "run_id": run_id, "timestamp": timestamps})
    segs = tmp.groupby("run_id").agg(state=("state", "first"), start_timestamp=("timestamp", "first"),
                                      end_timestamp=("timestamp", "last"), number_of_bars=("timestamp", "size")
                                      ).reset_index(drop=True)
    segs["duration_minutes"] = segs["number_of_bars"] * tf_minutes
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


def compute_model_stats(config, tf_label, n_states, feat_cols, n_valid, total_days, segments, transitions,
                         best_ll, best_seed, fit_time, decode_time, regime_labels, transmat):
    dur = segments["duration_minutes"]
    bars = segments["number_of_bars"]
    n_transitions = len(transitions)
    return {
        "config": config, "timeframe": tf_label, "n_states_model": n_states, "n_features": len(feat_cols),
        "feature_columns": feat_cols, "n_valid_bars": n_valid, "total_days": total_days,
        "n_transitions": n_transitions, "n_segments": len(segments),
        "transitions_per_1k_bars": n_transitions / n_valid * 1000,
        "transitions_per_day": n_transitions / total_days if total_days > 0 else None,
        "mean_duration_minutes": float(dur.mean()), "median_duration_minutes": float(dur.median()),
        "min_duration_minutes": float(dur.min()), "max_duration_minutes": float(dur.max()),
        "mean_segment_bars": float(bars.mean()), "median_segment_bars": float(bars.median()),
        "min_segment_bars": int(bars.min()), "max_segment_bars": int(bars.max()),
        "pct_1bar": float((bars == 1).mean() * 100),
        "pct_gt30m": float((dur > 30).mean() * 100), "pct_gt1h": float((dur > 60).mean() * 100),
        "pct_gt2h": float((dur > 120).mean() * 100), "pct_gt4h": float((dur > 240).mean() * 100),
        "pct_gt8h": float((dur > 480).mean() * 100),
        "n_gt1h": int((dur > 60).sum()), "n_gt2h": int((dur > 120).sum()),
        "n_gt4h": int((dur > 240).sum()), "n_gt8h": int((dur > 480).sum()),
        "train_ll": best_ll, "train_ll_per_obs": (best_ll / n_valid) if best_ll is not None else None,
        "best_seed": best_seed, "fit_time_s": fit_time, "decode_time_s": decode_time,
        "distinct_regime_labels": len(set(regime_labels)),
        "regime_labels": {int(s): r for s, r in enumerate(regime_labels)},
        "transmat": transmat,
    }


def future_return_stats(states, forward_rets_by_horizon, n_states, regime_labels):
    rows = []
    for state in range(n_states):
        mask = states == state
        for h_name, fr in forward_rets_by_horizon.items():
            vals = fr[mask]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            rows.append({
                "state": state, "regime": regime_labels[state], "horizon": h_name, "n": len(vals),
                "mean_ret": float(np.mean(vals)), "median_ret": float(np.median(vals)),
                "std_ret": float(np.std(vals)), "p_positive": float((vals > 0).mean()),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 5. Per-timeframe pipeline: build features -> validity -> scale -> fit both
#    HMM-4 and HMM-6 -> decode -> stats -> save
# ===========================================================================

def run_timeframe_config(tf: str, agg: pd.DataFrame, n_states: int, checkpoint: dict):
    tag = f"{tf}_hmm{n_states}"
    if tag in checkpoint.get("completed", {}):
        log(f"  {tag}: already checkpointed, skipping")
        return checkpoint["completed"][tag]

    feat_df = build_tf_features(agg, tf)
    max_lookback = max(EMA_SLOW_BARS, max(RETURN_HORIZON_BARS[tf].values()))
    valid_flag = build_validity_mask_native(agg, feat_df.index, max_lookback)
    valid_mask = valid_flag & feat_df.notna().all(axis=1)
    valid_index = feat_df.index[valid_mask]
    valid_values = feat_df.loc[valid_mask].values
    bar_interval = pd.Timedelta(minutes=TF_MINUTES[tf])
    lengths = segment_lengths_native(valid_index, bar_interval)
    assert sum(lengths) == len(valid_values)
    seg_id = persistence_common.segment_ids_from_lengths(lengths)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(valid_values)

    t0 = time.time()
    model, best_ll, best_seed, restart_log = fit_hmm(scaled, lengths, n_states)
    fit_time = time.time() - t0
    if model is None:
        raise RuntimeError(f"{tag}: all restarts failed -- {restart_log}")

    t1 = time.time()
    states = model.predict(scaled, lengths=lengths)
    posteriors = model.predict_proba(scaled, lengths=lengths)
    decode_time = time.time() - t1
    confidence = posteriors[np.arange(len(states)), states]

    feat_cols = list(feat_df.columns)
    means_df = pd.DataFrame(model.means_, columns=feat_cols)
    labels_scores = [characterize_state_tf(means_df.iloc[s], tf) for s in range(n_states)]
    regime_labels = [x[0] for x in labels_scores]
    trend_scores = np.array([x[1] for x in labels_scores])
    vol_scores = np.array([x[2] for x in labels_scores])

    segments = build_segments(states, seg_id, valid_index, TF_MINUTES[tf])
    segments["regime"] = segments["state"].map(dict(enumerate(regime_labels)))
    transitions = build_transitions(segments)

    n_valid = len(valid_index)
    total_days = (valid_index.max() - valid_index.min()).total_seconds() / 86400

    close_full = agg["close"].reindex(valid_index)
    log_close = np.log(close_full.values)
    fwd_by_h = {}
    for h_name, h_bars in FUTURE_HORIZON_BARS[tf].items():
        j = np.arange(n_valid) + h_bars
        fwd_valid = (j < n_valid) & (seg_id[np.minimum(j, n_valid - 1)] == seg_id)
        fr = np.full(n_valid, np.nan)
        fr[fwd_valid] = log_close[j[fwd_valid]] - log_close[np.arange(n_valid)[fwd_valid]]
        fwd_by_h[h_name] = fr
    future_ret_df = future_return_stats(states, fwd_by_h, n_states, regime_labels)

    state_stats_rows = []
    for s in range(n_states):
        sub = segments[segments["state"] == s]
        n_bars_state = int((states == s).sum())
        state_stats_rows.append({
            "timeframe": tf, "n_states_model": n_states, "state": s, "regime": regime_labels[s],
            "trend_score": float(trend_scores[s]), "vol_score": float(vol_scores[s]),
            "n_segments": len(sub), "n_bars": n_bars_state, "pct_bars": n_bars_state / n_valid * 100,
            "mean_duration_minutes": float(sub["duration_minutes"].mean()) if len(sub) else None,
            "median_duration_minutes": float(sub["duration_minutes"].median()) if len(sub) else None,
            "max_duration_minutes": float(sub["duration_minutes"].max()) if len(sub) else None,
            "stay_prob": float(model.transmat_[s, s]),
        })

    model_stats = compute_model_stats(tf.upper(), tf, n_states, feat_cols, n_valid, total_days,
                                       segments, transitions, best_ll, best_seed, fit_time, decode_time,
                                       regime_labels, model.transmat_.tolist())

    tag_out = f"{tf}_hmm{n_states}"
    with open(os.path.join(RESULTS_DIR, f"model_{tag_out}.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_columns": feat_cols, "timeframe": tf}, f)
    regime_df = pd.DataFrame({"timestamp": valid_index, "state": states, "regime": [regime_labels[s] for s in states],
                               "confidence": confidence})
    regime_df.to_parquet(os.path.join(RESULTS_DIR, f"regime_{tag_out}.parquet"), index=False)
    pd.DataFrame(state_stats_rows).to_csv(os.path.join(RESULTS_DIR, f"state_stats_{tag_out}.csv"), index=False)
    segments.to_csv(os.path.join(RESULTS_DIR, f"segments_{tag_out}.csv"), index=False)
    future_ret_df.to_csv(os.path.join(RESULTS_DIR, f"future_returns_{tag_out}.csv"), index=False)
    json.dump(model_stats, open(os.path.join(RESULTS_DIR, f"model_stats_{tag_out}.json"), "w"), indent=2, default=str)
    pd.DataFrame(restart_log).to_csv(os.path.join(RESULTS_DIR, f"restarts_{tag_out}.csv"), index=False)

    log(f"  {tag_out}: n_valid={n_valid:,} transitions={len(transitions):,} "
        f"({model_stats['transitions_per_1k_bars']:.2f}/1k bars, {model_stats['transitions_per_day']:.2f}/day) "
        f"median_dur={model_stats['median_duration_minutes']:.0f}min "
        f"LL/obs={model_stats['train_ll_per_obs']:.4f} distinct_labels={model_stats['distinct_regime_labels']}/{n_states}")

    result = {"model_stats": model_stats, "state_stats": state_stats_rows,
              "trend_scores": trend_scores.tolist(), "vol_scores": vol_scores.tolist(),
              "regime_labels": regime_labels}
    checkpoint.setdefault("completed", {})[tag] = result
    json.dump(checkpoint, open(os.path.join(RESULTS_DIR, "v22_checkpoint.json"), "w"), indent=2, default=str)
    return result


# ===========================================================================
# 6. A baseline (5m) -- decode existing model_A_HMM4.pkl/model_A_HMM6.pkl only
# ===========================================================================

def run_5m_baseline(n_states: int, df_5m, model_cols_5m):
    from gap_aware import valid_values_and_lengths  # 5m-native, unaffected by either disclosed bug
    tag_out = f"5m_hmm{n_states}"
    path = os.path.join(RESULTS_DIR, f"model_A_HMM{n_states}.pkl")
    log(f"  Loading existing (NOT refit) {path}")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    model, scaler, feat_cols = artifact["model"], artifact["scaler"], artifact["feature_columns"]

    valid_df, lengths, valid_index = valid_values_and_lengths(df_5m, model_cols_5m)
    assert feat_cols == model_cols_5m
    scaled = scaler.transform(valid_df.values)
    states = model.predict(scaled, lengths=lengths)
    posteriors = model.predict_proba(scaled, lengths=lengths)
    confidence = posteriors[np.arange(len(states)), states]
    seg_id = persistence_common.segment_ids_from_lengths(lengths)

    from infer_regime import RETURN_COLS, VOL_COLS, characterize_state
    means_df = pd.DataFrame(model.means_, columns=feat_cols)
    regime_labels = [characterize_state(means_df.iloc[s]) for s in range(n_states)]

    segments = build_segments(states, seg_id, valid_index, 5.0)
    segments["regime"] = segments["state"].map(dict(enumerate(regime_labels)))
    transitions = build_transitions(segments)
    n_valid = len(valid_index)
    total_days = (valid_index.max() - valid_index.min()).total_seconds() / 86400

    model_stats = compute_model_stats("A", "5m", n_states, feat_cols, n_valid, total_days, segments, transitions,
                                       artifact.get("train_log_likelihood") if "train_log_likelihood" in artifact else None,
                                       artifact.get("selected_seed"), 0.0, 0.0, regime_labels, model.transmat_.tolist())

    state_stats_rows = []
    for s in range(n_states):
        sub = segments[segments["state"] == s]
        n_bars_state = int((states == s).sum())
        trend_score = float(means_df.iloc[s][[c for c in RETURN_COLS if c in means_df.columns]].mean())
        vol_score = float(means_df.iloc[s][[c for c in VOL_COLS if c in means_df.columns]].mean())
        state_stats_rows.append({
            "timeframe": "5m", "n_states_model": n_states, "state": s, "regime": regime_labels[s],
            "trend_score": trend_score, "vol_score": vol_score,
            "n_segments": len(sub), "n_bars": n_bars_state, "pct_bars": n_bars_state / n_valid * 100,
            "mean_duration_minutes": float(sub["duration_minutes"].mean()) if len(sub) else None,
            "median_duration_minutes": float(sub["duration_minutes"].median()) if len(sub) else None,
            "max_duration_minutes": float(sub["duration_minutes"].max()) if len(sub) else None,
            "stay_prob": float(model.transmat_[s, s]),
        })

    regime_df = pd.DataFrame({"timestamp": valid_index, "state": states, "regime": [regime_labels[s] for s in states],
                               "confidence": confidence})
    regime_df.to_parquet(os.path.join(RESULTS_DIR, f"regime_{tag_out}.parquet"), index=False)
    pd.DataFrame(state_stats_rows).to_csv(os.path.join(RESULTS_DIR, f"state_stats_{tag_out}.csv"), index=False)
    segments.to_csv(os.path.join(RESULTS_DIR, f"segments_{tag_out}.csv"), index=False)
    json.dump(model_stats, open(os.path.join(RESULTS_DIR, f"model_stats_{tag_out}.json"), "w"), indent=2, default=str)

    log(f"  {tag_out} (decode-only, not refit): n_valid={n_valid:,} transitions={len(transitions):,} "
        f"({model_stats['transitions_per_1k_bars']:.2f}/1k bars, {model_stats['transitions_per_day']:.2f}/day) "
        f"median_dur={model_stats['median_duration_minutes']:.0f}min distinct_labels={model_stats['distinct_regime_labels']}/{n_states}")

    return {"model_stats": model_stats, "state_stats": state_stats_rows,
            "regime_labels": regime_labels, "states": states, "confidence": confidence,
            "valid_index": valid_index}


# ===========================================================================
# 7. D config -- 15m+30m weighted MTF score (Section 11/12), no new HMM fit
# ===========================================================================

def build_mtf_score(n_states: int, res_15m: dict, res_30m: dict, agg_15m: pd.DataFrame, agg_30m: pd.DataFrame):
    tag15 = f"15m_hmm{n_states}"
    tag30 = f"30m_hmm{n_states}"

    r15 = pd.read_parquet(os.path.join(RESULTS_DIR, f"regime_{tag15}.parquet"))
    r30 = pd.read_parquet(os.path.join(RESULTS_DIR, f"regime_{tag30}.parquet"))

    trend15 = np.array(res_15m["trend_scores"])
    trend30 = np.array(res_30m["trend_scores"])

    r15 = r15.copy()
    r15["trend_score"] = trend15[r15["state"].values]
    r15["score_weighted"] = r15["trend_score"] * r15["confidence"]

    r30 = r30.copy()
    r30["trend_score"] = trend30[r30["state"].values]
    r30["score_weighted"] = r30["trend_score"] * r30["confidence"]
    # Section 12: a 30m candle at timestamp t0 (label="left") covers
    # [t0, t0+30m) and is only fully known at t0+30m -- its score becomes
    # "available" at t0+30m, never before.
    r30["available_at"] = r30["timestamp"] + pd.Timedelta(minutes=30)

    r15_sorted = r15.sort_values("timestamp")
    r30_sorted = r30[["available_at", "trend_score", "score_weighted"]].rename(
        columns={"trend_score": "trend_score_30m", "score_weighted": "score_weighted_30m"}).sort_values("available_at")

    merged = pd.merge_asof(r15_sorted, r30_sorted, left_on="timestamp", right_on="available_at",
                            direction="backward")
    merged = merged.dropna(subset=["trend_score_30m"]).reset_index(drop=True)

    merged["direction_score"] = 0.6 * merged["trend_score"] + 0.4 * merged["trend_score_30m"]
    merged["mtf_score"] = 0.6 * merged["score_weighted"] + 0.4 * merged["score_weighted_30m"]
    merged["regime_bin"] = np.where(merged["direction_score"] > TREND_Z_THRESHOLD, "Bullish",
                                     np.where(merged["direction_score"] < -TREND_Z_THRESHOLD, "Bearish", "Ranging"))
    merged["n_states_model"] = n_states
    return merged[["timestamp", "n_states_model", "state", "trend_score", "trend_score_30m", "direction_score",
                   "mtf_score", "regime_bin"]]


def analyze_mtf(merged: pd.DataFrame, agg_15m: pd.DataFrame, n_states: int):
    bin_map = {"Bearish": 0, "Ranging": 1, "Bullish": 2}
    codes = merged["regime_bin"].map(bin_map).values
    # NOTE: pd.DatetimeIndex(merged["timestamp"].values) (i.e. calling
    # .values first) silently DROPS the UTC tz info -- verified directly:
    # produces datetime64[us] (tz-naive) instead of datetime64[us, UTC].
    # reindex()ing agg_15m (tz-aware) against a tz-naive index then matches
    # ZERO rows (pandas treats tz-naive/tz-aware as incomparable, not an
    # error, just silent all-NaN) -- found by an empty
    # mtf_score_correlation_D_hmm*.csv after the first full run, root-caused
    # and fixed here (dropping .values preserves the tz), not silently
    # patched over.
    idx = pd.DatetimeIndex(merged["timestamp"])
    lengths = segment_lengths_native(idx, pd.Timedelta(minutes=15))
    assert sum(lengths) == len(codes)
    seg_id = persistence_common.segment_ids_from_lengths(lengths)
    segments = build_segments(codes, seg_id, idx, 15.0)
    inv_map = {v: k for k, v in bin_map.items()}
    segments["regime"] = segments["state"].map(inv_map)
    transitions = build_transitions(segments)

    n_valid = len(codes)
    total_days = (idx.max() - idx.min()).total_seconds() / 86400
    empirical_transmat = np.zeros((3, 3))
    for i in range(3):
        from_mask = codes[:-1] == i
        if from_mask.sum() > 0:
            for j in range(3):
                empirical_transmat[i, j] = (codes[1:][from_mask] == j).mean()

    model_stats = compute_model_stats("D", "15m+30m", n_states, ["direction_score(0.6x15m+0.4x30m)"], n_valid,
                                       total_days, segments, transitions, None, None, 0.0, 0.0,
                                       [inv_map[i] for i in range(3)], empirical_transmat.tolist())
    model_stats["note"] = ("D has no fitted HMM -- transmat above is EMPIRICAL (observed bin-to-bin transition "
                            "frequencies of the 3-way Bullish/Ranging/Bearish direction_score bin), not a model "
                            "parameter. train_ll/best_seed are null because nothing was fit.")

    # future-return coherence (Section 14) on the continuous mtf_score, plus per-bin
    close15 = agg_15m["close"].reindex(idx)
    log_close = np.log(close15.values)
    fwd_by_h = {}
    for h_name, h_bars in RETURN_HORIZON_BARS["15m"].items():
        j = np.arange(n_valid) + h_bars
        fwd_valid = (j < n_valid) & (seg_id[np.minimum(j, n_valid - 1)] == seg_id)
        fr = np.full(n_valid, np.nan)
        fr[fwd_valid] = log_close[j[fwd_valid]] - log_close[np.arange(n_valid)[fwd_valid]]
        fwd_by_h[h_name] = fr
    future_ret_df = future_return_stats(codes, fwd_by_h, 3, [inv_map[i] for i in range(3)])

    corr_rows = []
    for h_name, fr in fwd_by_h.items():
        mask = ~np.isnan(fr)
        if mask.sum() > 10:
            corr = float(np.corrcoef(merged["mtf_score"].values[mask], fr[mask])[0, 1])
            corr_rows.append({"horizon": h_name, "n": int(mask.sum()), "pearson_corr_mtf_score_vs_fwd_ret": corr})
    corr_df = pd.DataFrame(corr_rows)

    return model_stats, segments, transitions, future_ret_df, corr_df


# ===========================================================================
# main
# ===========================================================================

def main():
    t0 = time.time()
    checkpoint_path = os.path.join(RESULTS_DIR, "v22_checkpoint.json")
    checkpoint = json.load(open(checkpoint_path)) if os.path.exists(checkpoint_path) else {}

    raw5m = load_raw_5m()
    log("Building genuine 15m and 30m candles from raw 5m OHLCV...")
    agg_15m = build_agg_candles(raw5m, "15m")
    agg_30m = build_agg_candles(raw5m, "30m")

    # --- benchmark (Section 10), then proceed automatically ---
    log("Benchmarking 1 restart each of 15m-HMM6 and 30m-HMM6 (largest of the 4 new fits)...")
    bench_times = {}
    for tf, agg in [("15m", agg_15m), ("30m", agg_30m)]:
        feat_df = build_tf_features(agg, tf)
        max_lookback = max(EMA_SLOW_BARS, max(RETURN_HORIZON_BARS[tf].values()))
        valid_flag = build_validity_mask_native(agg, feat_df.index, max_lookback)
        valid_mask = valid_flag & feat_df.notna().all(axis=1)
        valid_values = feat_df.loc[valid_mask].values
        lengths = segment_lengths_native(feat_df.index[valid_mask], pd.Timedelta(minutes=TF_MINUTES[tf]))
        scaled = StandardScaler().fit_transform(valid_values)
        tb0 = time.time()
        _fit_one(scaled, lengths, 6, RANDOM_STATE)
        bench_times[tf] = time.time() - tb0
        log(f"  {tf}: {len(valid_values):,} valid rows, 1 restart of HMM-6 took {bench_times[tf]:.1f}s")

    est_total = sum(bench_times.values()) * N_RESTARTS * 1.75  # 4 jobs total, HMM4~0.75x HMM6, some overhead
    log(f"ESTIMATED total runtime for all 4 new fits (15m/30m x HMM4/HMM6, {N_RESTARTS} restarts each): "
        f"~{est_total:.0f}s (~{est_total/60:.1f} min). Proceeding automatically now.")

    jobs = [(tf, n) for tf in ["15m", "30m"] for n in N_STATES_LIST]
    results = {}
    for tf, n_states in tqdm(jobs, desc="B/C: 15m & 30m HMM fits"):
        agg = agg_15m if tf == "15m" else agg_30m
        results[f"{tf}_hmm{n_states}"] = run_timeframe_config(tf, agg, n_states, checkpoint)

    log("Decoding A (5m) baseline from existing model_A_HMM4.pkl / model_A_HMM6.pkl (NOT refit)...")
    df_5m, model_cols_5m = load_masked_features()
    for n_states in N_STATES_LIST:
        results[f"5m_hmm{n_states}"] = run_5m_baseline(n_states, df_5m, model_cols_5m)

    log("Building D (15m+30m weighted MTF score)...")
    mtf_frames = []
    d_summaries = {}
    for n_states in N_STATES_LIST:
        merged = build_mtf_score(n_states, results[f"15m_hmm{n_states}"], results[f"30m_hmm{n_states}"],
                                  agg_15m, agg_30m)
        mtf_frames.append(merged)
        model_stats, segments, transitions, future_ret_df, corr_df = analyze_mtf(merged, agg_15m, n_states)
        d_summaries[n_states] = {"model_stats": model_stats, "future_ret": future_ret_df, "corr": corr_df}
        segments.to_csv(os.path.join(RESULTS_DIR, f"segments_D_hmm{n_states}.csv"), index=False)
        json.dump(model_stats, open(os.path.join(RESULTS_DIR, f"model_stats_D_hmm{n_states}.json"), "w"),
                   indent=2, default=str)
        future_ret_df.to_csv(os.path.join(RESULTS_DIR, f"future_returns_D_hmm{n_states}.csv"), index=False)
        corr_df.to_csv(os.path.join(RESULTS_DIR, f"mtf_score_correlation_D_hmm{n_states}.csv"), index=False)
        log(f"  D-HMM{n_states}: n_valid={model_stats['n_valid_bars']:,} transitions={model_stats['n_transitions']:,} "
            f"({model_stats['transitions_per_1k_bars']:.2f}/1k bars, {model_stats['transitions_per_day']:.2f}/day) "
            f"median_dur={model_stats['median_duration_minutes']:.0f}min")

    pd.concat(mtf_frames, ignore_index=True).to_parquet(os.path.join(RESULTS_DIR, "mtf_15m_30m_60_40.parquet"),
                                                          index=False)

    # save full (unmasked) feature tables per Section 19
    for tf, agg in [("15m", agg_15m), ("30m", agg_30m)]:
        feat_df = build_tf_features(agg, tf)
        max_lookback = max(EMA_SLOW_BARS, max(RETURN_HORIZON_BARS[tf].values()))
        valid_flag = build_validity_mask_native(agg, feat_df.index, max_lookback)
        out = feat_df.copy()
        out["valid"] = (valid_flag & feat_df.notna().all(axis=1)).values
        out.to_parquet(os.path.join(RESULTS_DIR, f"features_{tf}.parquet"))

    runtime = time.time() - t0
    log(f"Total runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    all_stats = {k: v["model_stats"] for k, v in results.items()}
    for n_states in N_STATES_LIST:
        all_stats[f"D_hmm{n_states}"] = d_summaries[n_states]["model_stats"]
    json.dump({"model_stats": all_stats, "runtime_s": runtime, "bench_times": bench_times},
               open(os.path.join(RESULTS_DIR, "v22_final_results.json"), "w"), indent=2, default=str)
    log(f"Done. All artifacts saved to {RESULTS_DIR}")
    log("EXIT_MARKER_V22_DONE")


if __name__ == "__main__":
    main()
