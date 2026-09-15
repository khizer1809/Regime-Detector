"""
v9_trend_features.py -- Step 6: new TREND-specific feature construction.
Pure cache-replay: derives everything from Data/features_out.csv's already-
cached ret_5m/ret_15m/ret_1h columns (no raw OHLCV refetch, no HMM
computation). A relative log-price proxy is reconstructed via a per-segment
cumulative sum of ret_5m (ret_5m = log(close).diff(1), so its cumsum is
log(close) up to an additive constant -- irrelevant for structure features,
which only ever compare relative levels). Reset to 0 at the start of each
gap-aware segment so no structure computation spans a genuine data gap.

All rolling/backward-looking computations are grouped by segment id (same
discipline as v3_adaptive_target.py's volatility calculation) -- a window
can never reach across a gap.

FEATURE DICTIONARY (12 features, 6 families, ~2 per family):

Family 1 -- MARKET STRUCTURE (window W=48 bars=4h)
  structure_score       : sign(recent_W_high - prior_W_high) + sign(recent_W_low - prior_W_low)
                           range [-2,+2]. +2 = clean HH+HL (uptrend structure),
                           -2 = clean LH+LL (downtrend structure), 0 = mixed/chop.
                           Lookback: 2*W=96 bars. No future confirmation needed
                           (both windows are backward-looking, non-overlapping).
  dist_from_swing_high  : log_price - rolling_W_max(log_price). Always <=0.
                           Lookback: W=48 bars. Contemporaneous, no delay.
  dist_from_swing_low   : log_price - rolling_W_min(log_price). Always >=0.
                           Lookback: W=48 bars. Contemporaneous, no delay.

Family 2 -- DIRECTIONAL EFFICIENCY (Kaufman efficiency ratio)
  efficiency_1h  : |price[t]-price[t-12]| / sum(|diff| over the 12 bars). Range [0,1].
                    Lookback: 12 bars. Contemporaneous.
  efficiency_4h  : same, 48-bar window. Lookback: 48 bars. Contemporaneous.

Family 3 -- TREND ACCELERATION
  accel_short  : (mean bar-return over trailing 12 bars) - (mean bar-return over
                  the PRIOR non-overlapping 12 bars). Lookback: 24 bars. No delay.
  accel_medium : same construction, 48-bar windows (96 bars lookback).

Family 4 -- MULTI-TIMEFRAME ALIGNMENT
  mtf_alignment_score : sign(ret_5m)+sign(ret_15m)+sign(ret_1h), range [-3,+3].
                         Lookback: 12 bars (ret_1h's own window). Contemporaneous
                         (ret_5m/15m/1h are already-existing causal columns).
  mtf_conflict_flag   : 1 if sign(ret_5m) != sign(ret_1h), else 0. Lookback: 12 bars.

Family 5 -- TREND AGE / MATURITY (price-structure based, NOT HMM duration)
  log1p_bars_since_structure_flip : log1p(bars since structure_score's SIGN
      last changed). A running counter on structure_score's sign, same
      reset-on-change mechanism as persistence_common.compute_running_duration
      but driven by price structure, not the HMM's decoded state. Lookback:
      unbounded backward within the segment. No delay -- purely backward.

Family 6 -- TREND DETERIORATION
  efficiency_deterioration : efficiency_1h - efficiency_1h shifted back 12 bars
      (is directional cleanliness declining vs an hour ago). Lookback: 24 bars.
  alignment_deterioration  : mtf_alignment_score - mtf_alignment_score shifted
      back 12 bars (is timeframe agreement declining vs an hour ago).
      Lookback: 24 bars.

Every feature rejected from this set for using future information, duplicating
an existing feature, or requiring unresolved future confirmation: none were
included in the first place (fractal/pivot-style swing points requiring
future bars to confirm were deliberately replaced with backward-only rolling
max/min, per the module's own "reject or delay" instruction -- backward-only
was chosen to avoid a confirmation-delay design entirely).

Run: `python v9_trend_features.py` (from src/) to build and cache the feature
table. Expected: well under a minute.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_aware import valid_values_and_lengths
from save_production_model import load_masked_features
import persistence_common

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "Data", "v2_cache", "trend_threshold_experiment", "trend_features")
W_STRUCT = 48
W_EFF_SHORT = 12
W_EFF_LONG = 48
W_ACCEL_SHORT = 12
W_ACCEL_MED = 48
W_MTF = 12  # matches ret_1h's own window, just for documentation

FEATURE_FAMILIES = {
    "family1_structure": ["structure_score", "dist_from_swing_high", "dist_from_swing_low"],
    "family2_efficiency": ["efficiency_1h", "efficiency_4h"],
    "family3_acceleration": ["accel_short", "accel_medium"],
    "family4_mtf_alignment": ["mtf_alignment_score", "mtf_conflict_flag"],
    "family5_trend_age": ["log1p_bars_since_structure_flip"],
    "family6_deterioration": ["efficiency_deterioration", "alignment_deterioration"],
}
ALL_NEW_FEATURES = [f for fs in FEATURE_FAMILIES.values() for f in fs]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def seg_rolling(series: pd.Series, seg_id: np.ndarray, window: int, func: str):
    """Rolling computation grouped by segment id -- never crosses a gap."""
    s = pd.Series(series.values, index=pd.RangeIndex(len(series)))
    g = s.groupby(seg_id)
    if func == "max":
        out = g.rolling(window, min_periods=window).max()
    elif func == "min":
        out = g.rolling(window, min_periods=window).min()
    elif func == "sum_abs_diff":
        out = g.apply(lambda x: x.diff().abs().rolling(window, min_periods=window).sum())
    else:
        raise ValueError(func)
    out.index = out.index.droplevel(0)
    return out.reindex(s.index).values


def build_features():
    log("Loading already-cached masked feature file (no HMM computation)...")
    df, model_cols = load_masked_features()
    valid_df, lengths, valid_index = valid_values_and_lengths(df, model_cols)
    seg_id = persistence_common.segment_ids_from_lengths(lengths)
    n = len(valid_df)
    log(f"  {n:,} valid rows, {len(lengths)} segments")

    ret5 = valid_df["ret_5m"].values
    ret15 = valid_df["ret_15m"].values
    ret1h = valid_df["ret_1h"].values

    # relative log-price proxy, reset to 0 at each segment start (first bar's
    # ret_5m may span a gap, so it's excluded from the cumsum)
    first_in_seg = np.empty(n, dtype=bool)
    first_in_seg[0] = True
    first_in_seg[1:] = seg_id[1:] != seg_id[:-1]
    ret5_clean = ret5.copy()
    ret5_clean[first_in_seg] = 0.0
    log_price = pd.Series(ret5_clean).groupby(seg_id).cumsum().values

    out = pd.DataFrame(index=valid_index)

    # --- Family 1: structure ---
    recent_high = seg_rolling(pd.Series(log_price), seg_id, W_STRUCT, "max")
    recent_low = seg_rolling(pd.Series(log_price), seg_id, W_STRUCT, "min")
    prior_high = pd.Series(recent_high).groupby(seg_id).shift(W_STRUCT).values
    prior_low = pd.Series(recent_low).groupby(seg_id).shift(W_STRUCT).values
    out["structure_score"] = np.sign(recent_high - prior_high) + np.sign(recent_low - prior_low)
    out["dist_from_swing_high"] = log_price - recent_high
    out["dist_from_swing_low"] = log_price - recent_low

    # --- Family 2: efficiency ---
    def efficiency(window):
        net = pd.Series(log_price).groupby(seg_id).diff(window).abs().values
        total = seg_rolling(pd.Series(log_price), seg_id, window, "sum_abs_diff")
        return np.where(total > 0, net / total, np.nan)
    out["efficiency_1h"] = efficiency(W_EFF_SHORT)
    out["efficiency_4h"] = efficiency(W_EFF_LONG)

    # --- Family 3: acceleration ---
    def accel(window):
        recent_mean = pd.Series(ret5).groupby(seg_id).rolling(window, min_periods=window).mean()
        recent_mean.index = recent_mean.index.droplevel(0)
        recent_mean = recent_mean.reindex(pd.RangeIndex(n)).values
        prior_mean = pd.Series(recent_mean).groupby(seg_id).shift(window).values
        return recent_mean - prior_mean
    out["accel_short"] = accel(W_ACCEL_SHORT)
    out["accel_medium"] = accel(W_ACCEL_MED)

    # --- Family 4: multi-timeframe alignment ---
    s5, s15, s1h = np.sign(ret5), np.sign(ret15), np.sign(ret1h)
    out["mtf_alignment_score"] = s5 + s15 + s1h
    out["mtf_conflict_flag"] = (s5 != s1h).astype(float)

    # --- Family 5: trend age (price-structure based) ---
    struct_sign = np.sign(out["structure_score"].values)
    struct_sign_filled = pd.Series(struct_sign).fillna(0).values
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = (struct_sign_filled[1:] != struct_sign_filled[:-1]) | (seg_id[1:] != seg_id[:-1])
    run_id = np.cumsum(changed)
    bars_since_flip = pd.Series(run_id).groupby(run_id).cumcount().values + 1
    out["log1p_bars_since_structure_flip"] = np.log1p(bars_since_flip)

    # --- Family 6: deterioration ---
    eff1h_shifted = pd.Series(out["efficiency_1h"].values).groupby(seg_id).shift(W_EFF_SHORT).values
    out["efficiency_deterioration"] = out["efficiency_1h"].values - eff1h_shifted
    align_shifted = pd.Series(out["mtf_alignment_score"].values).groupby(seg_id).shift(W_EFF_SHORT).values
    out["alignment_deterioration"] = out["mtf_alignment_score"].values - align_shifted

    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    feats = build_features()
    runtime = time.time() - t0
    log(f"Built {len(ALL_NEW_FEATURES)} new features across {len(FEATURE_FAMILIES)} families in {runtime:.1f}s")

    out_path = os.path.join(OUT_DIR, "trend_features.csv")
    feats.to_csv(out_path, index=False)
    log(f"Saved to {out_path}")

    print("\nFeature dictionary summary:")
    for fam, cols in FEATURE_FAMILIES.items():
        print(f"  {fam}: {cols}")
    print("\nNaN counts (expected: nonzero only in the warm-up rows at the start of each segment):")
    print(feats[ALL_NEW_FEATURES].isna().sum().to_string())
    print("\nDescribe:")
    print(feats[ALL_NEW_FEATURES].describe().to_string())

    json.dump({"families": FEATURE_FAMILIES, "windows": {
        "W_STRUCT": W_STRUCT, "W_EFF_SHORT": W_EFF_SHORT, "W_EFF_LONG": W_EFF_LONG,
        "W_ACCEL_SHORT": W_ACCEL_SHORT, "W_ACCEL_MED": W_ACCEL_MED},
        "runtime_s": runtime}, open(os.path.join(OUT_DIR, "feature_dictionary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
