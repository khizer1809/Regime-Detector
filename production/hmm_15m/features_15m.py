"""
15m feature engineering for the production HMM-4 regime classifier.

Extracted verbatim (same formulas, same constants) from the validated
research pipeline (research/timeframe_comparison/v22_fast_timeframe_screening.py)
so production does not depend on the research tree at runtime. No feature
definitions, thresholds, or windows were changed during this extraction.

Pipeline: raw 5m OHLCV -> genuine 15m candles -> 18 features -> gap-aware
validity mask. All rolling windows are backward-looking only; no feature
here uses information from t+1 or later.
"""

import numpy as np
import pandas as pd

RAW_OHLCV_PATH = r"C:\Users\MohammedkhezerK\Downloads\New System\ICT_STRUCTURE_RESEARCH\data\BTCUSDT_5M.csv"

TF_MINUTES = 15
TF_BARS_PER_5M = 3

# Economic-time horizons (bars), not raw 5m bar counts carried over unscaled.
RETURN_HORIZON_BARS = {"15m": 1, "30m": 2, "1h": 4, "2h": 8, "4h": 16}
RETURN_COLS = ["ret_15m", "ret_30m", "ret_1h", "ret_2h", "ret_4h"]
VOL_COLS = ["vol_30m", "vol_1h", "vol_2h"]  # 3 shortest windows, matches production's VOL_COLS convention

EMA_FAST_BARS = 8
EMA_SLOW_BARS = 24
EMA_SLOPE_LAG_BARS = 3

# Rolling skew is undefined for windows < 3 bars (bias-corrected formula
# divides by n-2). "2h" (8 bars) is the shortest window in this timeframe's
# own horizon ladder wide enough to define it.
SKEW_WINDOW_LABEL = "2h"

MAX_LOOKBACK_BARS = max(EMA_SLOW_BARS, max(RETURN_HORIZON_BARS.values()))


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def segment_lengths(index: pd.DatetimeIndex, bar_interval: pd.Timedelta = pd.Timedelta(minutes=15)) -> list:
    """Length of each maximal contiguous (15-minute-spaced) run in `index`."""
    if len(index) == 0:
        return []
    diffs = index.to_series().diff()
    new_segment = diffs != bar_interval
    new_segment.iloc[0] = True
    segment_id = new_segment.cumsum()
    return segment_id.value_counts().sort_index().tolist()


def load_raw_5m(path: str = RAW_OHLCV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "avg_price",
                                     "buy_vol", "sell_vol"], parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    diffs = df["timestamp"].diff().dropna().unique()
    assert len(diffs) == 1 and pd.Timedelta(diffs[0]) == pd.Timedelta(minutes=5), (
        f"raw 5m data is not contiguous (found diffs {diffs}) -- candle aggregation below "
        f"assumes contiguity and must not fabricate across a real gap"
    )
    return df.set_index("timestamp")


def build_15m_candles(raw5m: pd.DataFrame) -> pd.DataFrame:
    """Genuine 15m candles from real 5m OHLCV. A bucket is only kept if it
    contains all 3 underlying 5m bars -- a partial edge bucket is dropped,
    never fabricated from incomplete data."""
    g = raw5m.resample("15min", label="left", closed="left")
    complete = g["close"].count() == TF_BARS_PER_5M

    agg = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
        "buy_vol": g["buy_vol"].sum(), "sell_vol": g["sell_vol"].sum(),
    })
    quote_vol = (raw5m["avg_price"] * raw5m["volume"]).resample("15min", label="left", closed="left").sum()
    agg["avg_price"] = (quote_vol / agg["volume"]).where(agg["volume"] > 0, agg["close"])
    agg["ofi"] = agg["buy_vol"] - agg["sell_vol"]
    return agg[complete].copy()


def build_features(agg: pd.DataFrame) -> pd.DataFrame:
    close = agg["close"]
    log_close = np.log(close)
    ret1 = log_close.diff(1)

    feats = {f"ret_{label}": log_close.diff(nbars) for label, nbars in RETURN_HORIZON_BARS.items()}
    for label, nbars in RETURN_HORIZON_BARS.items():
        if nbars > 1:
            feats[f"vol_{label}"] = ret1.rolling(nbars).std()

    ema_fast = close.ewm(span=EMA_FAST_BARS, adjust=False, min_periods=EMA_FAST_BARS).mean()
    ema_slow = close.ewm(span=EMA_SLOW_BARS, adjust=False, min_periods=EMA_SLOW_BARS).mean()
    feats["ema_dist_fast"] = np.log(close / ema_fast)
    feats["ema_dist_slow"] = np.log(close / ema_slow)
    feats["ema_slope_fast"] = np.log(ema_fast / ema_fast.shift(EMA_SLOPE_LAG_BARS))

    volume = agg["volume"]
    feats["vol_change"] = _winsorize(_safe_div(volume - volume.shift(1), volume.shift(1)))
    roll_1h = RETURN_HORIZON_BARS["1h"]
    feats["vol_zscore"] = _safe_div(volume - volume.rolling(roll_1h).mean(), volume.rolling(roll_1h).std())

    ofi = agg["ofi"]
    feats["ofi_raw"] = _winsorize(ofi)
    feats["ofi_zscore"] = _winsorize(_safe_div(ofi - ofi.rolling(roll_1h).mean(), ofi.rolling(roll_1h).std()))

    roll_skew = RETURN_HORIZON_BARS[SKEW_WINDOW_LABEL]
    feats[f"skew_{SKEW_WINDOW_LABEL}"] = ret1.rolling(roll_skew).skew()
    up, down = ret1.clip(lower=0), ret1.clip(upper=0).abs()
    avg_up, avg_down = up.rolling(roll_skew).mean(), down.rolling(roll_skew).mean()
    feats["updown_asymmetry"] = _winsorize(_safe_div(avg_up, avg_up + avg_down))

    return pd.DataFrame(feats, index=agg.index)


def build_validity_mask(agg: pd.DataFrame, feature_index, max_lookback_bars: int = MAX_LOOKBACK_BARS) -> pd.Series:
    """A bar is invalid if a zero-volume (no-trading) candle touches its own
    rolling lookback window. Timeframe-native -- does not reuse 5m zero-volume
    flags directly."""
    is_zero_vol = agg["volume"] == 0
    touched = is_zero_vol.rolling(max_lookback_bars, min_periods=1).max()
    return (touched == 0).reindex(feature_index).fillna(False)


def build_feature_table(raw5m: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (features, valid) for the full history in `raw5m`."""
    agg = build_15m_candles(raw5m)
    feats = build_features(agg)
    valid = build_validity_mask(agg, feats.index) & feats.notna().all(axis=1)
    return feats, valid
