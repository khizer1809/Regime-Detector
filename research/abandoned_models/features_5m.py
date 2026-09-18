"""
features.py -- feature engineering for the BTCUSDT 5m regime detector.

Builds a numeric feature matrix from raw 5-minute OHLCV + order-flow bars,
grouped into these feature families:

    returns      (6)  ret_5m, ret_15m, ret_30m, ret_1h, ret_2h, ret_4h
    volatility   (3)  vol_15m, vol_1h, vol_2h
    volume/flow  (6)  vol_change, vol_zscore, buy_sell_ratio, ofi_raw,
                       ofi_zscore, vwap_dist
    skew         (3)  skew_1h, skew_4h, updown_asymmetry
    session      (7)  hour_sin, hour_cos, session_asia, session_europe,
                       session_us, session_asia_europe_overlap,
                       session_europe_us_overlap

Total: 25 columns.

vwap_dist is measured against a rolling 1h volume-weighted average price
(from avg_price/volume), not the raw all-time-cumulative vwap column, since
the latter is non-stationary and tracks BTC's multi-year appreciation
rather than any regime signal.

vol_change, ofi_raw, ofi_zscore, and updown_asymmetry are winsorized at the
1st/99th percentile of the full column so outliers don't distort a
downstream StandardScaler fit; that clip should be computed per
walk-forward training window once that pipeline exists, rather than over
the whole column as is done here.
"""

import numpy as np
import pandas as pd


BARS_PER = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "2h": 24,
    "4h": 48,
}


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns over 5m/15m/30m/1h/2h/4h horizons."""
    price = np.log(df["close"])
    return pd.DataFrame({
        "ret_5m": price.diff(BARS_PER["5m"]),
        "ret_15m": price.diff(BARS_PER["15m"]),
        "ret_30m": price.diff(BARS_PER["30m"]),
        "ret_1h": price.diff(BARS_PER["1h"]),
        "ret_2h": price.diff(BARS_PER["2h"]),
        "ret_4h": price.diff(BARS_PER["4h"]),
    }, index=df.index)


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:

    close = df["close"]
    returns = np.log(close).diff(1)

    vol_15m = returns.rolling(BARS_PER["15m"]).std()
    vol_1h = returns.rolling(BARS_PER["1h"]).std()
    vol_2h = returns.rolling(BARS_PER["2h"]).std()

    return pd.DataFrame({
        "vol_15m": vol_15m,
        "vol_1h": vol_1h,
        "vol_2h": vol_2h,
    }, index=df.index)


def add_volume_orderflow(df: pd.DataFrame) -> pd.DataFrame:
    """Volume change, volume z-score, buy/sell ratio, OFI (rolling-normalized), rolling-VWAP distance."""
    volume = df["volume"]
    buy = df["buy_vol"]
    sell = df["sell_vol"]
    ofi = df["ofi"]
    close = df["close"]
    avg_price = df["avg_price"]

    vol_change = _safe_divide(volume - volume.shift(1), volume.shift(1))
    vol_change = _winsorize(vol_change)

    roll_n = BARS_PER["1h"]
    mean = volume.rolling(roll_n).mean()
    std = volume.rolling(roll_n).std()
    vol_zscore = _safe_divide(volume - mean, std)

    buy_sell_ratio = _safe_divide(buy, buy + sell)

    ofi_mean = ofi.rolling(roll_n).mean()
    ofi_std = ofi.rolling(roll_n).std()
    ofi_zscore = _safe_divide(ofi - ofi_mean, ofi_std)
    ofi_zscore = _winsorize(ofi_zscore)

    ofi_raw = _winsorize(ofi)

    # Rolling 1h VWAP from avg_price/volume, not the raw (all-time-cumulative,
    # non-stationary) vwap column.
    rolling_vwap = _safe_divide((avg_price * volume).rolling(roll_n).sum(), volume.rolling(roll_n).sum())
    vwap_dist = (close - rolling_vwap) / close

    return pd.DataFrame({
        "vol_change": vol_change,
        "vol_zscore": vol_zscore,
        "buy_sell_ratio": buy_sell_ratio,
        "ofi_raw": ofi_raw,
        "ofi_zscore": ofi_zscore,
        "vwap_dist": vwap_dist,
    }, index=df.index)


def add_skew(df: pd.DataFrame) -> pd.DataFrame:

    returns = np.log(df["close"]).diff(1)

    skew_1h = returns.rolling(BARS_PER["1h"]).skew()
    skew_4h = returns.rolling(BARS_PER["4h"]).skew()

    up = returns.clip(lower=0)
    down = returns.clip(upper=0).abs()
    roll_n = BARS_PER["1h"]
    avg_up = up.rolling(roll_n).mean()
    avg_down = down.rolling(roll_n).mean()
    # Fraction of the window's movement that was upward, in [0, 1]. A plain
    # avg_up/avg_down ratio goes to NaN (via _safe_divide) whenever a window
    # has zero down-moves, which happens routinely during clean trends and
    # was silently dropping those rows out of the feature matrix.
    asymmetry = _safe_divide(avg_up, avg_up + avg_down)
    asymmetry = _winsorize(asymmetry)

    return pd.DataFrame({
        "skew_1h": skew_1h,
        "skew_4h": skew_4h,
        "updown_asymmetry": asymmetry,
    }, index=df.index)


def add_session(df: pd.DataFrame) -> pd.DataFrame:

    hour = df.index.hour + df.index.minute / 60.0

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    asia = (hour >= 0) & (hour < 9)
    europe = (hour >= 7) & (hour < 16)
    us = (hour >= 12) & (hour < 21)

    return pd.DataFrame({
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "session_asia": asia.astype(int),
        "session_europe": europe.astype(int),
        "session_us": us.astype(int),
        "session_asia_europe_overlap": (asia & europe).astype(int),
        "session_europe_us_overlap": (europe & us).astype(int),
    }, index=df.index)


def build_feature_matrix(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:

    required = {"open", "high", "low", "close", "volume", "avg_price", "buy_vol", "sell_vol", "ofi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")

    parts = [
        add_returns(df),
        add_volatility(df),
        add_volume_orderflow(df),
        add_skew(df),
        add_session(df),
    ]
    features = pd.concat(parts, axis=1)

    if dropna:
        features = features.dropna()

    return features


if __name__ == "__main__":
    INPUT_PATH = r"C:\Users\MohammedkhezerK\Downloads\New System\ICT_STRUCTURE_RESEARCH\data\BTCUSDT_5M.csv"
    OUTPUT_PATH = r"C:\Users\MohammedkhezerK\Regime Detector\Data\features_out.csv"
 
    raw = pd.read_csv(INPUT_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.set_index("timestamp").sort_index()
 
    features = build_feature_matrix(raw)
 
    print(features.shape)
    print(features.head())
 
    features.to_csv(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")