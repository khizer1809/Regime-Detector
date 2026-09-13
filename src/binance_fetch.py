"""
binance_fetch.py -- pulls 5-minute BTCUSDT candles from Binance's public
REST klines endpoint (no API key required) and maps them onto the raw
column schema features.py expects: open, high, low, close, volume,
avg_price, buy_vol, sell_vol, ofi.

IMPORTANT CAVEAT, read before trusting buy_vol/sell_vol/ofi/avg_price:
The original historical raw data file this project was built on (whatever
generated features_out.csv / features_out_masked.csv) is no longer
available on this machine, and how it computed avg_price/buy_vol/sell_vol/
ofi was never documented in the codebase. Binance's basic public klines
endpoint does NOT provide those fields directly -- it provides
`taker_buy_base_asset_volume` (aggressive buy volume) and
`quote_asset_volume` (turnover), from which this module derives PROXIES:

    avg_price = quote_asset_volume / volume          (candle VWAP)
    buy_vol   = taker_buy_base_asset_volume           (aggressive buys)
    sell_vol  = volume - taker_buy_base_asset_volume  (aggressive sells)
    ofi       = buy_vol - sell_vol                    (order-flow imbalance)

These are standard, defensible proxies, but there is no guarantee they
match whatever methodology produced the original historical columns of
the same name. open/high/low/close/volume are exact (Binance's own
authoritative values, the same source verify_zero_volume_bars.py already
cross-checked this project's history against).

Binance's public klines endpoint serves FULL history (not just recent
data), so this module can fetch any date range, including the lookback
context refresh_features.py needs to correctly compute rolling features
at the seam where old and new data meet.
"""

import time

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
BAR_MS = 5 * 60 * 1000
MAX_RETRIES = 3
REQUEST_PAUSE_SEC = 0.15  # polite pacing, well under Binance's public rate limit


def _fetch_klines_page(start_ms: int, end_ms: int = None, limit: int = 1000) -> list:
    params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": start_ms, "limit": limit}
    if end_ms is not None:
        params["endTime"] = end_ms
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BINANCE_URL, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    raise RuntimeError(f"Binance klines request failed after {MAX_RETRIES} retries "
                        f"(start_ms={start_ms}, end_ms={end_ms})")


def _klines_to_raw_df(klines: list) -> pd.DataFrame:
    rows = []
    for k in klines:
        open_ms, o, h, l, c, vol = k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        quote_vol, n_trades, taker_buy_base = float(k[7]), int(k[8]), float(k[9])
        avg_price = quote_vol / vol if vol > 0 else (o + h + l + c) / 4
        buy_vol = taker_buy_base
        sell_vol = vol - taker_buy_base
        rows.append({
            "timestamp": pd.Timestamp(open_ms, unit="ms", tz="UTC"),
            "open": o, "high": h, "low": l, "close": c, "volume": vol,
            "avg_price": avg_price, "buy_vol": buy_vol, "sell_vol": sell_vol,
            "ofi": buy_vol - sell_vol,
        })
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


def fetch_range(start: pd.Timestamp, end: pd.Timestamp = None) -> pd.DataFrame:
    """Fetches every 5m candle from `start` through `end` (default: now),
    paging past Binance's 1000-candle-per-request limit. Returns a raw
    DataFrame with columns open/high/low/close/volume/avg_price/buy_vol/
    sell_vol/ofi, indexed by UTC timestamp, ready for
    features.build_feature_matrix()."""
    end = end or pd.Timestamp.now(tz="UTC")
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_klines = []
    cursor = start_ms
    while cursor < end_ms:
        page = _fetch_klines_page(cursor, end_ms, limit=1000)
        if not page:
            break
        all_klines.extend(page)
        last_open_ms = page[-1][0]
        next_cursor = last_open_ms + BAR_MS
        if next_cursor <= cursor:  # safety against an infinite loop on a malformed response
            break
        cursor = next_cursor
        time.sleep(REQUEST_PAUSE_SEC)

    if not all_klines:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume",
                                      "avg_price", "buy_vol", "sell_vol", "ofi"])
    return _klines_to_raw_df(all_klines)


def fetch_recent(n_bars: int) -> pd.DataFrame:
    """Fetches the most recent n_bars 5-minute candles (a single request if
    n_bars <= 1000)."""
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(minutes=5 * (n_bars + 5))  # small buffer
    df = fetch_range(start, end)
    return df.iloc[-n_bars:] if len(df) > n_bars else df


if __name__ == "__main__":
    df = fetch_recent(10)
    print(f"Fetched {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(df.to_string())
