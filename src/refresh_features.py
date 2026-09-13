"""
refresh_features.py -- appends newly-available bars to
Data/features_out_masked.csv by fetching new raw candles from Binance
(binance_fetch.py) and running them through the exact same pipeline that
built the file in the first place: features.build_feature_matrix(dropna=
False) + gap_aware.py's zero-volume/gap validity mask.

Does NOT rebuild history from scratch -- reads only the last timestamp
already in the file (via a byte-seek, not a full 372MB load), fetches a
lookback-buffered window from Binance from there through "now" (enough
context for every rolling feature -- the longest lookback is 48 bars/4h --
to be correctly computed at the seam with no artificial NaN), computes
features + validity over that window, and appends ONLY the genuinely new
rows (strictly after the file's last timestamp) to the CSV.

The raw source file this project originally used is gone from this
machine; nothing here modifies or requires it. See binance_fetch.py's
docstring for the caveat on avg_price/buy_vol/sell_vol/ofi being proxies
derived from Binance's public klines fields, not a guaranteed match to
whatever produced the original historical columns of the same name.

Run: `python refresh_features.py` (from src/).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # features.py at project root

from binance_fetch import fetch_range
from features import build_feature_matrix
from gap_aware import MAX_LOOKBACK_BARS, build_validity_mask

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASKED_PATH = os.path.join(_ROOT, "Data", "features_out_masked.csv")

BAR_INTERVAL = pd.Timedelta(minutes=5)
LOOKBACK_BUFFER_BARS = MAX_LOOKBACK_BARS + 12  # +1h margin beyond the strict minimum


def _get_existing_columns(path: str) -> list:
    """Reads just the header line to get the EXACT column set/order already
    on disk (18 model columns + `valid`, session columns excluded) -- so
    appended rows are guaranteed to match, rather than hardcoding a second
    copy of that list that could drift out of sync."""
    with open(path, "r") as f:
        header = f.readline().strip()
    return header.split(",")[1:]  # drop the leading "timestamp" index column name


def _get_last_timestamp(path: str) -> pd.Timestamp:
    """Reads only the last line of the CSV (byte-seek from the end) to get
    the most recent timestamp already on disk, without loading the whole
    (300MB+) file."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        chunk_size = 4096
        data = b""
        pos = file_size
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            data = f.read(read_size) + data
            if data.count(b"\n") >= 2:  # last line + enough to be sure it's complete
                break
        lines = data.strip().split(b"\n")
        last_line = lines[-1].decode("utf-8")
    ts_str = last_line.split(",")[0]
    return pd.Timestamp(ts_str)


def refresh():
    if not os.path.exists(MASKED_PATH):
        raise FileNotFoundError(
            f"{MASKED_PATH} does not exist -- this script only appends to an existing file, "
            f"it does not bootstrap history from scratch."
        )

    existing_cols = _get_existing_columns(MASKED_PATH)
    last_ts = _get_last_timestamp(MASKED_PATH)
    print(f"Last timestamp on disk: {last_ts}", flush=True)
    print(f"Existing schema: {existing_cols}", flush=True)

    fetch_start = last_ts - BAR_INTERVAL * LOOKBACK_BUFFER_BARS
    print(f"Fetching from Binance: {fetch_start} -> now (includes {LOOKBACK_BUFFER_BARS}-bar "
          f"lookback buffer so rolling features are correct at the seam)...", flush=True)
    raw = fetch_range(fetch_start)
    if raw.empty:
        print("No data returned from Binance -- nothing to do.", flush=True)
        return

    print(f"Fetched {len(raw)} raw bars, {raw.index.min()} -> {raw.index.max()}", flush=True)

    new_raw = raw[raw.index > last_ts]
    if len(new_raw) == 0:
        print("No bars newer than what's already on disk -- nothing to append.", flush=True)
        return

    feats = build_feature_matrix(raw, dropna=False)
    valid = build_validity_mask(raw, feats.index)
    feats["valid"] = valid.values
    feats = feats[existing_cols]  # match the on-disk schema exactly (drops session columns)

    new_feats = feats.loc[feats.index > last_ts]
    if len(new_feats) == 0:
        print("Fetched raw bars produced no new feature rows -- nothing to append.", flush=True)
        return

    with open(MASKED_PATH, "a", newline="") as f:
        new_feats.to_csv(f, header=False)

    print(f"Appended {len(new_feats)} new rows to:\n  {MASKED_PATH}", flush=True)
    print(f"New range on disk now ends at: {new_feats.index.max()}", flush=True)
    print(f"Valid (usable for HMM fitting/inference): {int(new_feats['valid'].sum())}/{len(new_feats)}", flush=True)


if __name__ == "__main__":
    refresh()
