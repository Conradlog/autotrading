"""
Download BTC perpetual futures data from Binance Futures API.
Produces parquet files compatible with prepare.py's load_market_data().

Usage:
    uv run binance_data.py                  # download and merge
    uv run binance_data.py --start 2022-01-01
    uv run binance_data.py --no-merge       # download only, don't merge with HyperLiquid
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BINANCE_FAPI = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrading", "data")
DEFAULT_START = "2022-01-01"
RATE_LIMIT_SLEEP = 0.25  # seconds between API calls


# ---------------------------------------------------------------------------
# Candle download
# ---------------------------------------------------------------------------

def download_binance_klines(start_date: str, end_date: str = None) -> pd.DataFrame:
    """Download 1h klines from Binance Futures API with pagination."""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000) if end_date is None else \
             int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_candles = []
    cursor = start_ts
    batch = 0

    print(f"Downloading Binance BTC-PERP 1h candles from {start_date}...")

    while cursor < end_ts:
        batch += 1
        params = {
            "symbol": SYMBOL,
            "interval": "1h",
            "startTime": cursor,
            "endTime": end_ts,
            "limit": 1500,
        }

        for attempt in range(3):
            try:
                resp = requests.get(f"{BINANCE_FAPI}/fapi/v1/klines", params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  ERROR after 3 attempts: {e}")
                    raise
                time.sleep(2)

        if not data:
            break

        all_candles.extend(data)
        last_ts = data[-1][0]
        cursor = last_ts + 3600_000  # next hour

        n_days = len(all_candles) / 24
        print(f"  Batch {batch}: {len(all_candles)} candles ({n_days:.0f} days)", end="\r")
        time.sleep(RATE_LIMIT_SLEEP)

    print(f"  Total: {len(all_candles)} candles ({len(all_candles)/24:.0f} days)      ")

    if not all_candles:
        return pd.DataFrame()

    # Map to prepare.py format
    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    result = pd.DataFrame({
        "timestamp": pd.to_datetime(df["open_time"], unit="ms", utc=True),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "num_trades": df["num_trades"].astype(int),
    })

    result = result.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Funding rate download
# ---------------------------------------------------------------------------

def download_binance_funding(start_date: str, end_date: str = None) -> pd.DataFrame:
    """Download funding rates from Binance Futures API with pagination."""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000) if end_date is None else \
             int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_funding = []
    cursor = start_ts
    batch = 0

    print(f"Downloading Binance BTC-PERP funding rates from {start_date}...")

    while cursor < end_ts:
        batch += 1
        params = {
            "symbol": SYMBOL,
            "startTime": cursor,
            "limit": 1000,
        }

        for attempt in range(3):
            try:
                resp = requests.get(f"{BINANCE_FAPI}/fapi/v1/fundingRate", params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  ERROR after 3 attempts: {e}")
                    raise
                time.sleep(2)

        if not data:
            break

        all_funding.extend(data)
        last_ts = data[-1]["fundingTime"]
        cursor = last_ts + 1

        print(f"  Batch {batch}: {len(all_funding)} entries", end="\r")
        time.sleep(RATE_LIMIT_SLEEP)

    print(f"  Total: {len(all_funding)} funding entries                    ")

    if not all_funding:
        return pd.DataFrame()

    df = pd.DataFrame(all_funding)

    result = pd.DataFrame({
        "timestamp": pd.to_datetime(df["fundingTime"], unit="ms", utc=True),
        "funding_rate": df["fundingRate"].astype(float),
        "premium": 0.0,  # Binance doesn't expose premium in this endpoint
    })

    result = result.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Merge with HyperLiquid data
# ---------------------------------------------------------------------------

def merge_with_hyperliquid(binance_candles: pd.DataFrame, binance_funding: pd.DataFrame):
    """
    Merge Binance + HyperLiquid data, preferring HyperLiquid where overlapping.
    Saves as BTC_candles_extended.parquet and BTC_funding_extended.parquet.
    """
    hl_candle_path = os.path.join(CACHE_DIR, "BTC_candles.parquet")
    hl_funding_path = os.path.join(CACHE_DIR, "BTC_funding.parquet")

    # Candles
    if os.path.exists(hl_candle_path):
        hl_candles = pd.read_parquet(hl_candle_path)
        hl_candles["timestamp"] = pd.to_datetime(hl_candles["timestamp"], utc=True)

        # Use Binance for dates before HyperLiquid starts
        hl_start = hl_candles["timestamp"].min()
        binance_before_hl = binance_candles[binance_candles["timestamp"] < hl_start]

        merged_candles = pd.concat([binance_before_hl, hl_candles], ignore_index=True)
        merged_candles = merged_candles.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

        print(f"Candles: {len(binance_before_hl)} Binance + {len(hl_candles)} HyperLiquid = {len(merged_candles)} total")
    else:
        merged_candles = binance_candles
        print(f"Candles: {len(merged_candles)} Binance only (no HyperLiquid data found)")

    # Funding
    if os.path.exists(hl_funding_path):
        hl_funding = pd.read_parquet(hl_funding_path)
        hl_funding["timestamp"] = pd.to_datetime(hl_funding["timestamp"], utc=True)

        hl_start = hl_funding["timestamp"].min()
        binance_fund_before = binance_funding[binance_funding["timestamp"] < hl_start]

        merged_funding = pd.concat([binance_fund_before, hl_funding], ignore_index=True)
        merged_funding = merged_funding.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

        print(f"Funding: {len(binance_fund_before)} Binance + {len(hl_funding)} HyperLiquid = {len(merged_funding)} total")
    else:
        merged_funding = binance_funding
        print(f"Funding: {len(merged_funding)} Binance only")

    # Save
    candle_out = os.path.join(CACHE_DIR, "BTC_candles_extended.parquet")
    funding_out = os.path.join(CACHE_DIR, "BTC_funding_extended.parquet")

    merged_candles.to_parquet(candle_out, index=False)
    merged_funding.to_parquet(funding_out, index=False)

    days = len(merged_candles) / 24
    first = merged_candles["timestamp"].iloc[0]
    last = merged_candles["timestamp"].iloc[-1]
    print(f"\nExtended dataset saved:")
    print(f"  {candle_out}")
    print(f"  {funding_out}")
    print(f"  Range: {first} → {last} ({days:.0f} days)")

    return merged_candles, merged_funding


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance BTC-PERP data")
    parser.add_argument("--start", default=DEFAULT_START, help=f"Start date (default: {DEFAULT_START})")
    parser.add_argument("--end", default=None, help="End date (default: now)")
    parser.add_argument("--no-merge", action="store_true", help="Don't merge with HyperLiquid data")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Download
    candles = download_binance_klines(args.start, args.end)
    funding = download_binance_funding(args.start, args.end)

    # Save raw Binance data
    candles.to_parquet(os.path.join(CACHE_DIR, "BTC_candles_binance.parquet"), index=False)
    funding.to_parquet(os.path.join(CACHE_DIR, "BTC_funding_binance.parquet"), index=False)
    print(f"\nBinance raw data saved to {CACHE_DIR}")

    # Merge
    if not args.no_merge:
        print()
        merge_with_hyperliquid(candles, funding)

    print("\nDone!")
